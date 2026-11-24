"""
Tests for the AI analysis REST API.

Covers 12 scenarios:
 1.  Unauthorized request (401)
 2.  Insufficient role (403) — data_scientist cannot trigger analysis
 3.  Analyze on nonexistent case (404)
 4.  Analyze with image that belongs to a different case (404)
 5.  Analyze when image has no processed_path (422)
 6.  Successful analysis — returns 200 and expected fields
 7.  GET /analysis — 404 when no analysis exists yet
 8.  GET /analysis — returns latest result after analysis
 9.  GET /overlay — 404 when no analysis exists
10.  GET /overlay — returns PNG bytes after analysis
11.  GET /overlay — specific analysis_id
12.  GET /overlay — dynamic generation when overlay file is deleted
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

# ------------------------------------------------------------------
# Environment must be set BEFORE importing app modules
# This mirrors conftest.py's pattern; conftest sets these for the
# shared fixtures, but we guard here for standalone execution.
# ------------------------------------------------------------------
_TMP = tempfile.mkdtemp(prefix="artery-analysis-tests-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")
os.environ.setdefault("STORAGE_DIR", f"{_TMP}/storage")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("FIRST_ADMIN_EMAIL", "admin@example.com")
os.environ.setdefault("FIRST_ADMIN_PASSWORD", "adminpass123")

import numpy as np
import pytest
import SimpleITK as sitk
from fastapi.testclient import TestClient

from app.db.session import Base, engine
from app.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as c:
        yield c


def _login(client, email, password):
    r = client.post("/api/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _register(client, email, role="researcher"):
    r = client.post(
        "/api/auth/register",
        json={"email": email, "full_name": "Test User", "password": "password123", "role": role},
    )
    assert r.status_code == 201, r.text
    return _login(client, email, "password123")


@pytest.fixture()
def admin(client):
    return _login(client, "admin@example.com", "adminpass123")


@pytest.fixture()
def clinician(client):
    return _register(client, "clin@example.com", "clinician")


@pytest.fixture()
def scientist(client):
    return _register(client, "sci@example.com", "data_scientist")


def _make_nifti(tmp_path: Path, shape=(48, 64, 64)) -> Path:
    """Create a small synthetic CT NIfTI with a bright tube (≥32 voxels per axis)."""
    arr = np.full(shape, -1000, dtype=np.int16)
    arr[:, 24:32, 24:32] = 400
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 1.0))
    path = tmp_path / "scan.nii.gz"
    sitk.WriteImage(img, str(path))
    return path


@pytest.fixture()
def uploaded_image(client, clinician, tmp_path):
    """Patient → Case → uploaded+preprocessed Image.  Returns (case_id, image_id)."""
    # Patient
    r = client.post("/api/patients", json={"patient_code": "P-ANALYSIS-001"}, headers=clinician)
    assert r.status_code == 201, r.text
    patient_id = r.json()["id"]

    # Case
    r = client.post(
        "/api/cases",
        json={"case_code": "CASE-ANALYSIS-001", "patient_id": patient_id},
        headers=clinician,
    )
    assert r.status_code == 201, r.text
    case_id = r.json()["id"]

    # Upload NIfTI
    nifti = _make_nifti(tmp_path)
    with open(nifti, "rb") as fh:
        r = client.post(
            f"/api/cases/{case_id}/images",
            files={"file": ("scan.nii.gz", fh, "application/octet-stream")},
            headers=clinician,
        )
    assert r.status_code == 201, r.text
    image_id = r.json()["id"]
    return case_id, image_id


@pytest.fixture()
def other_case(client, clinician):
    """A second case that owns no images — used for wrong-case tests."""
    r = client.post("/api/patients", json={"patient_code": "P-ANALYSIS-002"}, headers=clinician)
    patient_id = r.json()["id"]
    r = client.post(
        "/api/cases",
        json={"case_code": "CASE-ANALYSIS-002", "patient_id": patient_id},
        headers=clinician,
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------
# 1. Unauthorized (no token)
# ---------------------------------------------------------------------------

def test_analyze_unauthorized(client, uploaded_image):
    case_id, image_id = uploaded_image
    r = client.post(f"/api/cases/{case_id}/analyze", json={"image_id": image_id})
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# 2. Insufficient role — data_scientist cannot trigger analysis
# ---------------------------------------------------------------------------

def test_analyze_forbidden_role(client, scientist, uploaded_image):
    case_id, image_id = uploaded_image
    r = client.post(
        f"/api/cases/{case_id}/analyze",
        json={"image_id": image_id, "model_name": "unet"},
        headers=scientist,
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# 3. Case not found
# ---------------------------------------------------------------------------

def test_analyze_case_not_found(client, clinician, uploaded_image):
    _, image_id = uploaded_image
    r = client.post(
        "/api/cases/99999/analyze",
        json={"image_id": image_id},
        headers=clinician,
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 4. Image belongs to a different case
# ---------------------------------------------------------------------------

def test_analyze_wrong_image_for_case(client, clinician, uploaded_image, other_case):
    _, image_id = uploaded_image
    r = client.post(
        f"/api/cases/{other_case}/analyze",
        json={"image_id": image_id},
        headers=clinician,
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 5. Image has no processed_path (simulate by patching the DB record)
# ---------------------------------------------------------------------------

def test_analyze_no_processed_path(client, clinician, uploaded_image):
    from sqlalchemy.orm import Session
    from app.db.session import engine as db_engine
    from app.models.models import Image

    case_id, image_id = uploaded_image

    # Temporarily blank the processed_path
    with Session(db_engine) as sess:
        img = sess.get(Image, image_id)
        original_path = img.processed_path
        img.processed_path = None
        sess.commit()

    try:
        r = client.post(
            f"/api/cases/{case_id}/analyze",
            json={"image_id": image_id},
            headers=clinician,
        )
        assert r.status_code == 422
    finally:
        # Restore
        with Session(db_engine) as sess:
            img = sess.get(Image, image_id)
            img.processed_path = original_path
            sess.commit()


# ---------------------------------------------------------------------------
# 6. Successful analysis
# ---------------------------------------------------------------------------

def test_analyze_success(client, clinician, uploaded_image):
    case_id, image_id = uploaded_image
    r = client.post(
        f"/api/cases/{case_id}/analyze",
        json={"image_id": image_id, "model_name": "attention_unet"},
        headers=clinician,
    )
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["case_id"] == case_id
    assert data["image_id"] == image_id
    assert data["model_name"] == "attention_unet"
    assert data["status"] in ("success", "partial", "failed")
    assert data["slice_index"] is not None
    assert "inference_duration_ms" in data
    assert "disclaimer" in data
    assert "RESEARCH" in data["disclaimer"].upper()

    # Case status should now be 'analyzed' (or 'failed')
    r2 = client.get(f"/api/cases/{case_id}", headers=clinician)
    assert r2.json()["status"] in ("analyzed", "failed")


# ---------------------------------------------------------------------------
# 7. GET /analysis — 404 when no analysis exists
# ---------------------------------------------------------------------------

def test_get_analysis_not_found(client, clinician, other_case):
    r = client.get(f"/api/cases/{other_case}/analysis", headers=clinician)
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 8. GET /analysis — returns latest result after analysis
# ---------------------------------------------------------------------------

def test_get_analysis_after_analyze(client, clinician, uploaded_image):
    case_id, image_id = uploaded_image
    client.post(
        f"/api/cases/{case_id}/analyze",
        json={"image_id": image_id},
        headers=clinician,
    )
    r = client.get(f"/api/cases/{case_id}/analysis", headers=clinician)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["case_id"] == case_id
    assert data["image_id"] == image_id


# ---------------------------------------------------------------------------
# 9. GET /overlay — 404 when no analysis exists
# ---------------------------------------------------------------------------

def test_overlay_not_found(client, clinician, uploaded_image):
    _, image_id = uploaded_image
    r = client.get(f"/api/images/{image_id}/overlay", headers=clinician)
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 10. GET /overlay — returns PNG after analysis
# ---------------------------------------------------------------------------

def test_overlay_after_analysis(client, clinician, uploaded_image):
    case_id, image_id = uploaded_image
    client.post(
        f"/api/cases/{case_id}/analyze",
        json={"image_id": image_id},
        headers=clinician,
    )
    r = client.get(f"/api/images/{image_id}/overlay", headers=clinician)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 100   # non-trivial PNG


# ---------------------------------------------------------------------------
# 11. GET /overlay — specific analysis_id
# ---------------------------------------------------------------------------

def test_overlay_specific_analysis_id(client, clinician, uploaded_image):
    case_id, image_id = uploaded_image
    ar = client.post(
        f"/api/cases/{case_id}/analyze",
        json={"image_id": image_id},
        headers=clinician,
    )
    assert ar.status_code == 200
    analysis_id = ar.json()["id"]

    r = client.get(f"/api/images/{image_id}/overlay?analysis_id={analysis_id}", headers=clinician)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


# ---------------------------------------------------------------------------
# 12. GET /overlay — dynamic generation when overlay file is deleted
# ---------------------------------------------------------------------------

def test_overlay_dynamic_fallback(client, clinician, uploaded_image):
    case_id, image_id = uploaded_image
    ar = client.post(
        f"/api/cases/{case_id}/analyze",
        json={"image_id": image_id},
        headers=clinician,
    )
    assert ar.status_code == 200
    data = ar.json()
    analysis_id = data["id"]

    # Remove the saved overlay file from disk if it was written
    from sqlalchemy.orm import Session
    from app.db.session import engine as db_engine
    from app.models.models import AnalysisResult as AR

    with Session(db_engine) as sess:
        result = sess.get(AR, analysis_id)
        if result and result.overlay_path:
            p = Path(result.overlay_path)
            if p.is_file():
                p.unlink()

    # Should still return a PNG (CT-only fallback or dynamic overlay)
    r = client.get(
        f"/api/images/{image_id}/overlay?analysis_id={analysis_id}",
        headers=clinician,
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
