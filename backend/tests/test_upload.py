from pathlib import Path

from app.core.config import settings
from tests.test_cases import make_case, make_patient


def setup_case(client, headers):
    p = make_patient(client, headers)
    return make_case(client, headers, p["id"])


def upload(client, headers, case_id, path: Path, name=None):
    with path.open("rb") as fh:
        return client.post(f"/api/cases/{case_id}/images", files={"file": (name or path.name, fh)}, headers=headers)


def test_upload_nifti_preprocesses(client, researcher, nifti_path):
    case = setup_case(client, researcher)
    r = upload(client, researcher, case["id"], nifti_path)
    assert r.status_code == 201, r.text
    img = r.json()
    assert img["original_shape"] == [48, 48, 48]
    assert img["voxel_spacing_mm"] == [0.8, 0.8, 1.0]
    assert img["processed_spacing_mm"] == settings.TARGET_SPACING_MM
    assert img["processed_shape"] == [77, 77, 96]  # 48*0.8/0.5=76.8 -> 77, 48*1.0/0.5=96
    assert client.get(f"/api/cases/{case['id']}", headers=researcher).json()["status"] == "preprocessed"
    assert len(client.get(f"/api/cases/{case['id']}/images", headers=researcher).json()) == 1


def test_upload_dicom_zip(client, researcher, dicom_zip):
    case = setup_case(client, researcher)
    r = upload(client, researcher, case["id"], dicom_zip)
    assert r.status_code == 201, r.text
    assert r.json()["file_format"] == "zip" and r.json()["original_shape"] == [48, 48, 48]


def test_slice_endpoint(client, researcher, nifti_path):
    case = setup_case(client, researcher)
    image = upload(client, researcher, case["id"], nifti_path).json()
    for axis in ("axial", "coronal", "sagittal"):
        r = client.get(f"/api/images/{image['id']}/slice", params={"axis": axis, "index": 10}, headers=researcher)
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    r = client.get(f"/api/images/{image['id']}/slice", params={"source": "original"}, headers=researcher)
    assert r.status_code == 200  # default index = middle
    assert client.get(f"/api/images/{image['id']}/slice", params={"index": 9999}, headers=researcher).status_code == 422
    assert client.get(f"/api/images/{image['id']}/slice", params={"axis": "diagonal"}, headers=researcher).status_code == 422


def test_download_volume(client, researcher, nifti_path):
    case = setup_case(client, researcher)
    image = upload(client, researcher, case["id"], nifti_path).json()
    r = client.get(f"/api/images/{image['id']}/file", headers=researcher)
    assert r.status_code == 200 and len(r.content) > 0


def test_rejects_bad_uploads(client, researcher, scientist, tmp_path, nifti_path):
    case = setup_case(client, researcher)
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    assert upload(client, researcher, case["id"], txt).status_code == 400
    fake = tmp_path / "fake.nii.gz"
    fake.write_bytes(b"not a real volume")
    assert upload(client, researcher, case["id"], fake).status_code == 422
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"not a zip")
    assert upload(client, researcher, case["id"], bad_zip).status_code == 422
    assert upload(client, scientist, case["id"], nifti_path).status_code == 403
    assert upload(client, researcher, 9999, nifti_path).status_code == 404
    assert client.get(f"/api/cases/{case['id']}", headers=researcher).json()["images"] == []


def test_rejects_blank_and_small_volumes(client, researcher, tmp_path):
    import numpy as np
    import SimpleITK as sitk
    case = setup_case(client, researcher)
    blank = tmp_path / "blank.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((40, 40, 40), dtype=np.int16)), str(blank))
    assert upload(client, researcher, case["id"], blank).status_code == 422
    small = tmp_path / "small.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.random.randint(0, 5, (8, 8, 8)).astype(np.int16)), str(small))
    assert upload(client, researcher, case["id"], small).status_code == 422


def test_size_limit(client, researcher, nifti_path, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0)
    case = setup_case(client, researcher)
    assert upload(client, researcher, case["id"], nifti_path).status_code == 413


def test_delete_case_removes_files(client, admin, researcher, nifti_path):
    from app.services import storage
    case = setup_case(client, researcher)
    upload(client, researcher, case["id"], nifti_path)
    assert storage.case_dir(case["id"]).exists()
    assert client.delete(f"/api/cases/{case['id']}", headers=admin).status_code == 204
    assert not storage.case_dir(case["id"]).exists()
