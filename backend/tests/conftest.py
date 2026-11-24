import os
import tempfile
import zipfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="artery-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["STORAGE_DIR"] = f"{_TMP}/storage"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["FIRST_ADMIN_EMAIL"] = "admin@example.com"
os.environ["FIRST_ADMIN_PASSWORD"] = "adminpass123"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.session import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as c:  # runs lifespan -> creates the bootstrap admin
        yield c


def _login(client, email, password):
    r = client.post("/api/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _register(client, email, role="researcher"):
    r = client.post("/api/auth/register", json={"email": email, "full_name": "Test User",
                                                 "password": "password123", "role": role})
    assert r.status_code == 201, r.text
    return _login(client, email, "password123")


@pytest.fixture()
def admin(client):
    return _login(client, "admin@example.com", "adminpass123")


@pytest.fixture()
def researcher(client):
    return _register(client, "res@example.com", "researcher")


@pytest.fixture()
def scientist(client):
    return _register(client, "ds@example.com", "data_scientist")


def synthetic_ct(shape=(48, 48, 48), spacing=(0.8, 0.8, 1.0)) -> sitk.Image:
    """Air background with a bright 'vessel' tube along z."""
    arr = np.full(shape, -1000, dtype=np.int16)  # (z, y, x)
    arr[:, 20:28, 20:28] = 400
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


@pytest.fixture()
def nifti_path(tmp_path) -> Path:
    path = tmp_path / "scan.nii.gz"
    sitk.WriteImage(synthetic_ct(), str(path))
    return path


@pytest.fixture()
def dicom_zip(tmp_path) -> Path:
    img = synthetic_ct()
    folder = tmp_path / "dicom"
    folder.mkdir()
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    series_uid = "1.2.826.0.1.3680043.2.1125.1"
    for i in range(img.GetDepth()):
        sl = img[:, :, i]
        for tag, value in {
            "0008|0060": "CT", "0020|000e": series_uid, "0020|0013": str(i + 1),
            "0020|0032": f"0\\0\\{float(i)}", "0020|0037": "1\\0\\0\\0\\1\\0",
            "0028|1052": "0", "0028|1053": "1", "0018|0050": "1.0",
        }.items():
            sl.SetMetaData(tag, value)
        writer.SetFileName(str(folder / f"{i:04d}.dcm"))
        writer.Execute(sl)
    zpath = tmp_path / "series.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in sorted(folder.iterdir()):
            zf.write(f, f"study/{f.name}")
    return zpath
