"""File layout: <STORAGE_DIR>/cases/<case_id>/<image_key>/{original.<ext>, processed.nii.gz}"""
import shutil
from pathlib import Path

from app.core.config import settings


def storage_root() -> Path:
    root = Path(settings.STORAGE_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def case_dir(case_id: int) -> Path:
    return storage_root() / "cases" / str(case_id)


def image_dir(case_id: int, key: str) -> Path:
    return case_dir(case_id) / key


def delete_case_files(case_id: int) -> None:
    shutil.rmtree(case_dir(case_id), ignore_errors=True)


def delete_image_files(case_id: int, key: str) -> None:
    shutil.rmtree(image_dir(case_id, key), ignore_errors=True)
