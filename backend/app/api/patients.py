from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import admins, get_current_user, writers
from app.db.session import get_db
from app.models.models import Patient, User
from app.schemas.schemas import PatientIn, PatientOut
from app.services import audit, storage

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.get("", response_model=list[PatientOut])
def list_patients(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
                  db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return db.scalars(select(Patient).order_by(Patient.id.desc()).offset(skip).limit(limit)).all()


@router.post("", response_model=PatientOut, status_code=201)
def create_patient(data: PatientIn, db: Session = Depends(get_db), user: User = Depends(writers)):
    if db.scalar(select(Patient).where(Patient.patient_code == data.patient_code)):
        raise HTTPException(400, "patient_code already exists")
    patient = Patient(**data.model_dump(), created_by=user.id)
    db.add(patient)
    db.commit()
    audit.log(db, user.id, "create", "patient", patient.id)
    return patient


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    return patient


@router.put("/{patient_id}", response_model=PatientOut)
def update_patient(patient_id: int, data: PatientIn, db: Session = Depends(get_db),
                   user: User = Depends(writers)):
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    clash = db.scalar(select(Patient).where(Patient.patient_code == data.patient_code, Patient.id != patient_id))
    if clash:
        raise HTTPException(400, "patient_code already exists")
    for key, value in data.model_dump().items():
        setattr(patient, key, value)
    db.commit()
    audit.log(db, user.id, "update", "patient", patient.id)
    return patient


@router.delete("/{patient_id}", status_code=204)
def delete_patient(patient_id: int, db: Session = Depends(get_db), admin: User = Depends(admins)):
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    case_ids = [c.id for c in patient.cases]
    db.delete(patient)
    db.commit()
    for case_id in case_ids:
        storage.delete_case_files(case_id)
    audit.log(db, admin.id, "delete", "patient", patient_id)
