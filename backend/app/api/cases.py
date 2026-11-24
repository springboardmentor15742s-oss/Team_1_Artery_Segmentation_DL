from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import admins, get_current_user, writers
from app.db.session import get_db
from app.models.models import AnalysisResult, Case, CaseStatus, Patient, User
from app.schemas.schemas import AnalysisOut, AnalysisRequest, CaseIn, CaseOut, CaseUpdate
from app.services import audit, storage

router = APIRouter(prefix="/api/cases", tags=["cases"])


@router.get("", response_model=list[CaseOut])
def list_cases(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
               patient_id: int | None = None, status: CaseStatus | None = None,
               db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    query = select(Case).options(selectinload(Case.images)).order_by(Case.id.desc())
    if patient_id is not None:
        query = query.where(Case.patient_id == patient_id)
    if status is not None:
        query = query.where(Case.status == status)
    return db.scalars(query.offset(skip).limit(limit)).all()


@router.post("", response_model=CaseOut, status_code=201)
def create_case(data: CaseIn, db: Session = Depends(get_db), user: User = Depends(writers)):
    if not db.get(Patient, data.patient_id):
        raise HTTPException(404, "Patient not found")
    if db.scalar(select(Case).where(Case.case_code == data.case_code)):
        raise HTTPException(400, "case_code already exists")
    case = Case(**data.model_dump(), created_by=user.id)
    db.add(case)
    db.commit()
    audit.log(db, user.id, "create", "case", case.id)
    return case


def _get_case(db: Session, case_id: int) -> Case:
    case = db.scalar(select(Case).options(selectinload(Case.images)).where(Case.id == case_id))
    if not case:
        raise HTTPException(404, "Case not found")
    return case


@router.get("/{case_id}", response_model=CaseOut)
def get_case(case_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return _get_case(db, case_id)


@router.put("/{case_id}", response_model=CaseOut)
def update_case(case_id: int, data: CaseUpdate, db: Session = Depends(get_db), user: User = Depends(writers)):
    case = _get_case(db, case_id)
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(case, key, value)
    db.commit()
    audit.log(db, user.id, "update", "case", case.id)
    return case


@router.delete("/{case_id}", status_code=204)
def delete_case(case_id: int, db: Session = Depends(get_db), admin: User = Depends(admins)):
    case = _get_case(db, case_id)
    db.delete(case)
    db.commit()
    storage.delete_case_files(case_id)
    audit.log(db, admin.id, "delete", "case", case_id)


# ---------------------------------------------------------------------------
# AI Analysis endpoints
# ---------------------------------------------------------------------------

@router.post("/{case_id}/analyze", response_model=AnalysisOut, status_code=200)
def analyze_case(
    case_id: int,
    body: AnalysisRequest,
    db: Session = Depends(get_db),
    user: User = Depends(writers),
):
    """
    Run AI segmentation on one axial slice of the specified image.

    **RESEARCH PROTOTYPE – NOT FOR CLINICAL USE.**
    All measurements are geometric estimates and have not been validated.
    """
    from app.models.models import Image
    from app.services.analysis_service import run_case_analysis

    # 1. Verify the case exists
    case = _get_case(db, case_id)

    # 2. Verify the image belongs to this case
    image = db.get(Image, body.image_id)
    if not image or image.case_id != case_id:
        raise HTTPException(404, "Image not found in this case")

    # 3. Verify the processed file is present
    if not image.processed_path:
        raise HTTPException(422, "Image has not been preprocessed yet")

    try:
        analysis = run_case_analysis(
            image=image,
            slice_index=body.slice_index,
            model_name=body.model_name,
            user_id=user.id,
            db=db,
        )
        case.status = CaseStatus.analyzed
        case.model_version = body.model_name
        db.commit()
        audit.log(db, user.id, "analyze", "case", case_id)
        db.refresh(analysis)
        return analysis

    except HTTPException:
        raise
    except Exception as exc:
        # Mark case as failed and persist
        try:
            case.status = CaseStatus.failed
            db.commit()
        except Exception:
            db.rollback()
        raise HTTPException(500, f"Analysis failed: {exc}") from exc


@router.get("/{case_id}/analysis", response_model=AnalysisOut)
def get_latest_analysis(
    case_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Return the most recent analysis result for a case."""
    from app.models.models import AnalysisResult

    _get_case(db, case_id)   # 404 if case missing
    result = db.scalar(
        select(AnalysisResult)
        .where(AnalysisResult.case_id == case_id)
        .order_by(AnalysisResult.id.desc())
    )
    if not result:
        raise HTTPException(404, "No analysis found for this case")
    return result
