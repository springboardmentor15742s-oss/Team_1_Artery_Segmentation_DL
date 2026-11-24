from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models.models import CaseStatus, Role


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=8)
    role: Role = Role.researcher

    @field_validator("password")
    @classmethod
    def _bcrypt_limit(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:
            raise ValueError("password must be at most 72 bytes")
        return v


class UserOut(ORM):
    id: int
    email: EmailStr
    full_name: str
    role: Role
    is_active: bool


class RoleUpdate(BaseModel):
    role: Role


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class PatientIn(BaseModel):
    patient_code: str = Field(min_length=1, max_length=64)  # pseudonymous ID, e.g. P-0001
    age: int | None = Field(default=None, ge=0, le=130)
    gender: str | None = None
    medical_history: str | None = None
    risk_factors: str | None = None
    clinical_notes: str | None = None


class PatientOut(PatientIn, ORM):
    id: int
    created_at: datetime


class CaseIn(BaseModel):
    case_code: str = Field(min_length=1, max_length=64)
    patient_id: int
    modality: str = "ccta"
    scan_date: date | None = None
    anatomical_region: str | None = None
    coronary_dominance: str | None = None
    image_quality: str | None = None


class CaseUpdate(BaseModel):
    modality: str | None = None
    scan_date: date | None = None
    anatomical_region: str | None = None
    coronary_dominance: str | None = None
    image_quality: str | None = None


class ImageOut(ORM):
    id: int
    case_id: int
    file_format: str
    size_bytes: int
    original_shape: list[int] | None
    voxel_spacing_mm: list[float] | None
    processed_shape: list[int] | None
    processed_spacing_mm: list[float] | None
    uploaded_at: datetime


class CaseOut(ORM):
    id: int
    case_code: str
    patient_id: int
    modality: str
    scan_date: date | None
    anatomical_region: str | None
    coronary_dominance: str | None
    image_quality: str | None
    status: CaseStatus
    model_version: str | None
    created_at: datetime
    images: list[ImageOut] = []


# ---------------------------------------------------------------------------
# AI Analysis schemas
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    image_id: int
    model_name: str = Field(default="attention_unet", pattern="^(attention_unet|unet)$")
    slice_index: int | None = Field(default=None, ge=0)


class AnalysisOut(ORM):
    id: int
    case_id: int
    image_id: int
    model_name: str
    status: str
    slice_index: int | None

    # Timing
    inference_duration_ms: float | None

    # Segmentation
    foreground_pixels: int | None

    # Vessel morphology
    centerline_status: str | None
    centerline_px: int | None
    mean_diameter_px: float | None
    mean_diameter_mm: float | None

    # Stenosis geometry
    mld_px: float | None
    mld_mm: float | None
    reference_diameter_px: float | None
    reference_diameter_mm: float | None
    stenosis_pct: float | None
    severity_category: str | None

    # Spatial info
    pixel_spacing_mm: float | None
    physical_spacing_available: bool

    # Metadata
    error_message: str | None
    created_at: datetime

    # Research disclaimer — always present in the response
    disclaimer: str = (
        "RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. "
        "All measurements are geometric estimates from a research model "
        "and have not been validated for diagnostic or therapeutic decisions."
    )
