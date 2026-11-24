import enum
from datetime import date, datetime, timezone

from sqlalchemy import JSON, BigInteger, Boolean, Date, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    clinician = "clinician"
    researcher = "researcher"
    data_scientist = "data_scientist"
    admin = "admin"


class CaseStatus(str, enum.Enum):
    created = "created"
    uploaded = "uploaded"
    preprocessed = "preprocessed"
    analyzed = "analyzed"
    failed = "failed"


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(Enum(Role, native_enum=False, length=32), default=Role.researcher)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # pseudonymous only
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(32), nullable=True)
    medical_history: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_factors: Mapped[str | None] = mapped_column(Text, nullable=True)
    clinical_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    cases: Mapped[list["Case"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class Case(Base):
    __tablename__ = "cases"
    id: Mapped[int] = mapped_column(primary_key=True)
    case_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    modality: Mapped[str] = mapped_column(String(64), default="ccta")
    scan_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    anatomical_region: Mapped[str | None] = mapped_column(String(128), nullable=True)
    coronary_dominance: Mapped[str | None] = mapped_column(String(32), nullable=True)
    image_quality: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[CaseStatus] = mapped_column(
        Enum(CaseStatus, native_enum=False, length=32), default=CaseStatus.created
    )
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    patient: Mapped[Patient] = relationship(back_populates="cases")
    images: Mapped[list["Image"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    analyses: Mapped[list["AnalysisResult"]] = relationship(back_populates="case", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"
    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    storage_key: Mapped[str] = mapped_column(String(64), unique=True)  # folder name under the case
    original_path: Mapped[str] = mapped_column(String(512))
    processed_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    file_format: Mapped[str] = mapped_column(String(16))  # nii | nii.gz | zip
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    original_shape: Mapped[list | None] = mapped_column(JSON, nullable=True)  # [x, y, z] voxels
    voxel_spacing_mm: Mapped[list | None] = mapped_column(JSON, nullable=True)  # [x, y, z]
    processed_shape: Mapped[list | None] = mapped_column(JSON, nullable=True)
    processed_spacing_mm: Mapped[list | None] = mapped_column(JSON, nullable=True)
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    case: Mapped[Case] = relationship(back_populates="images")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(64))
    resource: Mapped[str] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnalysisResult(Base):
    """Persisted result of a single-slice AI segmentation analysis."""
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))          # "success" | "failed" | "partial"
    slice_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inference_duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Storage paths
    mask_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Segmentation summary
    foreground_pixels: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Vessel morphology (may be absent when no vessel is detected)
    centerline_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    centerline_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mean_diameter_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_diameter_mm: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Stenosis geometry (requires vessel morphology)
    mld_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    mld_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_diameter_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_diameter_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    stenosis_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity_category: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Spatial calibration
    pixel_spacing_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    physical_spacing_available: Mapped[bool] = mapped_column(Boolean, default=False)

    # Error info
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))

    # Relationships
    case: Mapped["Case"] = relationship(back_populates="analyses")
    image: Mapped["Image"] = relationship()
