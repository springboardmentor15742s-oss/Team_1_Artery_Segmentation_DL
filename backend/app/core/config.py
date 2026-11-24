from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Infrastructure
    DATABASE_URL: str = "postgresql+psycopg2://artery:artery@localhost:5432/artery"
    STORAGE_DIR: str = "./storage"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Security
    SECRET_KEY: str = "dev-only-change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    # Optional bootstrap: creates this admin on startup if it does not exist.
    FIRST_ADMIN_EMAIL: str | None = None
    FIRST_ADMIN_PASSWORD: str | None = None

    # Upload limits
    MAX_UPLOAD_MB: int = 1024
    MAX_UNZIPPED_MB: int = 4096
    MAX_ZIP_FILES: int = 5000

    # Preprocessing defaults. The ML team confirms these (see ml/configs/preprocessing.yaml).
    TARGET_SPACING_MM: list[float] = [0.5, 0.5, 0.5]
    HU_MIN: float = -200.0
    HU_MAX: float = 800.0
    MIN_DIM_VOXELS: int = 32
    MAX_PROCESSED_VOXELS: int = 300_000_000

    # AI / ML Inference settings
    DEFAULT_MODEL: str = "attention_unet"
    MODEL_THRESHOLD: float = 0.50
    MODEL_HU_MIN: float = -100.0
    MODEL_HU_MAX: float = 700.0
    MODEL_IMAGE_SIZE: int = 128


settings = Settings()
