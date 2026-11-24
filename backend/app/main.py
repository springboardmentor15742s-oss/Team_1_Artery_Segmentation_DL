import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from app.api import auth, cases, health, images, patients, users
from app.core.config import settings
from app.core.security import hash_password
from app.db.session import SessionLocal
from app.models.models import Role, User

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("artery")


def bootstrap_admin() -> None:
    if not (settings.FIRST_ADMIN_EMAIL and settings.FIRST_ADMIN_PASSWORD):
        return
    email = settings.FIRST_ADMIN_EMAIL.lower()
    with SessionLocal() as db:
        if db.scalar(select(User).where(func.lower(User.email) == email)):
            return
        db.add(User(email=email, full_name="Administrator", role=Role.admin,
                    hashed_password=hash_password(settings.FIRST_ADMIN_PASSWORD)))
        db.commit()
        log.info("Created bootstrap admin %s", email)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.SECRET_KEY == "dev-only-change-me":
        log.warning("SECRET_KEY is the development default; set a real one outside local development")
    bootstrap_admin()
    try:
        from app.services.ai_engine import ai_engine
        ai_engine.preload_defaults()
    except Exception as exc:
        log.warning("AI Engine preload deferred: %s", exc)
    yield


app = FastAPI(title="Artery Analysis API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (health.router, auth.router, users.router, patients.router, cases.router, images.router):
    app.include_router(router)
