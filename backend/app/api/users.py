from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import admins
from app.db.session import get_db
from app.models.models import User
from app.schemas.schemas import RoleUpdate, UserOut
from app.services import audit

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("", response_model=list[UserOut])
def list_users(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
               db: Session = Depends(get_db), _: User = Depends(admins)):
    return db.scalars(select(User).order_by(User.id).offset(skip).limit(limit)).all()


@router.patch("/{user_id}/role", response_model=UserOut)
def set_role(user_id: int, data: RoleUpdate, db: Session = Depends(get_db), admin: User = Depends(admins)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    user.role = data.role
    db.commit()
    audit.log(db, admin.id, "set_role", "user", user.id)
    return user
