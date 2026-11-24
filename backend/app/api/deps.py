import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.security import decode_token
from app.db.session import get_db
from app.models.models import Role, User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_error = HTTPException(
        status.HTTP_401_UNAUTHORIZED, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        user = db.get(User, int(decode_token(token)["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        raise credentials_error
    if not user or not user.is_active:
        raise credentials_error
    return user


def require_roles(*roles: Role):
    allowed = set(roles) | {Role.admin}

    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient permissions")
        return user

    return checker


writers = require_roles(Role.clinician, Role.researcher)
admins = require_roles(Role.admin)
