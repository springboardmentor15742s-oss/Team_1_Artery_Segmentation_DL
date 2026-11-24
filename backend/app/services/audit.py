from sqlalchemy.orm import Session

from app.models.models import AuditLog


def log(db: Session, user_id: int | None, action: str, resource: str, resource_id=None) -> None:
    db.add(
        AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=str(resource_id) if resource_id is not None else None,
        )
    )
    db.commit()
