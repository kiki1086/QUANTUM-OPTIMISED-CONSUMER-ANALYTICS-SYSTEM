from typing import Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api import dependencies
from app.schemas.user import User, UserCreate
from app.models.user import User as UserModel
from app.core.security import get_password_hash

router = APIRouter()

@router.post("/", response_model=User)
def create_user(
    *,
    db: Session = Depends(dependencies.get_db),
    user_in: UserCreate,
) -> Any:
    # Stub: Create user in DB
    user = UserModel(
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
        is_superuser=user_in.is_superuser,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.get("/me", response_model=User)
def read_user_me() -> Any:
    # Stub: Return current user based on token
    return {"email": "admin@qocas.com", "is_active": True, "is_superuser": True}
