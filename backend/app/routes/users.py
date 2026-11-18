"""User preferences routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import User, UserPreferences
from app.schemas import PreferencesOut, PreferencesUpdate

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/me/preferences", response_model=PreferencesOut)
def get_prefs(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    prefs = db.query(UserPreferences).filter(UserPreferences.user_id == user.id).first()
    if not prefs:
        prefs = UserPreferences(user_id=user.id); db.add(prefs); db.commit(); db.refresh(prefs)
    return prefs


@router.put("/me/preferences", response_model=PreferencesOut)
def update_prefs(payload: PreferencesUpdate, db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)):
    prefs = db.query(UserPreferences).filter(UserPreferences.user_id == user.id).first()
    if not prefs:
        prefs = UserPreferences(user_id=user.id); db.add(prefs)
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(prefs, k, v)
    db.commit(); db.refresh(prefs)
    return prefs
