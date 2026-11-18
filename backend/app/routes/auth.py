"""Auth routes: register, login, /me, guest."""
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import create_access_token, get_current_user, hash_password, verify_password
from app.models import User, UserPreferences
from app.schemas import GuestCreate, Token, UserCreate, UserLogin, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _token_for(user: User) -> Token:
    token = create_access_token({"sub": user.id, "email": user.email})
    return Token(access_token=token, user=UserOut.model_validate(user))


@router.post("/register", response_model=Token, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=payload.email, name=payload.name, hashed_password=hash_password(payload.password))
    db.add(user); db.commit(); db.refresh(user)
    db.add(UserPreferences(user_id=user.id)); db.commit()
    return _token_for(user)


@router.post("/login", response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    return _token_for(user)


@router.post("/guest", response_model=Token, status_code=201)
def guest(payload: GuestCreate = GuestCreate(), db: Session = Depends(get_db)):
    """Demo/guest mode: no password, 24h JWT, is_guest=True."""
    email = f"guest-{uuid.uuid4().hex[:8]}@guest.signspeak.local"
    user = User(email=email, name=payload.name or "Guest", hashed_password="", is_guest=True)
    db.add(user); db.commit(); db.refresh(user)
    db.add(UserPreferences(user_id=user.id)); db.commit()
    return _token_for(user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user
