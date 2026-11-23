"""Auth dependencies: JWT verification, current-user lookup."""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app import config
from app.database import get_db
from app.models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    if not hashed:
        return False
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    from datetime import datetime, timedelta, timezone

    to_encode = dict(data)
    to_encode["exp"] = datetime.now(timezone.utc) + timedelta(hours=config.ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)


def decode_token(token: str):
    try:
        return jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
    except JWTError:
        return None


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)
) -> User:
    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_token(creds.credentials)
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_optional_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)
):
    if creds is None:
        return None
    payload = decode_token(creds.credentials)
    if not payload or not payload.get("sub"):
        return None
    return db.query(User).filter(User.id == payload["sub"]).first()


def get_admin_user(user: User = Depends(get_current_user)) -> User:
    """Admin-only dependency (contract for POST /api/ai/sign/train and
    other privileged operations). The caller must be authenticated AND
    listed in config.ADMIN_EMAILS (by email) or config.ADMIN_USER_IDS
    (by user id). Both lists default to empty, so nobody is an admin
    until the operator sets ADMIN_EMAILS / ADMIN_USER_IDS in the
    environment. Non-admins get 403; unauthenticated callers get 401
    from get_current_user."""
    email = (user.email or "").strip().lower()
    if email in config.ADMIN_EMAILS or user.id in config.ADMIN_USER_IDS:
        return user
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin privileges required")
