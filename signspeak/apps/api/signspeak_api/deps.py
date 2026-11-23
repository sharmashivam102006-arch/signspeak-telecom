from __future__ import annotations

import uuid

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session
from .models import User
from .security import decode_access_token

bearer = HTTPBearer(auto_error=False)


async def current_user_optional(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    session: AsyncSession = Depends(get_session),
) -> User | None:
    """Anonymous access is a feature here, not an oversight - see rooms.py."""
    if creds is None:
        return None
    try:
        payload = decode_access_token(creds.credentials)
        uid = uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
    return (await session.execute(select(User).where(User.id == uid))).scalar_one_or_none()


async def current_user(user: User | None = Depends(current_user_optional)) -> User:
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Sign in required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
