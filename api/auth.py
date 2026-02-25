# api/auth.py
"""
JWT-based authentication using the existing User model.

Dependencies:
    pip install bcrypt python-jose[cryptography]
"""

from __future__ import annotations

import os
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from dotenv import load_dotenv  # pyre-ignore[21]
from fastapi import Depends, HTTPException, status  # pyre-ignore[21]
from fastapi.security import OAuth2PasswordBearer  # pyre-ignore[21]
from jose import JWTError, jwt  # pyre-ignore[21]
from sqlalchemy import select  # pyre-ignore[21]
from sqlalchemy.orm import Session  # pyre-ignore[21]
import bcrypt  # pyre-ignore[21]

load_dotenv()

# --------------- Config ---------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# --------------- Password hashing ---------------
_BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))


def _prehash_password(plain: str) -> bytes:
    # Avoid bcrypt's 72-byte input limit and keep behavior deterministic.
    return hashlib.sha256(plain.encode("utf-8")).digest()


def hash_password(plain: str) -> str:
    digest = _prehash_password(plain)
    return bcrypt.hashpw(digest, bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash_password(plain), hashed.encode("utf-8"))
    except Exception:
        return False


# --------------- JWT tokens ---------------
def create_access_token(user_id: UUID, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[str]:
    """Returns user_id string or None."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# --------------- FastAPI dependency ---------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/login", auto_error=False)


def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    """
    Dependency that extracts the current user from the JWT bearer token.
    Returns the user_id (str) or raises 401.
    """
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = decode_access_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id
