"""
Authentication & Authorization Module
=======================================
Handles user registration, login, JWT token management, and role-based access.

Endpoints:
    POST /auth/register  — Create a new Teacher/HOD account with institutional metadata.
    POST /auth/login     — Authenticate with email + password, receive a JWT token.

Security Flow:
    1. User registers → password hashed with bcrypt → stored in DB.
    2. User logs in → credentials verified → JWT token issued.
    3. Protected endpoints use `get_current_user` dependency to:
       - Extract the JWT from the Authorization header
       - Decode and validate the token
       - Look up the user in the database
       - Return the authenticated User ORM object

JWT Payload Structure:
    {
        "sub":  "user@example.com",   # Subject (user's email)
        "role": "TEACHER",            # User role for frontend routing
        "name": "Dr. Smith",          # Display name for the navbar
        "exp":  1699999999            # Expiration timestamp
    }
"""

import os
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
import bcrypt

from app.database import get_db
from app.models import User, UserRole

# ── JWT Configuration ────────────────────────────────────────────────
# SECRET_KEY should be overridden via environment variable in production.
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 3000  # ~2 days; generous for demo/development

# ── OAuth2 Scheme ────────────────────────────────────────────────────
# Tells FastAPI where to find the token (Authorization: Bearer <token>)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["Authentication"])


# ── Pydantic Schemas ─────────────────────────────────────────────────

class Token(BaseModel):
    """Response schema for both /auth/register and /auth/login endpoints."""
    access_token: str          # The JWT token string
    token_type: str            # Always "bearer"
    role: str | None = None    # User role (TEACHER/HOD) for frontend routing
    full_name: str | None = None  # User's display name for the navbar


class UserCreate(BaseModel):
    """
    Request schema for POST /auth/register.
    Captures all fields needed to create a Teacher or HOD account.
    """
    email: EmailStr                    # Must be a valid email format
    password: str                      # Plain-text password (hashed before storing)
    role: UserRole                     # TEACHER or HOD
    full_name: str                     # Display name (shown in navbar)
    university: str | None = None      # e.g., "FAST NUCES" — used for HOD data scoping
    department: str | None = None      # e.g., "Computer Science" — used for HOD data scoping


# ── Password Utilities ───────────────────────────────────────────────

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compare a plain-text password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        return False


def get_password_hash(password: str) -> str:
    """Hash a plain-text password using bcrypt with an auto-generated salt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


# ── JWT Token Creation ───────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """
    Generate a signed JWT token.

    Args:
        data: Payload dict (must include "sub" for the user's email).
        expires_delta: Optional custom expiration time. Defaults to 15 minutes.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# ── Registration Endpoint ────────────────────────────────────────────

@router.post("/auth/register", response_model=Token)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """
    Create a new user account.

    Steps:
        1. Check if email is already registered (reject duplicates).
        2. Hash the password with bcrypt.
        3. Create User record with university/department metadata.
        4. Issue a JWT token so the user is immediately authenticated.
    """
    # Prevent duplicate registrations
    result = await db.execute(select(User).where(User.email == user.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create the new user in the database
    db_user = User(
        email=user.email,
        hashed_password=get_password_hash(user.password),
        role=user.role,
        full_name=user.full_name,
        university=user.university,
        department=user.department
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    
    # Issue JWT and return to the frontend
    access_token = create_access_token(data={"sub": db_user.email, "role": db_user.role, "name": db_user.full_name})
    return {"access_token": access_token, "token_type": "bearer", "role": db_user.role}


# ── Login Endpoint ───────────────────────────────────────────────────

@router.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    """
    Authenticate a user with email + password.

    Note: OAuth2PasswordRequestForm uses 'username' field — we treat it as email.
    Returns JWT token + role + full_name for frontend display.
    """
    # Look up user by email
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    
    # Validate credentials
    if not user or not user.hashed_password or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
        
    # Issue JWT and return with role + name for frontend routing/display
    access_token = create_access_token(data={"sub": user.email, "role": user.role, "name": user.full_name})
    return {"access_token": access_token, "token_type": "bearer", "role": user.role, "full_name": user.full_name}


# ── Token Validation Dependency ──────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    """
    FastAPI dependency that extracts and validates the JWT token.

    Used in protected endpoints like /upload and /analytics/hod:
        async def my_endpoint(current_user: User = Depends(get_current_user)):

    Raises HTTP 401 if the token is missing, expired, or invalid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the JWT and extract the email from the "sub" claim
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Look up the user in the database to ensure they still exist
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user
