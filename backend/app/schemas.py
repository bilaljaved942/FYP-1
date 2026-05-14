"""
Pydantic Schemas — Request/Response Validation
================================================
Defines Pydantic models used for:
    - Validating incoming API request payloads (e.g., UserCreate)
    - Serializing outgoing API responses (e.g., JobResponse)

Note: The UserCreate schema used during registration lives in auth.py
alongside the authentication logic. These schemas cover the core
video upload and job tracking flows.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr

from app.models import JobStatus


# ── User Schemas ─────────────────────────────────────────────────────

class UserBase(BaseModel):
    """Base schema containing fields common to all user operations."""
    email: EmailStr


class UserCreate(UserBase):
    """Schema for creating a new user (basic — see auth.py for full registration)."""
    password: str


class UserResponse(UserBase):
    """Schema returned after user creation/lookup. Excludes sensitive fields."""
    id: uuid.UUID

    class Config:
        # Allows automatic conversion from SQLAlchemy ORM objects
        from_attributes = True


# ── AnalysisJob Schemas ──────────────────────────────────────────────

class UploadResponse(BaseModel):
    """
    Returned immediately after a video is uploaded.

    Fields:
        job_id:  UUID of the created analysis job (used for polling status).
        message: Human-readable confirmation message.
    """
    job_id: uuid.UUID
    message: str


class JobResponse(BaseModel):
    """
    Returned when polling the status of an analysis job via GET /jobs/{job_id}.

    Fields:
        id:              Unique job identifier.
        video_path:      Server-side path to the uploaded video file.
        status:          Current lifecycle state (PENDING/PROCESSING/COMPLETED/FAILED).
        created_at:      Timestamp when the job was created.
        processing_time: Duration of AI analysis in seconds (null if still processing).
        ai_results:      Full per-student analysis JSON (null until COMPLETED).
    """
    id: uuid.UUID
    video_path: str
    status: JobStatus
    created_at: datetime
    processing_time: float | None = None
    ai_results: dict[str, Any] | None

    class Config:
        # Allows automatic conversion from SQLAlchemy ORM objects
        from_attributes = True
