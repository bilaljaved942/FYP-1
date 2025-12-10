import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr

from app.models import JobStatus


# User schemas
class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: uuid.UUID

    class Config:
        from_attributes = True


# AnalysisJob schemas
class UploadResponse(BaseModel):
    job_id: uuid.UUID
    message: str


class JobResponse(BaseModel):
    id: uuid.UUID
    video_path: str
    status: JobStatus
    created_at: datetime
    ai_results: dict[str, Any] | None

    class Config:
        from_attributes = True
