"""
ORM Models — User & AnalysisJob
================================
Defines the two core database tables using SQLAlchemy 2.0 mapped columns.

Tables:
    users          — Stores registered teachers and HODs with their
                     institutional metadata (university, department).
    analysis_jobs  — Stores each video upload and its processing results,
                     linked to the uploading teacher via a foreign key.

Relationships:
    User (1) ←→ (many) AnalysisJob
    - A User can have many AnalysisJobs (one per video upload).
    - Each AnalysisJob belongs to exactly one User (the uploading teacher).
    - Cascade delete: if a User is removed, all their jobs are deleted too.
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import String, DateTime, Enum, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# ── Enumerations ─────────────────────────────────────────────────────

class JobStatus(str, PyEnum):
    """Tracks the lifecycle of a video analysis job."""
    PENDING = "PENDING"         # Job created, waiting to start
    PROCESSING = "PROCESSING"   # AI pipeline is actively analyzing the video
    COMPLETED = "COMPLETED"     # Analysis finished successfully; results available
    FAILED = "FAILED"           # Analysis encountered an error


class UserRole(str, PyEnum):
    """Defines the two user roles in the system."""
    TEACHER = "TEACHER"   # Can upload videos and view per-student results
    HOD = "HOD"           # Can view department-wide aggregated analytics


# ── User Model ───────────────────────────────────────────────────────

class User(Base):
    """
    Represents a registered user (Teacher or HOD).

    Key fields:
        - university / department: Used for data scoping — an HOD only sees
          analytics from teachers in the same university AND department.
        - role: Determines which dashboard the user is routed to after login.
    """
    __tablename__ = "users"

    # Primary key — auto-generated UUID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Authentication fields
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Profile fields
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(
        Enum(UserRole, name="user_role_enum"),
        default=UserRole.TEACHER,
        nullable=False
    )

    # Institutional metadata — captured during registration
    university: Mapped[str | None] = mapped_column(String(255), nullable=True)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Relationship: one user → many analysis jobs
    jobs: Mapped[list["AnalysisJob"]] = relationship(
        "AnalysisJob",
        back_populates="teacher",
        cascade="all, delete-orphan"
    )


# ── AnalysisJob Model ────────────────────────────────────────────────

class AnalysisJob(Base):
    """
    Represents a single video upload + AI analysis task.

    Lifecycle:
        1. Teacher uploads a video → job created with status PROCESSING
        2. Background task runs the AI pipeline (classroom_engagement.py)
        3. On success → status becomes COMPLETED, ai_results stores the JSON
        4. On failure → status becomes FAILED

    The `ai_results` JSONB column stores the full per-student analysis:
        {
          "students": [
            {
              "student_id": "1",
              "emotions":   {"neutral": 120, "happy": 30, ...},
              "actions":    {"writing_notes": 50, "sleeping": 10, ...},
              "engagement_over_time": [{"second": 1, "score": 85}, ...]
            },
            ...
          ]
        }
    """
    __tablename__ = "analysis_jobs"

    # Primary key — auto-generated UUID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Path to the uploaded video file on disk (e.g., uploads/<uuid>.mp4)
    video_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # Current processing status (PENDING → PROCESSING → COMPLETED/FAILED)
    status: Mapped[str] = mapped_column(
        Enum(JobStatus, name="job_status_enum"),
        default=JobStatus.PENDING,
        nullable=False
    )

    # Timestamp when the job was created (auto-set by database)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # How long the AI analysis took in seconds (set on completion/failure)
    processing_time: Mapped[float | None] = mapped_column(nullable=True)

    # Full AI analysis results as structured JSON (see docstring above)
    ai_results: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── Contextual tracking fields ───────────────────────────────────
    # Links each job to the teacher who uploaded it
    teacher_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    # Class section and course name — captured during upload for HOD filtering
    class_section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    course_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Relationship: many jobs → one teacher
    teacher: Mapped["User"] = relationship("User", back_populates="jobs")
