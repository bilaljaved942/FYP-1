import asyncio
import logging
import subprocess
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, init_db
from app.models import AnalysisJob, JobStatus
from app.schemas import UploadResponse, JobResponse
from app.ai_utils import transform_ai_output

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Student Engagement Analysis System",
    description="Backend for analyzing classroom video engagement",
    version="1.0.0"
)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Path resolution ──────────────────────────────────────────────
# All paths are resolved relative to the project root (backend/../)
BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
PROJECT_ROOT = BACKEND_DIR.parent                             # FYP-1/
UPLOAD_DIR = BACKEND_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount the uploads directory to serve static files like .mp4 and .json publicly
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Force the browser to download the file instead of playing it."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

# ── AI Script Routing ────────────────────────────────────────────
# Two scripts: one optimised for normal light, one for dim classrooms.
# The backend samples the first few frames and routes automatically.
SCRIPT_NORMAL    = PROJECT_ROOT / "ai-pipeline" / "scripts" / "classroom_engagement.py"
SCRIPT_DIM       = PROJECT_ROOT / "ai-pipeline" / "scripts" / "classroom_engagement_dim.py"
BRIGHTNESS_THRESHOLD = 75          # 0-255 grayscale mean; below this → dim script

EMOTION_MODEL = PROJECT_ROOT / "models" / "best_cnn_v2_emotions.keras"
CLASS_MAP     = PROJECT_ROOT / "models" / "class_map.json"


def detect_brightness(video_path: str, sample_frames: int = 5) -> float:
    """
    Quickly estimate the average brightness of a video by sampling
    the first `sample_frames` frames and taking the mean grayscale value.
    Returns a float in [0, 255]. Falls back to 255 (assume bright) on error.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 255.0
        brightness_values = []
        for _ in range(sample_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(float(np.mean(gray)))
        cap.release()
        return float(np.mean(brightness_values)) if brightness_values else 255.0
    except Exception:
        return 255.0   # Safe fallback: use normal script


@app.on_event("startup")
async def startup_event() -> None:
    await init_db()


async def process_video(job_id: uuid.UUID, file_path: str) -> None:
    """
    Background task: run the real AI pipeline on the uploaded video,
    transform the output, and save results to the database.

    The AI script is launched as a **subprocess** via asyncio.to_thread() so that:
      - It doesn't block the FastAPI async event loop.
      - GPU memory is fully released when the process exits.
      - Compatible with Windows SelectorEventLoop (avoids NotImplementedError).
      - Global state in the script is isolated per-request.
    """
    from app.database import async_session_maker
    import time
    
    start_time = time.time()

    # ── Build per-job output paths ────────────────────────────────
    job_str = str(job_id)
    output_json = UPLOAD_DIR / f"{job_str}_output.json"
    output_video = UPLOAD_DIR / f"{job_str}_output.mp4"
    output_summary = UPLOAD_DIR / f"{job_str}_summary.txt"
    faces_dir = UPLOAD_DIR / f"{job_str}_faces"
    faces_dir.mkdir(exist_ok=True)

    try:
        # ── Brightness-based script routing ──────────────────────
        avg_brightness = detect_brightness(file_path)
        if avg_brightness < BRIGHTNESS_THRESHOLD:
            chosen_script = SCRIPT_DIM
            script_label  = "DIM-LIGHT"
        else:
            chosen_script = SCRIPT_NORMAL
            script_label  = "NORMAL"
        logger.info(
            f"[Job {job_str}] Brightness={avg_brightness:.1f} (threshold={BRIGHTNESS_THRESHOLD}) "
            f"→ Using {script_label} script: {chosen_script.name}"
        )

        # ── Launch AI as subprocess ───────────────────────────────
        if sys.prefix != sys.base_prefix:
            python_exe = sys.executable
        else:
            venv_python = BACKEND_DIR / "venv" / "Scripts" / "python.exe"  # Windows venv
            python_exe = str(venv_python) if venv_python.exists() else sys.executable
        cmd = [
            python_exe,
            str(chosen_script),
            "--video", str(file_path),
            "--output-json", str(output_json),
            "--output-video", str(output_video),
            "--output-summary", str(output_summary),
            "--faces-dir", str(faces_dir),
            "--emotion-model", str(EMOTION_MODEL),
            "--class-map", str(CLASS_MAP),
        ]

        logger.info(f"[Job {job_str}] Using python: {python_exe}")
        logger.info(f"[Job {job_str}] Starting AI analysis subprocess...")

        # Run subprocess.run in a thread to avoid blocking the event loop
        # and sidestep Windows SelectorEventLoop limitations.
        # PYTHONIOENCODING + PYTHONUTF8 force UTF-8 stdout/stderr so that
        # emoji characters in the AI scripts don't crash with UnicodeEncodeError
        # on Windows (which defaults subprocess pipes to cp1252).
        import os as _os
        subprocess_env = _os.environ.copy()
        subprocess_env["PYTHONIOENCODING"] = "utf-8"
        subprocess_env["PYTHONUTF8"] = "1"

        completed = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=subprocess_env,
        )

        if completed.returncode != 0:
            error_msg = (completed.stderr or "")[-2000:]
            logger.error(f"[Job {job_str}] AI subprocess FAILED (rc={completed.returncode}):\n{error_msg}")
            raise RuntimeError(f"AI analysis failed (exit code {completed.returncode}): {error_msg}")

        logger.info(f"[Job {job_str}] AI subprocess completed successfully.")

        # ── Transform raw output → per-student aggregated JSON ────
        ai_results = transform_ai_output(str(output_json))
        student_count = len(ai_results.get("students", []))
        logger.info(f"[Job {job_str}] Transformed results: {student_count} students found.")

        # ── Save to database ──────────────────────────────────────
        async with async_session_maker() as session:
            result = await session.execute(
                select(AnalysisJob).where(AnalysisJob.id == job_id)
            )
            job = result.scalar_one_or_none()

            if job:
                job.status = JobStatus.COMPLETED
                job.ai_results = ai_results
                job.processing_time = round(time.time() - start_time, 2)
                await session.commit()
                logger.info(f"[Job {job_str}] Status → COMPLETED ✅ (Time: {job.processing_time}s)")

    except Exception as exc:
        logger.exception(f"[Job {job_str}] process_video failed: {exc}")

        # Mark job as FAILED in the database
        try:
            async with async_session_maker() as session:
                result = await session.execute(
                    select(AnalysisJob).where(AnalysisJob.id == job_id)
                )
                job = result.scalar_one_or_none()
                if job:
                    job.status = JobStatus.FAILED
                    job.processing_time = round(time.time() - start_time, 2)
                    await session.commit()
                    logger.info(f"[Job {job_str}] Status → FAILED ❌ (Time: {job.processing_time}s)")
        except Exception as db_exc:
            logger.error(f"[Job {job_str}] Could not update DB to FAILED: {db_exc}")


@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
) -> UploadResponse:
    """
    Upload a video file for engagement analysis.

    - Saves file to uploads/ directory
    - Creates a DB record with PROCESSING status
    - Triggers background processing task
    - Returns job_id for tracking
    """
    # Generate unique filename
    file_extension = Path(file.filename).suffix if file.filename else ".mp4"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename

    # Save file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Create DB record
    job = AnalysisJob(
        video_path=str(file_path),
        status=JobStatus.PROCESSING
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Trigger background processing
    background_tasks.add_task(process_video, job.id, str(file_path))

    return UploadResponse(
        job_id=job.id,
        message="Video uploaded successfully. Processing started."
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> JobResponse:
    """
    Get the status and results of an analysis job.

    - Returns job status (PENDING, PROCESSING, COMPLETED, FAILED)
    - Returns ai_results when processing is complete
    """
    result = await db.execute(
        select(AnalysisJob).where(AnalysisJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse.from_orm(job) if hasattr(JobResponse, 'from_orm') else JobResponse.model_validate(job)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
