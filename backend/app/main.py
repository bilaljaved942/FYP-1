import asyncio
import logging
import subprocess
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, init_db
from app.models import AnalysisJob, JobStatus, User
from app.schemas import UploadResponse, JobResponse
from app.ai_utils import transform_ai_output
from app.auth import router as auth_router, get_current_user

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

app.include_router(auth_router)

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
    class_section: str = Form(None),
    course_name: str = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
        status=JobStatus.PROCESSING,
        teacher_id=current_user.id,
        class_section=class_section,
        course_name=course_name
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

@app.get("/analytics/hod")
async def get_hod_analytics(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    from app.models import UserRole
    if current_user.role != UserRole.HOD:
        raise HTTPException(status_code=403, detail="Not authorized. HOD role required.")
    
    from sqlalchemy.orm import joinedload
    
    # Fetch all completed jobs for teachers in the same university & department
    query = (
        select(AnalysisJob)
        .join(User, AnalysisJob.teacher_id == User.id)
        .where(
            User.university == current_user.university,
            User.department == current_user.department,
            AnalysisJob.status == JobStatus.COMPLETED
        )
        .options(joinedload(AnalysisJob.teacher))
    )
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    # Aggregate Data
    total_lectures = len(jobs)
    teacher_stats = {}
    class_stats = {}    # keyed by class_section
    course_stats = {}   # keyed by course_name
    
    total_eng_sum = 0
    total_students_across_all = 0
    
    # Aggregate emotions & actions across all courses
    emotions_by_course = {}  # {course_name: {emotion: count}}
    actions_by_course = {}   # {course_name: {action: count}}
    overall_emotions = {}
    overall_actions = {}
    
    for job in jobs:
        teacher_name = job.teacher.full_name or "Unknown"
        class_sec = job.class_section or "Unknown"
        course = job.course_name or "Unknown"
        
        if teacher_name not in teacher_stats:
            teacher_stats[teacher_name] = {"sum": 0, "count": 0, "sessions": 0}
        if class_sec not in class_stats:
            class_stats[class_sec] = {"sum": 0, "count": 0}
        if course not in course_stats:
            course_stats[course] = {"sum": 0, "count": 0}
        if course not in emotions_by_course:
            emotions_by_course[course] = {}
        if course not in actions_by_course:
            actions_by_course[course] = {}
            
        teacher_stats[teacher_name]["sessions"] += 1
        
        if not job.ai_results or "students" not in job.ai_results:
            continue
            
        students = job.ai_results["students"]
        for s in students:
            timeline = s.get("engagement_over_time", [])
            if not timeline: continue
            
            avg_eng = sum(t["score"] for t in timeline) / len(timeline)
            
            total_eng_sum += avg_eng
            total_students_across_all += 1
            
            teacher_stats[teacher_name]["sum"] += avg_eng
            teacher_stats[teacher_name]["count"] += 1
            
            class_stats[class_sec]["sum"] += avg_eng
            class_stats[class_sec]["count"] += 1
            
            course_stats[course]["sum"] += avg_eng
            course_stats[course]["count"] += 1
            
            # Aggregate emotions
            emotions = s.get("emotions", {})
            for emo, val in emotions.items():
                emotions_by_course[course][emo] = emotions_by_course[course].get(emo, 0) + val
                overall_emotions[emo] = overall_emotions.get(emo, 0) + val
                
            # Aggregate actions
            actions = s.get("actions", {})
            for act, val in actions.items():
                actions_by_course[course][act] = actions_by_course[course].get(act, 0) + val
                overall_actions[act] = overall_actions.get(act, 0) + val
                
    # Format for frontend
    dept_avg = round(total_eng_sum / total_students_across_all) if total_students_across_all > 0 else 0
    
    teacherComparison = []
    for name, stats in teacher_stats.items():
        avg = round(stats["sum"] / stats["count"]) if stats["count"] > 0 else 0
        teacherComparison.append({"name": name, "engagement": avg, "sessions": stats["sessions"]})
        
    classEngagement = []
    colors = ['var(--brand-primary)', 'var(--accent-primary)', '#f59e0b', '#06b6d4', '#f43f5e']
    for idx, (name, stats) in enumerate(course_stats.items()):
        avg = round(stats["sum"] / stats["count"]) if stats["count"] > 0 else 0
        classEngagement.append({"name": name, "avg": avg, "fill": colors[idx % len(colors)]})
        
    most_active_course = max(classEngagement, key=lambda x: x["avg"])["name"] if classEngagement else "N/A"
    
    # Emotion data formatted for pie/bar chart
    emotionData = [{"name": k, "value": round(v)} for k, v in sorted(overall_emotions.items(), key=lambda x: -x[1])]
    
    # Action data formatted for bar chart
    actionData = [{"name": k, "value": round(v)} for k, v in sorted(overall_actions.items(), key=lambda x: -x[1])]
    
    # Per-course emotion breakdown for grouped chart
    emotionsByCourse = []
    for course, emos in emotions_by_course.items():
        entry = {"course": course}
        for emo, val in emos.items():
            entry[emo] = round(val)
        emotionsByCourse.append(entry)
    
    # Per-course action breakdown
    actionsByCourse = []
    for course, acts in actions_by_course.items():
        entry = {"course": course}
        for act, val in acts.items():
            entry[act] = round(val)
        actionsByCourse.append(entry)
    
    return {
        "kpis": {
            "deptAvg": dept_avg,
            "lecturesAnalyzed": total_lectures,
            "mostActiveCourse": most_active_course,
        },
        "teacherComparison": teacherComparison,
        "classEngagement": classEngagement,
        "emotionData": emotionData,
        "actionData": actionData,
        "emotionsByCourse": emotionsByCourse,
        "actionsByCourse": actionsByCourse,
        "recentActivity": []
    }

