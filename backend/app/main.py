"""
FastAPI Application — Main Entry Point
========================================
The core backend server for the ClassroomEye student engagement analysis system.

This module wires together:
    - Authentication (auth.py)         → User registration, login, JWT
    - Database models (models.py)      → User, AnalysisJob ORM tables
    - AI pipeline (ai_utils.py)        → Raw output → structured JSON
    - Video processing (subprocess)    → classroom_engagement.py scripts

Key Endpoints:
    POST /upload          → Upload a video for AI analysis (authenticated).
    GET  /jobs/{job_id}   → Poll the status/results of an analysis job.
    GET  /analytics/hod   → Aggregated department-wide analytics (HOD only).
    GET  /health          → Health check for monitoring.

Architecture:
    1. Teacher uploads a video via POST /upload.
    2. Backend saves the file, creates a DB record, and spawns a background task.
    3. Background task runs the AI script as a subprocess (GPU-accelerated).
    4. On completion, the structured results are saved to the DB as JSONB.
    5. Frontend polls GET /jobs/{job_id} until status is COMPLETED.
    6. HOD dashboard fetches GET /analytics/hod to see aggregated metrics.
"""

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

# ── FastAPI Application ──────────────────────────────────────────────
app = FastAPI(
    title="Student Engagement Analysis System",
    description="Backend for analyzing classroom video engagement",
    version="1.0.0"
)

# ── CORS Middleware ──────────────────────────────────────────────────
# Allow all origins so the frontend (running on a different port) can
# make API requests. In production, restrict to specific domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register authentication routes (/auth/register, /auth/login)
app.include_router(auth_router)

# ── Path Resolution ──────────────────────────────────────────────────
# All paths are resolved relative to the project root (backend/../)
BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
PROJECT_ROOT = BACKEND_DIR.parent                             # FYP-1/
UPLOAD_DIR = BACKEND_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve uploaded files (videos, JSON results) as static assets
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Force the browser to download a file (e.g., analysis results) instead of playing it."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')


# ── AI Script Routing ────────────────────────────────────────────────
# Two AI scripts exist:
#   - classroom_engagement.py     → Optimized for normal lighting conditions
#   - classroom_engagement_dim.py → Optimized for dim/low-light classrooms
# The backend automatically selects the right script based on video brightness.
SCRIPT_NORMAL    = PROJECT_ROOT / "ai-pipeline" / "scripts" / "classroom_engagement.py"
SCRIPT_DIM       = PROJECT_ROOT / "ai-pipeline" / "scripts" / "classroom_engagement_dim.py"
BRIGHTNESS_THRESHOLD = 75  # Grayscale mean (0-255); below this → use dim script

# Pre-trained model paths for emotion recognition
EMOTION_MODEL = PROJECT_ROOT / "models" / "best_cnn_v2_emotions.keras"
CLASS_MAP     = PROJECT_ROOT / "models" / "class_map.json"


def detect_brightness(video_path: str, sample_frames: int = 5) -> float:
    """
    Quickly estimate the average brightness of a video.

    Samples the first N frames, converts to grayscale, and returns
    the mean pixel value (0 = black, 255 = white).
    Falls back to 255 (assume bright) if the video can't be read.

    Args:
        video_path:    Path to the video file.
        sample_frames: Number of frames to sample for the estimate.

    Returns:
        Average brightness value as a float in [0, 255].
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


# ── Application Startup ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the database tables when the server starts."""
    await init_db()


# ── Background Video Processing ─────────────────────────────────────

async def process_video(job_id: uuid.UUID, file_path: str) -> None:
    """
    Background task: run the AI pipeline on an uploaded video.

    This function is called by FastAPI's BackgroundTasks after a video upload.
    It performs the following steps:
        1. Detect video brightness to choose the appropriate AI script.
        2. Launch the AI script as a subprocess (isolated GPU process).
        3. Wait for the subprocess to complete.
        4. Transform raw per-frame output into per-student aggregated JSON.
        5. Save the results to the database (status → COMPLETED).

    The AI script is run as a subprocess (not in-process) because:
        - GPU memory is fully released when the subprocess exits.
        - It doesn't block the FastAPI async event loop.
        - Global state in the AI script is isolated per request.
        - Compatible with Windows SelectorEventLoop.
    """
    from app.database import async_session_maker
    import time
    
    start_time = time.time()

    # ── Build per-job output file paths ──────────────────────────────
    job_str = str(job_id)
    output_json = UPLOAD_DIR / f"{job_str}_output.json"
    output_video = UPLOAD_DIR / f"{job_str}_output.mp4"
    output_summary = UPLOAD_DIR / f"{job_str}_summary.txt"
    faces_dir = UPLOAD_DIR / f"{job_str}_faces"
    faces_dir.mkdir(exist_ok=True)

    try:
        # ── Step 1: Brightness-based script selection ────────────────
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

        # ── Step 2: Build the subprocess command ─────────────────────
        # Use the venv's Python executable to ensure correct dependencies
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

        # ── Step 3: Run the subprocess ───────────────────────────────
        # Force UTF-8 encoding to prevent Windows cp1252 crashes on emoji output
        import os as _os
        subprocess_env = _os.environ.copy()
        subprocess_env["PYTHONIOENCODING"] = "utf-8"
        subprocess_env["PYTHONUTF8"] = "1"

        # Run in a thread to avoid blocking the async event loop
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

        # ── Step 4: Transform raw output → structured JSON ──────────
        ai_results = transform_ai_output(str(output_json))
        student_count = len(ai_results.get("students", []))
        logger.info(f"[Job {job_str}] Transformed results: {student_count} students found.")

        # ── Step 5: Save results to database ─────────────────────────
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

        # Mark job as FAILED in the database so the frontend can show an error
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


# ═══════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════


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
    Upload a classroom video for AI engagement analysis.

    Requires authentication (JWT token in Authorization header).

    Steps:
        1. Save the uploaded MP4 file to the uploads/ directory.
        2. Create an AnalysisJob record in the database (status: PROCESSING).
        3. Trigger the background AI processing task.
        4. Return the job_id immediately (frontend polls for results).

    Form Parameters:
        file:           The MP4 video file.
        class_section:  e.g., "CS-8A" — stored for HOD filtering.
        course_name:    e.g., "GenAI" — stored for HOD filtering.
    """
    # Generate a unique filename to prevent collisions
    file_extension = Path(file.filename).suffix if file.filename else ".mp4"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename

    # Save the uploaded file to disk
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Create a database record for this analysis job
    job = AnalysisJob(
        video_path=str(file_path),
        status=JobStatus.PROCESSING,
        teacher_id=current_user.id,       # Link to the authenticated teacher
        class_section=class_section,       # For HOD department filtering
        course_name=course_name            # For HOD course-level analytics
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Launch AI analysis in the background (non-blocking)
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
    Get the current status and results of a video analysis job.

    The frontend polls this endpoint every 3 seconds until status is COMPLETED.

    Returns:
        - status: PENDING | PROCESSING | COMPLETED | FAILED
        - ai_results: Full per-student analysis JSON (only when COMPLETED)
        - processing_time: Duration in seconds (set on completion/failure)
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
    """Simple health check endpoint for monitoring and load balancers."""
    return {"status": "healthy"}


# ═══════════════════════════════════════════════════════════════════════
#  HOD ANALYTICS ENDPOINT
# ═══════════════════════════════════════════════════════════════════════

@app.get("/analytics/hod")
async def get_hod_analytics(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Aggregated department-wide analytics for HOD users.

    Security:
        - Requires authentication (JWT token).
        - Only accessible to users with the HOD role (returns 403 otherwise).

    Data Scoping:
        - Only fetches completed analysis jobs from teachers who share the
          same university AND department as the authenticated HOD.
        - This ensures HODs cannot see data from other departments.

    Returns a JSON object with:
        - kpis:              Department avg engagement, lecture count, top course.
        - teacherComparison: Per-teacher engagement averages for bar chart.
        - classEngagement:   Per-course engagement averages for progress bars.
        - emotionData:       Overall emotion distribution for pie chart.
        - actionData:        Overall action breakdown for horizontal bar chart.
        - emotionsByCourse:  Grouped emotion counts per course.
        - actionsByCourse:   Grouped action counts per course.
    """
    from app.models import UserRole
    
    # ── Access Control: HOD role required ────────────────────────────
    if current_user.role != UserRole.HOD:
        raise HTTPException(status_code=403, detail="Not authorized. HOD role required.")
    
    from sqlalchemy.orm import joinedload
    
    # ── Fetch all completed jobs from teachers in the HOD's department ──
    query = (
        select(AnalysisJob)
        .join(User, AnalysisJob.teacher_id == User.id)
        .where(
            User.university == current_user.university,
            User.department == current_user.department,
            AnalysisJob.status == JobStatus.COMPLETED
        )
        .options(joinedload(AnalysisJob.teacher))  # Eager-load teacher data
    )
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    # ── Initialize aggregation containers ────────────────────────────
    total_lectures = len(jobs)
    teacher_stats = {}      # {teacher_name: {sum, count, sessions}}
    class_stats = {}        # {class_section: {sum, count}}
    course_stats = {}       # {course_name: {sum, count}}
    
    total_eng_sum = 0
    total_students_across_all = 0
    
    # Emotion & action aggregation across all courses
    emotions_by_course = {}  # {course_name: {emotion: count}}
    actions_by_course = {}   # {course_name: {action: count}}
    overall_emotions = {}    # {emotion: total_count}
    overall_actions = {}     # {action: total_count}
    
    # ── Iterate through all jobs and aggregate data ──────────────────
    for job in jobs:
        teacher_name = job.teacher.full_name or "Unknown"
        class_sec = job.class_section or "Unknown"
        course = job.course_name or "Unknown"
        
        # Initialize containers for new teachers/classes/courses
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
        
        # Skip jobs with no AI results (shouldn't happen for COMPLETED, but be safe)
        if not job.ai_results or "students" not in job.ai_results:
            continue
            
        # ── Process each student in the analysis results ─────────────
        students = job.ai_results["students"]
        for s in students:
            timeline = s.get("engagement_over_time", [])
            if not timeline: continue
            
            # Calculate average engagement score for this student
            avg_eng = sum(t["score"] for t in timeline) / len(timeline)
            
            # Accumulate into global and per-group totals
            total_eng_sum += avg_eng
            total_students_across_all += 1
            
            teacher_stats[teacher_name]["sum"] += avg_eng
            teacher_stats[teacher_name]["count"] += 1
            
            class_stats[class_sec]["sum"] += avg_eng
            class_stats[class_sec]["count"] += 1
            
            course_stats[course]["sum"] += avg_eng
            course_stats[course]["count"] += 1
            
            # Aggregate emotions (e.g., neutral: 120, happy: 30, ...)
            emotions = s.get("emotions", {})
            for emo, val in emotions.items():
                emotions_by_course[course][emo] = emotions_by_course[course].get(emo, 0) + val
                overall_emotions[emo] = overall_emotions.get(emo, 0) + val
                
            # Aggregate actions (e.g., writing_notes: 50, sleeping: 10, ...)
            actions = s.get("actions", {})
            for act, val in actions.items():
                actions_by_course[course][act] = actions_by_course[course].get(act, 0) + val
                overall_actions[act] = overall_actions.get(act, 0) + val
                
    # ── Format aggregated data for the frontend ──────────────────────
    
    # Department-wide average engagement percentage
    dept_avg = round(total_eng_sum / total_students_across_all) if total_students_across_all > 0 else 0
    
    # Per-teacher engagement comparison (for bar chart)
    teacherComparison = []
    for name, stats in teacher_stats.items():
        avg = round(stats["sum"] / stats["count"]) if stats["count"] > 0 else 0
        teacherComparison.append({"name": name, "engagement": avg, "sessions": stats["sessions"]})
    
    # Per-course engagement (for progress bars)
    classEngagement = []
    colors = ['var(--brand-primary)', 'var(--accent-primary)', '#f59e0b', '#06b6d4', '#f43f5e']
    for idx, (name, stats) in enumerate(course_stats.items()):
        avg = round(stats["sum"] / stats["count"]) if stats["count"] > 0 else 0
        classEngagement.append({"name": name, "avg": avg, "fill": colors[idx % len(colors)]})
    
    # Top course by engagement (shown in KPI card)
    most_active_course = max(classEngagement, key=lambda x: x["avg"])["name"] if classEngagement else "N/A"
    
    # Overall emotion distribution (for pie chart) — sorted by frequency
    emotionData = [{"name": k, "value": round(v)} for k, v in sorted(overall_emotions.items(), key=lambda x: -x[1])]
    
    # Overall action breakdown (for horizontal bar chart) — sorted by frequency
    actionData = [{"name": k, "value": round(v)} for k, v in sorted(overall_actions.items(), key=lambda x: -x[1])]
    
    # Per-course emotion breakdown (for grouped bar chart)
    emotionsByCourse = []
    for course, emos in emotions_by_course.items():
        entry = {"course": course}
        for emo, val in emos.items():
            entry[emo] = round(val)
        emotionsByCourse.append(entry)
    
    # Per-course action breakdown (for grouped bar chart)
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
        "recentActivity": []  # Placeholder for future implementation
    }
