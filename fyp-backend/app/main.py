import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, init_db
from app.models import AnalysisJob, JobStatus
from app.schemas import UploadResponse, JobResponse

app = FastAPI(
    title="Student Engagement Analysis System",
    description="Backend for analyzing classroom video engagement",
    version="1.0.0"
)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event() -> None:
    await init_db()


async def process_video(job_id: uuid.UUID, file_path: str) -> None:
    """Background task to process video and update results."""
    from app.database import async_session_maker
    
    # Simulate AI processing delay
    await asyncio.sleep(5)
    
    # Dummy AI results
    ai_results = {
        "emotions": {"angry": 4, "fear": 9, "happy": 4, "neutral": 8},
        "actions": {"sleeping": 10, "raising_hand": 4},
        "engagement_over_time": [
            {"second": 1, "score": 15},
            {"second": 2, "score": 70}
        ]
    }
    
    async with async_session_maker() as session:
        result = await session.execute(
            select(AnalysisJob).where(AnalysisJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        
        if job:
            job.status = JobStatus.COMPLETED
            job.ai_results = ai_results
            await session.commit()


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
