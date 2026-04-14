# ClassroomEye — AI-Powered Classroom Engagement Analytics 🎓

An end-to-end system for analysing **individual student engagement** in classroom
videos using computer vision, deep learning, and a modern React dashboard.

---

## 🏗️ Project Structure

```
FYP-1/
├── ai-pipeline/                         # AI analysis engine
│   ├── scripts/
│   │   ├── classroom_engagement.py      # Main pipeline (normal light)
│   │   ├── classroom_engagement_dim.py  # Dim-light variant
│   │   └── format_engagement_json.py    # Standalone JSON formatter utility
│   └── outputs/                         # AI output videos/JSONs (git-ignored)
│
├── backend/                             # FastAPI REST API
│   ├── app/
│   │   ├── main.py                      # API routes + video job orchestration
│   │   ├── ai_utils.py                  # Output transformer & engagement scorer
│   │   ├── database.py                  # Async SQLAlchemy + PostgreSQL
│   │   ├── models.py                    # ORM models (AnalysisJob, User)
│   │   └── schemas.py                   # Pydantic request/response schemas
│   ├── tests/
│   │   └── test_ai_utils.py             # Unit tests
│   ├── uploads/                         # Uploaded videos (git-ignored)
│   ├── Dockerfile                       # Backend container image
│   ├── docker-compose.yml               # Legacy: PostgreSQL only (kept for reference)
│   └── requirements.txt                 # All Python dependencies
│
├── frontend/                            # React Dashboard (Vite + Tailwind)
│   ├── src/
│   │   ├── components/                  # KPICard, Navbar
│   │   ├── pages/                       # LoginPage, TeacherDashboard, HODDashboard
│   │   └── services/                    # API client (api.js)
│   ├── nginx.conf                       # Nginx config for Docker deployment
│   ├── Dockerfile                       # Frontend container image (multi-stage)
│   └── package.json
│
├── models/                              # Model weights (git-ignored)
│   ├── best_cnn_v2_emotions.keras       # Custom CNN — emotion recognition (~23 MB)
│   └── class_map.json                   # Action class label map
│
├── archive/                             # Archived / legacy files
│   ├── backups/                         # Old script backups (.bak)
│   └── legacy-frontend/                 # Previous embedded frontend version
│
├── docker-compose.yml                   # 🐳 Full-stack: DB + Backend + Frontend
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🤖 AI Pipeline

The core AI system (`ai-pipeline/scripts/classroom_engagement.py`) runs as an
isolated subprocess and performs:

| Step | Model | Task |
|------|-------|------|
| Detection | YOLOv8m | Detect all persons per frame |
| Tracking | Kalman + Hungarian | Track each student across frames |
| Re-ID | CLIP gallery | Reassign IDs after occlusions |
| Emotion | CLIP ViT-B/32 | Classify facial expressions |
| Action | CLIP ViT-B/32 | Classify student actions |
| Scoring | Rule-based | Aggregate engagement score per second |

**Output format per student:**
```json
{
  "student_id": "1",
  "emotions": { "neutral": 45, "happy": 12 },
  "actions":  { "writing_notes": 30, "looking_away": 8 },
  "engagement_over_time": [{"second": 1, "score": 85}, {"second": 2, "score": 72}]
}
```

---

## ⚙️ Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Backend + AI pipeline |
| Node.js | 18+ | Frontend dev server |
| Docker Desktop | Latest | Full containerised deployment |
| CUDA GPU | Optional | Faster AI inference (CPU fallback included) |

**Model files** — place these in the `models/` directory before running:

| File | Size | Notes |
|------|------|-------|
| `best_cnn_v2_emotions.keras` | ~23 MB | Custom-trained CNN |
| `class_map.json` | Tiny | Included in repo under `models/` |
| `yolov8m.pt` | ~52 MB | Auto-downloaded by ultralytics on first run |

---

## 🐳 Option A — Run with Docker (Recommended for deployment)

This is the easiest way to run the **entire project** on any machine.

### 1. Install Docker Desktop
Download from https://www.docker.com/products/docker-desktop and start it.

### 2. Place model weights
```
FYP-1/
└── models/
    ├── best_cnn_v2_emotions.keras   ← put file here
    └── class_map.json               ← already present
```

### 3. Build and start all services
```powershell
# From the FYP-1 root directory:
docker compose up --build
```

This starts:
| Container | URL | Description |
|-----------|-----|-------------|
| `classroomeye_db` | `localhost:5432` | PostgreSQL database |
| `classroomeye_backend` | `http://localhost:8000` | FastAPI + AI pipeline |
| `classroomeye_frontend` | `http://localhost:80` | React dashboard |

### 4. Open the dashboard
Navigate to **http://localhost** in your browser.

### Useful Docker commands
```powershell
# Stop all containers
docker compose down

# Stop and wipe database
docker compose down -v

# View backend logs
docker compose logs -f backend

# Rebuild after code changes
docker compose up --build
```

---

## 🛠️ Option B — Run Locally (Development)

Use this for active development, so you benefit from hot-reload.

### Step 1 — Start PostgreSQL (Docker only for DB)
```powershell
cd backend
docker compose up -d        # starts only the database container
cd ..
```

### Step 2 — Run the Backend (FastAPI)
```powershell
cd backend

# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install all dependencies (includes AI pipeline deps)
pip install -r requirements.txt

# Start the API server
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend API: **http://127.0.0.1:8000**
API docs:    **http://127.0.0.1:8000/docs**

### Step 3 — Run the Frontend (React / Vite)
```powershell
cd frontend

# Install Node dependencies
npm install

# Start the dev server (hot-reload enabled)
npm run dev
```

Dashboard: **http://localhost:5173**

> The Vite dev server proxies `/api/*` requests to FastAPI — no CORS issues.

---

## 🤖 Option C — Run the AI Pipeline Standalone

You can run the AI analysis script directly, without the web app:

```powershell
cd ai-pipeline/scripts

# Activate the same venv as the backend (it has all AI deps)
..\..\backend\venv\Scripts\activate

# Run analysis on a video (normal light)
python classroom_engagement.py \
    --video path\to\your_video.mp4 \
    --output-json ..\..\ai-pipeline\outputs\result.json \
    --output-video ..\..\ai-pipeline\outputs\annotated.mp4 \
    --output-summary ..\..\ai-pipeline\outputs\summary.txt

# Run the dim-light optimised variant
python classroom_engagement_dim.py --video path\to\dim_video.mp4 ...

# Format an existing raw JSON into a summary
python format_engagement_json.py \
    --input ..\..\ai-pipeline\outputs\result.json \
    --output ..\..\ai-pipeline\outputs\formatted.json
```

**CLI arguments for `classroom_engagement.py`:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input `.mp4` file |
| `--output-json` | auto-derived | Path for raw per-frame output JSON |
| `--output-video` | auto-derived | Path for annotated output video |
| `--output-summary` | auto-derived | Path for text summary |
| `--faces-dir` | auto-derived | Directory for face crops (optional) |
| `--emotion-model` | unused | Kept for CLI compatibility |
| `--class-map` | unused | Kept for CLI compatibility |

---

## 🧪 Testing End-to-End (Local)

1. Start DB:      `cd backend && docker compose up -d`
2. Start API:     `cd backend && python -m uvicorn app.main:app --reload`
3. Start UI:      `cd frontend && npm run dev`
4. Open:          **http://localhost:5173**
5. Login as **Teacher** → Upload `.mp4` video → Wait → View per-student charts
6. Login as **HOD** → View department-wide analytics dashboard

### Backend Unit Tests
```powershell
cd backend
.\venv\Scripts\activate
python -m pytest tests/ -v
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload `.mp4` video → returns `job_id` |
| `GET`  | `/jobs/{job_id}` | Poll status + get `ai_results` |
| `GET`  | `/health` | Health check |

---

## 👥 Team

Final Year Project — AI-Based Classroom Engagement and Action Recognition System

---

## 📄 License

This project is for academic purposes only.
