# ClassroomEye — AI-Powered Classroom Engagement Analytics 🎓

An end-to-end system for analyzing **individual student engagement** in classroom videos using computer vision, deep learning, and a modern React dashboard.

---

## 🏗️ System Architecture

```
FYP-1/
├── New_Scripts/            # Optimized AI pipeline (YOLOv8m + CLIP + Kalman tracking)
├── fyp-backend/            # FastAPI backend with PostgreSQL
├── fyp-frontend/           # React dashboard (Vite + Tailwind CSS + Recharts)
├── best_cnn_v2_emotions.keras   # Emotion classification model weights
└── class_map.json          # Action class label map
```

---

## 🤖 AI Pipeline

The core AI system (`New_Scripts/classroom_engagement.py`) runs as an isolated subprocess and performs:

1. **Detection** — YOLOv8m detects all persons per frame
2. **Tracking** — Kalman filter + Hungarian algorithm tracks each student across frames
3. **Re-Identification** — CLIP gallery-based appearance matching reassigns IDs after occlusions
4. **Emotion Classification** — CLIP ViT-B/32 classifies facial expressions
5. **Action Recognition** — CLIP classifies student actions (writing, sleeping, mobile, etc.)
6. **Engagement Scoring** — Per-second aggregation into engagement score

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

- Python 3.10+
- Node.js 18+
- Docker Desktop (for PostgreSQL)
- CUDA GPU recommended (runs on CPU too, but slower)

---

## 🚀 Backend Setup

```powershell
# 1. Navigate to backend directory
cd fyp-backend

# 2. Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Start PostgreSQL database (Docker required)
docker-compose up -d

# 5. Run the FastAPI server
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend runs at: **http://127.0.0.1:8000**
API Docs at: **http://127.0.0.1:8000/docs**

### Backend API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload .mp4 video, returns `job_id` |
| `GET`  | `/jobs/{job_id}` | Poll job status & get `ai_results` |
| `GET`  | `/health` | Health check |

---

## 🖥️ Frontend Setup

```powershell
# 1. Navigate to frontend directory
cd fyp-frontend

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
```

Dashboard runs at: **http://localhost:5173**

> The frontend uses Vite's proxy to forward `/api/*` calls to the FastAPI backend — no CORS configuration needed during development.

---

## 🧪 Testing End-to-End

1. Start Docker PostgreSQL: `docker-compose up -d` (inside `fyp-backend/`)
2. Start FastAPI backend (Terminal 1)
3. Start React frontend (Terminal 2)
4. Open **http://localhost:5173**
5. Login as **Teacher** → Upload `.mp4` video → View per-student charts
6. Login as **HOD** → View department-wide analytics dashboard

---

## 📦 Model Files (Not in Repo)

The following heavy model files are excluded from Git and must be placed manually in the project root:

| File | Description | Notes |
|------|-------------|-------|
| `best_cnn_v2_emotions.keras` | Custom-trained CNN for emotion recognition | ~23 MB |
| `yolov8m.pt` | YOLOv8 Medium weights | Auto-downloaded by ultralytics on first run |
| `class_map.json` | Action class label mapping | Included in repo |

---

## 👥 Team

Final Year Project — AI-Based Classroom Engagement and Action Recognition System

---

## 📄 License

This project is for academic purposes only.
