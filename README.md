# ClassroomEye — AI-Powered Classroom Engagement Analytics 🎓

An end-to-end system for analysing **individual student engagement** in classroom
videos using computer vision, deep learning, and a modern React dashboard.

Teachers upload lecture recordings, and the AI pipeline detects each student,
tracks their emotions and actions frame-by-frame, and generates per-second
engagement scores. HODs (Heads of Department) get an aggregated analytics
dashboard to compare teacher performance, course engagement, and student
behavior patterns across their entire department.

---

## 🏗️ Project Structure

```
FYP-1/
├── ai-pipeline/                         # AI analysis engine
│   ├── scripts/
│   │   ├── classroom_engagement.py      # Main pipeline (normal light)
│   │   ├── classroom_engagement_dim.py  # Dim-light variant (auto-selected)
│   │   └── format_engagement_json.py    # Standalone JSON formatter utility
│   └── outputs/                         # AI output videos/JSONs (git-ignored)
│
├── backend/                             # FastAPI REST API
│   ├── app/
│   │   ├── main.py                      # API routes, video processing, HOD analytics
│   │   ├── auth.py                      # JWT authentication, registration, login
│   │   ├── ai_utils.py                  # Output transformer & engagement scorer
│   │   ├── database.py                  # Async SQLAlchemy + PostgreSQL config
│   │   ├── models.py                    # ORM models (User, AnalysisJob)
│   │   └── schemas.py                   # Pydantic request/response schemas
│   ├── reset_db.py                      # Database reset utility (drops & recreates)
│   ├── tests/
│   │   └── test_ai_utils.py             # Unit tests
│   ├── uploads/                         # Uploaded videos & results (git-ignored)
│   └── requirements.txt                 # All Python dependencies
│
├── frontend/                            # React Dashboard (Vite + Tailwind)
│   ├── src/
│   │   ├── App.jsx                      # Root component — navigation & auth state
│   │   ├── main.jsx                     # Entry point — StrictMode + ThemeProvider
│   │   ├── index.css                    # Global styles, CSS variables, animations
│   │   ├── components/
│   │   │   ├── Navbar.jsx               # Top nav bar with logo, role badge, theme toggle
│   │   │   └── KPICard.jsx              # Reusable metric card (color-coded)
│   │   ├── context/
│   │   │   └── ThemeContext.jsx          # Dark/light theme provider & hook
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx          # Marketing page with features & role cards
│   │   │   ├── LoginPage.jsx            # Sign in / Create account (Teacher/HOD)
│   │   │   ├── TeacherDashboard.jsx     # Video upload + per-student analysis results
│   │   │   └── HODDashboard.jsx         # Department-wide aggregated analytics
│   │   └── services/
│   │       └── api.js                   # API client (upload, login, analytics)
│   ├── vite.config.js                   # Dev server config (API proxy)
│   └── package.json
│
├── models/                              # Pre-trained model weights (git-ignored)
│   ├── best_cnn_v2_emotions.keras       # Custom CNN — emotion recognition (~23 MB)
│   └── class_map.json                   # Action class label map
│
├── .gitignore
└── README.md
```

---

## 🔐 Authentication & Roles

The system supports two user roles with distinct dashboards:

| Role | Registration Fields | Dashboard |
|------|-------------------|-----------|
| **Teacher** | Name, Email, Password, University, Department | Upload videos, view per-student engagement charts |
| **HOD** | Name, Email, Password, University, Department | View aggregated department analytics |

**Data Scoping:** An HOD only sees analytics from teachers who registered with the **same university AND department**. This prevents cross-department data leakage.

**Auth Flow:**
1. User registers → password hashed with bcrypt → JWT token issued
2. User logs in → credentials verified → JWT token returned with role + name
3. Protected endpoints validate the JWT via `Authorization: Bearer <token>` header

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

**Brightness-based script routing:** The backend samples the first 5 frames of
each uploaded video. If the average grayscale brightness is below 75 (out of 255),
it automatically uses the `classroom_engagement_dim.py` script optimized for
low-light classrooms.

**Output format per student:**
```json
{
  "student_id": "1",
  "emotions": { "neutral": 45, "happy": 12, "sad": 3 },
  "actions":  { "writing_notes": 30, "looking_away": 8, "using_mobile": 2 },
  "engagement_over_time": [{"second": 1, "score": 85}, {"second": 2, "score": 72}]
}
```

---

## 📊 Dashboard Visualizations

### Teacher Dashboard
- **Engagement Timeline** — second-by-second line chart per student
- **Emotion Distribution** — donut/pie chart of detected emotions
- **Action Breakdown** — horizontal bar chart of student behaviors
- **KPI Cards** — dominant emotion, primary action, average engagement
- **Student Selector** — dropdown to switch between detected students

### HOD Dashboard
- **Dept. Avg Engagement** — KPI showing department-wide average
- **Top Course** — course with highest average engagement
- **Teacher-wise Engagement** — bar chart comparing all teachers
- **Overall Emotion Distribution** — aggregated pie chart across all courses
- **Overall Action Breakdown** — aggregated horizontal bar chart
- **Emotions by Course** — grouped bar chart (emotion × course)
- **Actions by Course** — grouped bar chart (action × course)
- **Course Engagement** — progress bar per course

---

## ⚙️ Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Backend + AI pipeline |
| Node.js | 18+ | Frontend dev server |
| PostgreSQL | 14+ | Database (local or Docker) |
| CUDA GPU | Optional | Faster AI inference (CPU fallback included) |

**Model files** — place these in the `models/` directory before running:

| File | Size | Notes |
|------|------|-------|
| `best_cnn_v2_emotions.keras` | ~23 MB | Custom-trained CNN |
| `class_map.json` | Tiny | Included in repo under `models/` |
| `yolov8m.pt` | ~52 MB | Auto-downloaded by ultralytics on first run |

---

## 🛠️ Local Development Setup

### Step 1 — Setup PostgreSQL

```bash
# Install PostgreSQL and create the database
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'pakistan942';"
sudo -u postgres psql -c "CREATE DATABASE fyp_db OWNER postgres;"
```

### Step 2 — Run the Backend (FastAPI)

```bash
cd backend

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# .\venv\Scripts\activate   # Windows

# Install all dependencies
pip install -r requirements.txt

# Initialize the database tables
python reset_db.py

# Start the API server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend API: **http://127.0.0.1:8000**
API docs:    **http://127.0.0.1:8000/docs**

### Step 3 — Run the Frontend (React / Vite)

```bash
cd frontend

# Install Node dependencies
npm install

# Start the dev server (hot-reload enabled)
npm run dev
```

Dashboard: **http://localhost:5173**

> The Vite dev server proxies `/api/*` requests to FastAPI — no CORS issues.

---

## ☁️ EC2 Deployment (AWS)

### Instance Requirements
- **AMI:** Ubuntu 22.04/24.04 LTS (plain, no SQL Server)
- **Instance type:** `g4dn.xlarge` (Tesla T4 GPU)
- **Storage:** 100 GiB root volume (gp3)
- **Security Group Inbound:** Ports 22 (SSH), 5173 (Frontend), 8000 (Backend)

### Setup Commands

```bash
# 1. System dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv postgresql postgresql-contrib libpq-dev git
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g pm2

# 2. NVIDIA GPU drivers (if not pre-installed)
sudo apt install -y nvidia-driver-550
sudo reboot

# 3. PostgreSQL setup
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'pakistan942';"
sudo -u postgres psql -c "CREATE DATABASE fyp_db OWNER postgres;"

# 4. Clone and setup
cd ~
git clone https://github.com/YOUR_USERNAME/FYP-1.git
cd FYP-1
git checkout full-system-integration

# 5. Backend setup
cd ~/FYP-1/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python reset_db.py

# 6. Frontend setup
cd ~/FYP-1/frontend
npm install

# 7. Start with PM2 (process manager)
cd ~/FYP-1/backend
pm2 start "source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000" --name backend

cd ~/FYP-1/frontend
pm2 start "npm run dev -- --host 0.0.0.0" --name frontend

pm2 save
```

Access: **http://YOUR_EC2_IP:5173**

---

## 🤖 Standalone AI Pipeline

You can run the AI analysis script directly, without the web app:

```bash
cd ai-pipeline/scripts

# Activate the same venv as the backend
source ../../backend/venv/bin/activate

# Run analysis on a video
python classroom_engagement.py \
    --video path/to/your_video.mp4 \
    --output-json ../../ai-pipeline/outputs/result.json \
    --output-video ../../ai-pipeline/outputs/annotated.mp4 \
    --output-summary ../../ai-pipeline/outputs/summary.txt
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input `.mp4` file |
| `--output-json` | auto-derived | Path for raw per-frame output JSON |
| `--output-video` | auto-derived | Path for annotated output video |
| `--output-summary` | auto-derived | Path for text summary |
| `--faces-dir` | auto-derived | Directory for face crops |
| `--emotion-model` | required | Path to `.keras` emotion model |
| `--class-map` | required | Path to `class_map.json` |

---

## 🧪 Testing End-to-End

1. Start PostgreSQL
2. Start Backend: `cd backend && uvicorn app.main:app --reload`
3. Start Frontend: `cd frontend && npm run dev`
4. Open: **http://localhost:5173**
5. **Create Account** as Teacher → Upload `.mp4` video → Wait → View per-student charts
6. **Create Account** as HOD (same university & department) → View aggregated analytics

### Backend Unit Tests

```bash
cd backend
source venv/bin/activate
python -m pytest tests/ -v
```

---

## 🌐 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/register` | — | Create a Teacher/HOD account |
| `POST` | `/auth/login` | — | Login with email + password → JWT token |
| `POST` | `/upload` | Bearer | Upload MP4 video with class_section + course_name |
| `GET`  | `/jobs/{job_id}` | — | Poll job status + get ai_results |
| `GET`  | `/analytics/hod` | Bearer (HOD) | Department-wide aggregated analytics |
| `GET`  | `/health` | — | Health check |
| `GET`  | `/download/{filename}` | — | Download a file from uploads/ |

---

## 🔧 Useful Commands

```bash
# Reset database (WARNING: deletes all data)
cd backend && python reset_db.py

# Check GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# View PM2 process status (EC2)
pm2 list
pm2 logs backend --lines 20
pm2 restart backend
```

---

## 👥 Team

Final Year Project — AI-Based Classroom Engagement and Action Recognition System

---

## 📄 License

This project is for academic purposes only.
