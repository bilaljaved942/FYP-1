# Classroom Engagement Analysis Pipeline

This repository contains the AI pipeline for analyzing student engagement in a classroom setting from video feeds. The system tracks individual students, handles temporary occlusions / exiting the frame, and classifies both their physical actions and facial emotions over time to generate a comprehensive class engagement summary.

## Pipeline Architecture

The pipeline consists of four main AI components working synchronously:

### 1. Object Detection (Student Localization)
- **Model:** YOLOv8m (Ultralytics)
- **Role:** Detects all people in the classroom frame.
- **Details:** The system uses Non-Maximum Suppression (NMS) and Minimum Area thresholds to filter out false positives and isolate individual students, even in dense classroom crowds.

### 2. Multi-Object Tracking & Re-Identification
- **Algorithms:** Kalman Filter + Hungarian Algorithm + OpenAI CLIP matching
- **Role:** Assigns and maintains consistent IDs (e.g., S1, S2) for each student throughout the video.
- **Details:** 
  - **Tracking:** A Kalman Filter predicts the bounding box position of a student even if the detector misses them for a few frames. (Zero-velocity is assumed since seated students are mostly stationary).
  - **Re-Identification:** If a student leaves the frame or is heavily occluded for a long period, their previous position (Spatial Match) and a CLIP-generated visual embedding of their body crop (Appearance Match) are saved to a "Graveyard." When they reappear, the system matches their new appearance against the graveyard to reassign their original Student ID.

### 3. Action Classification (Zero-Shot)
- **Model:** OpenAI CLIP (`ViT-B/32`)
- **Role:** Determines what the student is physically doing based on their full-body bounding box.
- **Details:** The visual embedding of the student body crop is matched against text prompts for specific actions:
  - `writing_notes`
  - `using_mobile`
  - `looking_away`
  - `sleeping`
  - `raising_hand`
  - `neutral`

### 4. Emotion Classification
- **Model:** Custom Convolutional Neural Network (CNN) in Keras/TensorFlow (`best_cnn_v2_emotions.keras`)
- **Role:** Analyzes the student's face crop (upper 1/3rd of the bounding box) to deduce emotional state.
- **Details:** Outputs emotions based on `class_map.json` labels (e.g., neutral, happy, sad, angry).

---

## Technical Optimizations

For performance, `classroom_engagement.py` incorporates several optimizations:
- **Batch Processing:** Emotion and Action classification models process multiple crops simultaneously (batch inference).
- **Frame Skipping:** The `PROCESS_EVERY_N_FRAMES` parameter allows the pipeline to skip frames to achieve real-time performance without losing track data.
- **Temporal Smoothing:** Employs exponential moving average for bounding boxes (`SMOOTH_ALPHA`) and a voting system over a history of frames (`FRAME_SMOOTHING`) to prevent extreme flickering of actions and emotions.

---

## Setup & Execution

### 1. Environment Setup
Due to package constraints (combining ancient ML packages, Windows PyTorch, and tensorflow dependencies), the project relies on **Python 3.12**.

Initialize the virtual environment:
```powershell
python -m venv .venv312
.\.venv312\Scripts\activate
```

Install the dependencies:
```powershell
uv pip install -r requirements.txt
# Alternatively, manual installs:
# uv pip install opencv-python-headless pillow scipy filterpy
# uv pip install "numpy<2.0.0" setuptools<70.0.0
# uv pip install torch==2.2.0 torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu
# uv pip install tensorflow ultralytics ftfy regex tqdm
# uv pip install git+https://github.com/openai/CLIP.git
```

*Note: You must have the latest Microsoft Visual C++ redistributable installed for PyTorch DLLs to run successfully on Windows without crashing.*

### 2. Running The Pipeline
Ensure the input video is named `final_video.mp4` and placed in the `FYP-1` base directory.

Execute the script:
```powershell
.\.venv312\Scripts\python.exe scripts\classroom_engagement.py
```

### 3. Generated Output
All output artifacts are saved into the `outputs/` folder:
1. **`output_engagement2.mp4`**: The rendered video with bounding boxes, IDs, and live classification tags.
2. **`output_engagement2.json`**: Frame-by-frame raw data tracking the exact state of every student.
3. **`output_engagement_summary2.txt`**: A high-level text report summarizing the overall engagement percentage (tracking Distracted vs. Engaged activities) for individual students and the classroom average.
4. **`faces_engagement/`**: A directory saving a snapshot image of each registered student for manual reference.
