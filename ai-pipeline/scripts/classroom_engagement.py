"""
CLASSROOM ENGAGEMENT ANALYSIS SYSTEM - OPTIMIZED VERSION
Combined Detection, Tracking, and Classification with Performance Optimizations

Optimizations:
1. Batch inference for emotion model
2. Batch inference for CLIP action model  
3. ThreadPoolExecutor for parallel processing
4. Frame skipping option for faster processing
5. Reduced redundant computations
"""
import os
import cv2
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque, Counter
from filterpy.kalman import KalmanFilter
from PIL import Image
import clip
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # FYP-1 directory
# Video path and output paths are resolved at runtime from --video CLI argument
# (see main() — defaults to final_video1.mp4 if not provided)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Performance settings
PROCESS_EVERY_N_FRAMES = 1  # Set to 2 or 3 for faster processing (skip frames)
NUM_WORKERS = 4  # Number of threads for parallel processing
BATCH_SIZE = 8  # Batch size for classification

# Detection parameters
YOLO_CONF = 0.25              # Balanced: catches real students, avoids chairs/bags
YOLO_CONF_REGISTRATION = 0.22  # Slightly lower for registration to catch all seated students
IMG_SIZE = 640
NMS_IOU_THRESHOLD = 0.50       # Standard NMS threshold

# Tracking parameters
IOU_MATCH_THRESHOLD = 0.10     # Low: allow matching even with small overlap (moving students)
OVERLAP_MERGE_THRESHOLD = 0.40
MAX_MISSING_FRAMES = 300       # ~10s at 30fps for unconfirmed tracks
MAX_MISSING_FRAMES_CONFIRMED = 900  # ~30s at 30fps; confirmed students kept alive very long
MIN_HITS_TO_CONFIRM = 10       # Strict: need 10 consecutive hits to confirm (prevents phantom tracks)
CENTER_DIST_MATCH_THRESHOLD = 150  # Max pixel distance for center-distance matching (stage 2)
MIN_TRACK_AREA_RATIO = 0.0008
REGISTRATION_FRAMES = 180      # 6 seconds at 30fps for thorough initial scan
SMOOTH_ALPHA = 0.2
NEW_TRACK_GUARD_DIST = 80      # Large guard: prevents creating duplicate tracks near existing ones
REGISTRATION_CLUSTER_DIST = 65 # Larger: prevents splitting same person into multiple clusters

# Re-identification parameters (for students leaving and returning)
# NOTE: No spatial gate — students can return to ANY seat in the classroom.
# Re-ID is purely appearance-based (CLIP body embedding similarity via gallery).
REID_APPEARANCE_THRESHOLD = 0.40   # Low threshold: gallery matching compensates (max across 10 embeddings)
REID_MAX_DEAD_AGE = 9999           # Effectively infinite: never discard dead tracks (registry is permanent)
APPEARANCE_UPDATE_INTERVAL = 15    # Update gallery every 0.5s for richer appearance diversity

# Classification parameters
FRAME_SMOOTHING = 5
EMOTION_SMOOTHING = 15

# Post-processing: minimum fraction of total video frames a student must appear
# in to be considered real (filters ghost/phantom detections from the output).
# E.g. 0.05 = student must appear in at least 5% of the video's total frames.
MIN_FRAMES_FRACTION = 0.20

# CLIP action prompts
CLIP_LABELS = {
    "using_mobile": [
        "a student sitting at a desk holding a smartphone and looking at it",
        "a student in a classroom looking down at and using a mobile phone",
        "a person texting on a cell phone while seated at a desk",
        "a person scrolling on their phone under a desk"
    ],
    "writing_notes": [
        "a student writing with a pen on a notebook at a desk in a classroom",
        "a student taking notes on paper with a pen while sitting at a desk",
        "a student bent over a notebook writing something on a desk",
        "a person holding a pen and writing on paper at a desk"
    ],
    "raising_hand": [
        "a student raising one hand up high in a classroom",
        "a student with their arm raised up in the air to answer a question",
        "a person lifting their hand above their head in class"
    ],
    "sleeping": [
        "a student sleeping with their head down on a desk",
        "a student resting their head on their arms on a desk with eyes closed",
        "a person dozing off with their head on a classroom desk"
    ],
    "looking_away": [
        "a student looking sideways away from the front of the classroom",
        "a student turning their head to talk to another student",
        "a person in a classroom looking to the side or behind them"
    ],
    "neutral": [
        "a student sitting at a desk looking forward and listening in a classroom",
        "a student sitting upright at a desk paying attention",
        "a person sitting still at a classroom desk facing forward"
    ]
}

# CLIP active emotion prompts (Neutral is explicitly removed; we default to it algorithmically)
CLIP_EMOTION_LABELS = {
    "angry": [
        "a furious, angry face",
        "a face looking extremely mad and frustrated"
    ],
    "disgust": [
        "a face showing intense disgust",
        "a thoroughly grossed out or repulsed facial expression"
    ],
    "fear": [
        "a face showing extreme fear",
        "a deeply terrified or frightened face"
    ],
    "happy": [
        "a happy, smiling face",
        "a bright, genuine smile",
        "a laughing or cheerful expression"
    ],
    "sad": [
        "a genuinely sad face",
        "a completely downcast, crying, or depressed expression"
    ],
    "surprise": [
        "a face looking completely shocked or astonished",
        "a suddenly surprised face with wide eyes"
    ]
}

# (No faces_engagement directory — face images are not used by the pipeline)
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

# ==================== GLOBALS ====================
tracks = []
dead_tracks = []  # Graveyard: stores removed tracks for re-identification
yolo_model = None
clip_model = None
clip_preprocess = None
text_embeddings = None
emotion_text_embeddings = None
CLIP_CLASSES = []
CLIP_EMOTIONS = []
registration_complete = False
next_student_id = 1

# ---- PERMANENT STUDENT REGISTRY ----
# Maps student_id (int) -> dict with:
#   'gallery': list of CLIP embeddings (up to GALLERY_SIZE, from different frames/angles)
#   'face_img': reference face image
# This registry persists for the ENTIRE video — entries are NEVER deleted.
student_registry = {}
GALLERY_SIZE = 10  # Max embeddings stored per student for gallery-based matching

# Thread lock for JSON updates
json_lock = threading.Lock()

# JSON data structure
json_data = {
    "video_info": {},
    "students": {}
}

# ==================== KALMAN FILTER ====================
def create_kalman():
    """Kalman for STATIONARY objects"""
    kf = KalmanFilter(dim_x=8, dim_z=4)
    
    kf.F = np.array([
        [1,0,0,0,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,1],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1],
    ])
    
    kf.H = np.array([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
    ])
    
    kf.R *= 5
    kf.Q *= 0.001
    kf.P *= 10
    
    return kf

def bbox_to_z(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w/2, y1 + h/2, w, h])

def z_to_bbox(z):
    cx, cy, w, h = z.flatten()[:4]
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

# ==================== TRACK CLASS ====================
class Track:
    _counter = 1
    
    def __init__(self, bbox, frame_id):
        self.id = Track._counter
        Track._counter += 1
        
        self.kf = create_kalman()
        z = bbox_to_z(bbox)
        self.kf.x[:4] = z.reshape(-1, 1)
        
        self.bbox = np.array(bbox, dtype=np.float32)
        self.smooth_bbox = np.array(bbox, dtype=np.float32)
        self.start_frame = frame_id
        self.last_seen = frame_id
        self.hits = 1
        self.misses = 0
        
        self.student_id = None
        self.confirmed = False
        self.face_img = None
        
        # Appearance embedding for re-identification (latest single embedding)
        self.appearance_embedding = None
        self.last_embedding_frame = -999
        
        # Local gallery for this track instance (synced to student_registry)
        self.embedding_gallery = []
        
        self.emotion_history = deque(maxlen=EMOTION_SMOOTHING)
        self.action_history = deque(maxlen=FRAME_SMOOTHING)
        self.current_emotion = "neutral"
        self.current_action = "neutral"
    
    def predict(self):
        # Allow velocity — students may move (walk, shift seats)
        # Dampen velocity to avoid runaway drift
        self.kf.x[4:] *= 0.5
        self.kf.predict()
        self.bbox = z_to_bbox(self.kf.x[:4])
        self.misses += 1
        return self.bbox
    
    def update(self, bbox, frame_id, face_img=None):
        bbox = np.array(bbox, dtype=np.float32)
        
        z = bbox_to_z(bbox)
        self.kf.update(z.reshape(-1, 1))
        self.bbox = z_to_bbox(self.kf.x[:4])
        
        self.smooth_bbox = SMOOTH_ALPHA * self.bbox + (1 - SMOOTH_ALPHA) * self.smooth_bbox
        
        self.last_seen = frame_id
        self.hits += 1
        self.misses = 0
        
        if face_img is not None and self.face_img is None:
            self.face_img = face_img
    
    def update_emotion(self, emotion):
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > 0:
            self.current_emotion = Counter(self.emotion_history).most_common(1)[0][0]
    
    def update_action(self, action):
        self.action_history.append(action)
        if len(self.action_history) > 0:
            self.current_action = Counter(self.action_history).most_common(1)[0][0]
    
    def is_confirmed(self):
        return self.hits >= MIN_HITS_TO_CONFIRM
    
    def is_dead(self):
        # Confirmed students with IDs are kept alive much longer (30s) so they
        # can be re-matched if the detector picks them up again.
        if self.confirmed and self.student_id is not None:
            return self.misses > MAX_MISSING_FRAMES_CONFIRMED
        return self.misses > (MAX_MISSING_FRAMES if self.confirmed else MAX_MISSING_FRAMES // 3)
    
    def get_center(self):
        return np.array([(self.bbox[0] + self.bbox[2])/2, (self.bbox[1] + self.bbox[3])/2])

# ==================== UTILITIES ====================
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    
    return inter / (a1 + a2 - inter + 1e-6)

def strict_nms(boxes, threshold=NMS_IOU_THRESHOLD):
    """Very strict NMS - merge any overlapping boxes"""
    if len(boxes) == 0:
        return []
    
    boxes = [np.array(b) for b in boxes]
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    
    idx = np.argsort(areas)[::-1]
    
    keep = []
    while len(idx) > 0:
        i = idx[0]
        keep.append(i)
        
        if len(idx) == 1:
            break
        
        remaining = []
        for j in idx[1:]:
            iou = compute_iou(boxes[i], boxes[j])
            if iou < threshold:
                remaining.append(j)
        
        idx = np.array(remaining)
    
    return [boxes[i] for i in keep]

def extract_face(frame, bbox):
    """Extract upper portion (head) of detection — generous crop to capture face"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    # Use 40% of body height (instead of 33%) + 30px padding for better face capture
    head_h = int((y2 - y1) * 0.40)
    
    x1 = max(0, x1 - 10)  # Small horizontal padding
    y1 = max(0, y1)
    x2 = min(w, x2 + 10)
    y2 = min(h, y1 + head_h + 30)
    
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2].copy()
    return None

def extract_head_shoulders(frame, bbox):
    """Extract the upper 40% of the detection with generous padding, 
    creating a clear 'head and shoulders' portrait for emotion classification,
    ensuring the face is prominent but context remains."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    bw = x2 - x1
    bh = y2 - y1
    
    head_h = int(bh * 0.40)
    # Add 25% horizontal padding to catch the full head even if leaning
    pad_x = int(bw * 0.25)
    # Add some upward padding just in case the bounding box is a bit low
    pad_y = int(bh * 0.10)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y1 + head_h + pad_y)
    
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2].copy()
    return None

def extract_body(frame, bbox):
    """Extract full body crop with 15% padding for action classification.
    Padding gives CLIP spatial context (desk, phone, notebook)."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    # Add 15% padding around the body
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * 0.15)
    pad_y = int(bh * 0.15)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2].copy()
    return None

# ==================== APPEARANCE EMBEDDING ====================
def compute_appearance_embedding(crop):
    """Compute a CLIP embedding for a body crop (used for re-identification)"""
    if crop is None or crop.size == 0:
        return None
    try:
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        clip_img = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_image(clip_img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()
    except:
        return None

def update_student_gallery(student_id, embedding):
    """Add an embedding to the student's permanent gallery in the registry."""
    global student_registry
    if student_id is None or embedding is None:
        return
    if student_id not in student_registry:
        student_registry[student_id] = {'gallery': [], 'face_img': None}
    gallery = student_registry[student_id]['gallery']
    if len(gallery) < GALLERY_SIZE:
        gallery.append(embedding)
    else:
        # Replace the oldest embedding (FIFO)
        gallery.pop(0)
        gallery.append(embedding)

def gallery_similarity(det_embedding, student_id):
    """Compute max cosine similarity between a detection embedding and a student's
    entire gallery. Returns the highest similarity across all stored embeddings."""
    if student_id not in student_registry:
        return -1.0
    gallery = student_registry[student_id]['gallery']
    if not gallery:
        return -1.0
    max_sim = max(float(np.dot(det_embedding, g)) for g in gallery)
    return max_sim

# ==================== FRAME NORMALIZATION ====================
def normalize_frame(frame):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the luminance
    channel in LAB color space.  This normalizes brightness and contrast so that
    detection and classification are robust across classroom lighting conditions
    (bright morning light, dim evenings, shadows, projector glare, etc.)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

# ==================== RE-IDENTIFICATION ====================
def try_reidentify(detection, frame, frame_id, active_tracks=None, matched_trks=None):
    """Try to match a new detection to a known student using GALLERY-BASED
    appearance matching. Searches:
      1. Dead tracks (graveyard) — students who left the frame
      2. Active tracks with misses > 0 — students not matched this frame
      3. The permanent student registry — ALL ever-seen students

    Uses gallery_similarity (max across all stored embeddings per student)
    for robust matching across lighting, posture, and angle changes.

    No spatial gate — a student may return to any seat.
    """
    global dead_tracks, student_registry

    # Compute appearance embedding for the new detection
    body_crop = extract_body(frame, detection)
    det_embedding = compute_appearance_embedding(body_crop)

    # If no embedding available, skip re-id
    if det_embedding is None:
        return None

    best_student_id = None
    best_match_source = None  # "dead", "active", or "registry"
    best_match_idx = None
    best_score = -1

    # Collect student IDs that are currently active and matched (don't steal these)
    active_matched_sids = set()
    if active_tracks and matched_trks:
        for j in matched_trks:
            if active_tracks[j].student_id is not None:
                active_matched_sids.add(active_tracks[j].student_id)

    # --- Search Dead Tracks (gallery-based) ---
    for i, dead in enumerate(dead_tracks):
        if dead.student_id is None:
            continue
        if dead.student_id in active_matched_sids:
            continue  # This student is already matched by an active track

        sim = gallery_similarity(det_embedding, dead.student_id)
        # Also check single embedding as fallback
        if dead.appearance_embedding is not None:
            single_sim = float(np.dot(det_embedding, dead.appearance_embedding))
            sim = max(sim, single_sim)

        if sim >= REID_APPEARANCE_THRESHOLD and sim > best_score:
            best_score = sim
            best_student_id = dead.student_id
            best_match_idx = i
            best_match_source = "dead"

    # --- Search Active Unmatched Tracks (gallery-based) ---
    if active_tracks:
        for i, track in enumerate(active_tracks):
            if matched_trks and i in matched_trks:
                continue  # Already matched this frame
            if track.misses == 0 or track.student_id is None:
                continue
            if track.student_id in active_matched_sids:
                continue

            sim = gallery_similarity(det_embedding, track.student_id)
            if track.appearance_embedding is not None:
                single_sim = float(np.dot(det_embedding, track.appearance_embedding))
                sim = max(sim, single_sim)

            if sim >= REID_APPEARANCE_THRESHOLD and sim > best_score:
                best_score = sim
                best_student_id = track.student_id
                best_match_idx = i
                best_match_source = "active"

    # --- Search permanent registry for any student not currently tracked ---
    active_sids = set()
    if active_tracks:
        for t in active_tracks:
            if t.student_id is not None:
                active_sids.add(t.student_id)
    dead_sids = set(d.student_id for d in dead_tracks if d.student_id is not None)
    checked_sids = active_sids | dead_sids  # already checked above

    for sid, info in student_registry.items():
        if sid in checked_sids or sid in active_matched_sids:
            continue
        if not info['gallery']:
            continue
        sim = gallery_similarity(det_embedding, sid)
        if sim >= REID_APPEARANCE_THRESHOLD and sim > best_score:
            best_score = sim
            best_student_id = sid
            best_match_source = "registry"

    # --- Apply match ---
    if best_match_source == "dead":
        old_track = dead_tracks.pop(best_match_idx)
        new_track = Track(detection, frame_id)
        new_track.student_id = best_student_id
        new_track.confirmed = True
        new_track.hits = MIN_HITS_TO_CONFIRM
        new_track.face_img = old_track.face_img
        new_track.appearance_embedding = det_embedding
        new_track.last_embedding_frame = frame_id
        update_student_gallery(best_student_id, det_embedding)

        print(f"  ↩ Student {best_student_id} RE-IDENTIFIED from GRAVEYARD "
              f"(gallery_sim={best_score:.3f})")
        return new_track

    elif best_match_source == "active":
        track = active_tracks[best_match_idx]
        z = bbox_to_z(np.array(detection, dtype=np.float32))
        track.kf.x[:4] = z.reshape(-1, 1)
        track.kf.x[4:] = 0.0
        track.bbox = np.array(detection, dtype=np.float32)
        track.smooth_bbox = track.bbox.copy()
        track.last_seen = frame_id
        track.hits += 1
        track.misses = 0
        track.appearance_embedding = det_embedding
        track.last_embedding_frame = frame_id
        update_student_gallery(best_student_id, det_embedding)

        print(f"  ↩ Student {best_student_id} RELOCATED in-frame "
              f"(gallery_sim={best_score:.3f})")
        return track

    elif best_match_source == "registry":
        # Student found in permanent registry but has no active/dead track
        new_track = Track(detection, frame_id)
        new_track.student_id = best_student_id
        new_track.confirmed = True
        new_track.hits = MIN_HITS_TO_CONFIRM
        new_track.face_img = student_registry[best_student_id].get('face_img')
        new_track.appearance_embedding = det_embedding
        new_track.last_embedding_frame = frame_id
        update_student_gallery(best_student_id, det_embedding)

        print(f"  ↩ Student {best_student_id} RE-IDENTIFIED from REGISTRY "
              f"(gallery_sim={best_score:.3f})")
        return new_track

    return None

# ==================== MODEL INITIALIZATION ====================
def init_models():
    global yolo_model, clip_model, clip_preprocess, text_embeddings, emotion_text_embeddings
    global CLIP_CLASSES, CLIP_EMOTIONS
    
    print("=" * 60)
    print("CLASSROOM ENGAGEMENT ANALYSIS SYSTEM (OPTIMIZED)")
    print("=" * 60 + "\n")
    
    print(f"[INFO] Performance settings:")
    print(f"   - Process every {PROCESS_EVERY_N_FRAMES} frame(s)")
    print(f"   - Worker threads: {NUM_WORKERS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Device: {DEVICE}\n")
    
    # Load YOLO
    print("[LOAD] Loading YOLOv8m model...")
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8m.pt")  # Upgraded to Medium model for better dense crowd detection
    print("   [OK] YOLOv8m loaded")
    
    # Load CLIP
    print("[LOAD] Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
    clip_model.eval()
    print("   [OK] CLIP loaded")
    
    # Create CLIP text embeddings for actions
    print("[LOAD] Creating CLIP text embeddings for actions...")
    embeddings = []
    for label, prompts in CLIP_LABELS.items():
        tokens = clip.tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(dim=0, keepdim=True)
        embeddings.append(emb)
    text_embeddings = torch.cat(embeddings, dim=0)
    CLIP_CLASSES = list(CLIP_LABELS.keys())
    print(f"   [OK] Actions: {CLIP_CLASSES}")
    
    # Create CLIP text embeddings for emotions
    print("[LOAD] Creating CLIP text embeddings for emotions...")
    emotion_embeddings_list = []
    for label, prompts in CLIP_EMOTION_LABELS.items():
        tokens = clip.tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(dim=0, keepdim=True)
        emotion_embeddings_list.append(emb)
    emotion_text_embeddings = torch.cat(emotion_embeddings_list, dim=0)
    CLIP_EMOTIONS = list(CLIP_EMOTION_LABELS.keys())
    print(f"   [OK] Emotions: {CLIP_EMOTIONS}")
    
    print("\n[OK] All models loaded!\n")

# ==================== DETECTION ====================
def detect_persons(frame, conf=None):
    """Single-scale detection with strict NMS"""
    if conf is None:
        conf = YOLO_CONF
    
    results = yolo_model(frame, imgsz=IMG_SIZE, conf=conf, classes=[0], verbose=False)
    
    raw_boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            raw_boxes.append(box)
    
    return strict_nms(raw_boxes, NMS_IOU_THRESHOLD)

def detect_persons_registration(frame):
    """Detection during registration"""
    return detect_persons(frame, conf=YOLO_CONF_REGISTRATION)

# ==================== BATCH CLASSIFICATION ====================
def batch_predict_emotions(crops):
    """Batch predict emotions for multiple crops using CLIP and a sharpening filter"""
    if not crops:
        return ["neutral"] * len(crops)
    
    results = []
    valid_indices = []
    valid_images = []
    
    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            results.append("neutral")
        else:
            # VISIBILITY STRATEGY: 
            # If the crop is incredibly small (e.g., face width/height < 30 pixels),
            # it is impossible to read expressions. Force default to neutral to avoid hallucinations.
            if crop.shape[0] < 30 or crop.shape[1] < 30:
                results.append("neutral")
                continue
                
            try:
                # Apply lightweight OpenCV sharpening filter to enhance blurry faces
                kernel = np.array([[0, -1, 0], 
                                   [-1, 5,-1], 
                                   [0, -1, 0]])
                sharpened = cv2.filter2D(crop, -1, kernel)
                
                pil_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
                clip_img = clip_preprocess(pil_img)
                valid_images.append(clip_img)
                valid_indices.append(i)
                results.append(None)  # Placeholder
            except:
                results.append("neutral")
    
    if valid_images:
        try:
            batch_tensor = torch.stack(valid_images).to(DEVICE)
            with torch.no_grad():
                img_embs = clip_model.encode_image(batch_tensor)
                img_embs /= img_embs.norm(dim=-1, keepdim=True)
                similarities = img_embs @ emotion_text_embeddings.T
                
                for idx, sim in zip(valid_indices, similarities):
                    best_idx = sim.argmax().item()
                    best_score = sim[best_idx].item()
                    sorted_scores = torch.sort(sim, descending=True).values
                    second_best = sorted_scores[1].item() if len(sorted_scores) > 1 else 0
                    
                    best_emotion = CLIP_EMOTIONS[best_idx]
                    
                    # Threshold logic: Happy is easier to detect and highly relevant
                    # Fear, Disgust, Angry require a higher burden of proof to prevent hallucinations on bad pixels
                    if best_emotion == "happy":
                        threshold = 0.145
                    else:
                        threshold = 0.185  # Stricter gate for negative emotions
                        
                    # Also require a margin of victory to ensure it's not a generic guess
                    if best_score < threshold or (best_score - second_best) < 0.005:
                        results[idx] = "neutral"
                    else:
                        results[idx] = best_emotion
        except:
            for idx in valid_indices:
                results[idx] = "neutral"
    
    return results

def batch_predict_actions(crops):
    """Batch predict actions for multiple crops using CLIP"""
    if not crops:
        return ["neutral"] * len(crops)
    
    results = []
    valid_indices = []
    valid_images = []
    
    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            results.append("neutral")
        else:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                clip_img = clip_preprocess(pil_img)
                valid_images.append(clip_img)
                valid_indices.append(i)
                results.append(None)  # Placeholder
            except:
                results.append("neutral")
    
    if valid_images:
        try:
            batch_tensor = torch.stack(valid_images).to(DEVICE)
            with torch.no_grad():
                img_embs = clip_model.encode_image(batch_tensor)
                img_embs /= img_embs.norm(dim=-1, keepdim=True)
                similarities = img_embs @ text_embeddings.T
                
                for idx, sim in zip(valid_indices, similarities):
                    best_idx = sim.argmax().item()
                    best_score = sim[best_idx].item()
                    sorted_scores = torch.sort(sim, descending=True).values
                    second_best = sorted_scores[1].item() if len(sorted_scores) > 1 else 0
                    
                    # Lowered thresholds: accept action if CLIP is reasonably confident
                    if best_score < 0.12 or (best_score - second_best) < 0.005:
                        results[idx] = "neutral"
                    else:
                        results[idx] = CLIP_CLASSES[best_idx]
        except:
            for idx in valid_indices:
                results[idx] = "neutral"
    
    return results

# ==================== TRACKING ====================
def merge_overlapping_tracks():
    """Merge any tracks that overlap significantly"""
    global tracks
    
    if len(tracks) < 2:
        return
    
    merged = []
    used = set()
    
    for i, t1 in enumerate(tracks):
        if i in used:
            continue
        
        to_merge = [t1]
        for j, t2 in enumerate(tracks):
            if j <= i or j in used:
                continue
            
            iou = compute_iou(t1.bbox, t2.bbox)
            if iou > OVERLAP_MERGE_THRESHOLD:
                to_merge.append(t2)
                used.add(j)
        
        if len(to_merge) == 1:
            merged.append(t1)
        else:
            best = max(to_merge, key=lambda t: t.hits)
            merged.append(best)
    
    tracks = merged

def assign_ids():
    """Assign student IDs to confirmed tracks.
    
    AFTER registration, NO new student IDs are ever created. Unmatched
    detections that couldn't be re-identified are simply ignored — they
    are either ghosts (chairs, bags) or students who will be matched
    via gallery re-ID on a subsequent frame.
    """
    global next_student_id, json_data, student_registry
    
    for track in tracks:
        if track.is_confirmed() and track.student_id is None:
            # BLOCK new IDs after registration — only re-ID can assign IDs
            if registration_complete:
                # Don't assign a new ID. This track will either:
                # 1. Be matched via re-ID on a future frame
                # 2. Eventually die as unconfirmed/unassigned (ghost)
                continue
            
            track.student_id = next_student_id
            track.confirmed = True
            
            # Register in permanent student registry
            if next_student_id not in student_registry:
                student_registry[next_student_id] = {'gallery': [], 'face_img': track.face_img}
            if track.appearance_embedding is not None:
                update_student_gallery(next_student_id, track.appearance_embedding)
            
            with json_lock:
                if str(next_student_id) not in json_data["students"]:
                    json_data["students"][str(next_student_id)] = {"frames": {}}
            
            print(f"  ★ Student {next_student_id} registered")
            next_student_id += 1

def cluster_detections_by_center(detections, dist_threshold):
    """Group detections by spatial proximity of their centers.
    Dogs returns one representative box per cluster (the median box in each cluster).
    This correctly handles the same seated student appearing with slightly different boxes
    across multiple frames — unlike IoU NMS which would split them into duplicates."""
    if not detections:
        return []
    
    centers = np.array([
        [(d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0] for d in detections
    ])
    
    assigned = [-1] * len(detections)
    cluster_id = 0
    
    for i in range(len(detections)):
        if assigned[i] != -1:
            continue
        assigned[i] = cluster_id
        for j in range(i + 1, len(detections)):
            if assigned[j] != -1:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist <= dist_threshold:
                assigned[j] = cluster_id
        cluster_id += 1
    
    # One representative box per cluster: take median of all boxes in cluster
    result = []
    for cid in range(cluster_id):
        members = [detections[k] for k in range(len(detections)) if assigned[k] == cid]
        if members:
            median_box = np.median(members, axis=0)
            result.append(median_box)
    
    return result


def registration_phase(cap):
    """Scan initial frames to find all students using spatial center-based clustering.
    Replaces NMS-based merging which would split the same slightly-shifted student into
    multiple registration entries."""
    global tracks, registration_complete, next_student_id, MAX_ALLOWED_STUDENTS
    
    print(f"[REG] Registration phase: scanning {REGISTRATION_FRAMES} frames...")
    
    all_detections = []
    
    for i in range(REGISTRATION_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        
        dets = detect_persons_registration(frame)
        frame_area = frame.shape[0] * frame.shape[1]
        
        for det in dets:
            area = (det[2] - det[0]) * (det[3] - det[1])
            if area / frame_area >= MIN_TRACK_AREA_RATIO:
                all_detections.append(det)
        
        if (i + 1) % 30 == 0:
            print(f"   Frame {i + 1}/{REGISTRATION_FRAMES}")
    
    # Step 1: Spatial center-based clustering — groups all boxes from the same physical
    # seat together regardless of per-frame jitter, then takes the median box.
    final_positions = cluster_detections_by_center(all_detections, REGISTRATION_CLUSTER_DIST)
    
    # Step 2: Final IoU NMS pass to handle any remaining overlaps between nearby seats
    final_positions = strict_nms(final_positions, NMS_IOU_THRESHOLD)
    
    for bbox in final_positions:
        new_track = Track(bbox, 0)
        new_track.hits = MIN_HITS_TO_CONFIRM
        tracks.append(new_track)
    
    print(f"   Found {len(tracks)} student positions\n")
    
    # --- Compute initial CLIP embeddings for each student's gallery ---
    # Read a fresh frame to extract body crops and compute CLIP embeddings.
    # This is critical: without initial embeddings, re-ID cannot work on early frames.
    cap.set(cv2.CAP_PROP_POS_FRAMES, REGISTRATION_FRAMES // 2)  # mid-registration frame
    ret_emb, emb_frame = cap.read()
    if ret_emb:
        norm_emb_frame = normalize_frame(emb_frame)
    else:
        norm_emb_frame = None
    
    for track in tracks:
        track.student_id = next_student_id
        track.confirmed = True
        json_data["students"][str(next_student_id)] = {"frames": {}}
        
        # Compute initial CLIP embedding and populate gallery
        student_registry[next_student_id] = {'gallery': [], 'face_img': track.face_img}
        if norm_emb_frame is not None:
            body_crop = extract_body(norm_emb_frame, track.bbox)
            emb = compute_appearance_embedding(body_crop)
            if emb is not None:
                track.appearance_embedding = emb
                track.last_embedding_frame = 0
                update_student_gallery(next_student_id, emb)
        
        print(f"  ★ Student {track.student_id} registered "
              f"(gallery: {len(student_registry[next_student_id]['gallery'])} embeddings)")
        next_student_id += 1
    
    # Read a few more frames spread across registration to add gallery diversity
    sample_frames = [0, REGISTRATION_FRAMES // 4, REGISTRATION_FRAMES * 3 // 4,
                     REGISTRATION_FRAMES - 1]
    for sf in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        ret_sf, sf_frame = cap.read()
        if not ret_sf:
            continue
        norm_sf = normalize_frame(sf_frame)
        for track in tracks:
            body_crop = extract_body(norm_sf, track.bbox)
            emb = compute_appearance_embedding(body_crop)
            if emb is not None:
                update_student_gallery(track.student_id, emb)
    
    total_gallery = sum(len(student_registry[sid]['gallery']) for sid in student_registry)
    print(f"   Registered {len(tracks)} students in permanent registry")
    print(f"   Total gallery embeddings: {total_gallery}\n")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    registration_complete = True

def process_frame_optimized(frame, frame_id):
    """Process frame: normalize for lighting, track, re-id, batch-classify.
    
    4-stage matching pipeline:
      Stage 1: IoU-based Hungarian assignment
      Stage 2: Center-distance based matching
      Stage 3: Gallery-based appearance re-identification
      Stage 4: New track creation (last resort, only if no registry match)
    """
    global tracks, dead_tracks, json_data, student_registry

    # Normalize a copy for all AI processing (YOLO, emotion, action, Re-ID).
    norm_frame = normalize_frame(frame)

    detections = detect_persons(norm_frame)
    
    frame_area = norm_frame.shape[0] * norm_frame.shape[1]
    detections = [d for d in detections
                  if (d[2] - d[0]) * (d[3] - d[1]) / frame_area >= MIN_TRACK_AREA_RATIO]
    
    for track in tracks:
        track.predict()
    
    matched_dets = set()
    matched_trks = set()
    
    # ==================== STAGE 1: IoU-based Hungarian matching ====================
    if len(tracks) > 0 and len(detections) > 0:
        cost = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                iou = compute_iou(det, track.bbox)
                cost[i, j] = 1 - iou
        
        row_idx, col_idx = linear_sum_assignment(cost)
        
        for i, j in zip(row_idx, col_idx):
            if cost[i, j] < (1 - IOU_MATCH_THRESHOLD):
                face_img = extract_face(norm_frame, detections[i])
                tracks[j].update(detections[i], frame_id, face_img)
                matched_dets.add(i)
                matched_trks.add(j)

    # ==================== STAGE 2: Center-distance matching ====================
    # For unmatched detections and tracks, match by proximity of bounding box centers.
    # This handles students who shift position, lean, or move slightly between frames.
    unmatched_det_indices = [i for i in range(len(detections)) if i not in matched_dets]
    unmatched_trk_indices = [j for j in range(len(tracks)) if j not in matched_trks]
    
    if unmatched_det_indices and unmatched_trk_indices:
        dist_cost = np.full((len(unmatched_det_indices), len(unmatched_trk_indices)), 1e6)
        for di, i in enumerate(unmatched_det_indices):
            det_center = np.array([(detections[i][0]+detections[i][2])/2, (detections[i][1]+detections[i][3])/2])
            for dj, j in enumerate(unmatched_trk_indices):
                trk_center = tracks[j].get_center()
                dist_cost[di, dj] = np.linalg.norm(det_center - trk_center)
        
        d_rows, d_cols = linear_sum_assignment(dist_cost)
        for di, dj in zip(d_rows, d_cols):
            if dist_cost[di, dj] < CENTER_DIST_MATCH_THRESHOLD:
                i = unmatched_det_indices[di]
                j = unmatched_trk_indices[dj]
                face_img = extract_face(norm_frame, detections[i])
                tracks[j].update(detections[i], frame_id, face_img)
                matched_dets.add(i)
                matched_trks.add(j)
    
    # ==================== STAGE 3: Gallery-based Appearance Re-ID ====================
    # For still-unmatched detections, try to match against the PERMANENT student
    # registry (gallery of embeddings), dead tracks, and active missing tracks.
    for i, det in enumerate(detections):
        if i in matched_dets:
            continue

        resurrected = try_reidentify(det, norm_frame, frame_id,
                                     active_tracks=tracks,
                                     matched_trks=matched_trks)
        if resurrected is not None:
            if resurrected not in tracks:
                tracks.append(resurrected)
            matched_dets.add(i)
        else:
            # ==================== STAGE 4: New track (last resort) ====================
            # Only create a new track if no existing student matches AND no
            # confirmed track is nearby (spatial proximity guard).
            det_center = np.array([(det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0])
            too_close = False
            for track in tracks:
                if track.confirmed:
                    dist = np.linalg.norm(det_center - track.get_center())
                    if dist < NEW_TRACK_GUARD_DIST:
                        too_close = True
                        break

            if not too_close:
                new_track = Track(det, frame_id)
                tracks.append(new_track)
            matched_dets.add(i)
    
    # --- Move dead tracks to graveyard (for future re-identification) ---
    surviving_tracks = []
    for track in tracks:
        if track.is_dead():
            if track.confirmed and track.student_id is not None:
                dead_tracks.append(track)
            # else: unconfirmed track, just discard
        else:
            surviving_tracks.append(track)
    tracks = surviving_tracks
    
    # Clean up very old dead tracks
    dead_tracks = [d for d in dead_tracks if (frame_id - d.last_seen) <= REID_MAX_DEAD_AGE]
    
    merge_overlapping_tracks()
    assign_ids()
    
    # --- Update appearance embeddings periodically for confirmed tracks ---
    # Store in both the track's single embedding AND the permanent gallery.
    for track in tracks:
        if track.confirmed and track.student_id is not None and \
           (frame_id - track.last_embedding_frame) >= APPEARANCE_UPDATE_INTERVAL:
            body_crop = extract_body(norm_frame, track.smooth_bbox)
            emb = compute_appearance_embedding(body_crop)
            if emb is not None:
                track.appearance_embedding = emb
                track.last_embedding_frame = frame_id
                # Add to permanent student gallery
                update_student_gallery(track.student_id, emb)
    
    # Collect all confirmed tracks that were recently seen (e.g. <= 15 frames missed)
    # This prevents drawing boxes for students completely out of view
    confirmed_tracks = [t for t in tracks if t.confirmed and t.misses <= 15]
    
    if not confirmed_tracks:
        return []
    
    # Extract all crops from the normalized frame (better classification under bad lighting)
    hs_crops = [extract_head_shoulders(norm_frame, t.smooth_bbox) for t in confirmed_tracks]
    body_crops = [extract_body(norm_frame, t.smooth_bbox) for t in confirmed_tracks]
    
    # Batch classify
    # Emotions take the Head & Shoulders crop to prioritize the face
    emotions = batch_predict_emotions(hs_crops)
    actions = batch_predict_actions(body_crops)
    
    # Update tracks and prepare results
    results = []
    for track, emotion, action in zip(confirmed_tracks, emotions, actions):
        track.update_emotion(emotion)
        track.update_action(action)
        
        # Store in JSON (thread-safe)
        student_id_str = str(track.student_id)
        frame_id_str = str(frame_id)
        
        with json_lock:
            if student_id_str not in json_data["students"]:
                json_data["students"][student_id_str] = {"frames": {}}
            json_data["students"][student_id_str]["frames"][frame_id_str] = {
                "emotion": track.current_emotion,
                "action": track.current_action
            }
        
        results.append((track.smooth_bbox, track.student_id, track.current_emotion, track.current_action))
    
    return results

# ==================== DRAWING ====================
def draw(frame, assignments, frame_id):
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
        (128, 128, 255), (255, 128, 128), (128, 255, 128), (255, 255, 128)
    ]
    
    for bbox, sid, emotion, action in assignments:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        
        color = colors[(sid - 1) % len(colors)]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"S{sid}: {action}/{emotion}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    n = len(assignments)
    cv2.putText(frame, f"Students: {n}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_id}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    return frame

# ==================== ENGAGEMENT SUMMARY ====================
def generate_summary():
    """Generate engagement summary from JSON data"""
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("CLASSROOM ENGAGEMENT SUMMARY")
    summary_lines.append("=" * 60)
    
    total_students = len(json_data["students"])
    summary_lines.append(f"\nTotal Students: {total_students}")
    
    distracted_actions = ["using_mobile", "looking_away", "sleeping"]
    
    all_emotions = Counter()
    all_actions = Counter()
    per_student_engagement = {}
    
    for sid, data in json_data["students"].items():
        frames = data["frames"]
        total_frames = len(frames)
        
        emotions = Counter([f["emotion"] for f in frames.values()])
        actions = Counter([f["action"] for f in frames.values()])
        
        all_emotions.update(emotions)
        all_actions.update(actions)
        
        distracted = sum(actions.get(a, 0) for a in distracted_actions)
        engaged = total_frames - distracted
        engagement_pct = (engaged / total_frames * 100) if total_frames > 0 else 0
        per_student_engagement[sid] = engagement_pct
        
        summary_lines.append(f"\nStudent {sid}:")
        summary_lines.append(f"  Frames tracked: {total_frames}")
        summary_lines.append(f"  Top emotion: {emotions.most_common(1)[0][0] if emotions else 'N/A'}")
        summary_lines.append(f"  Top action: {actions.most_common(1)[0][0] if actions else 'N/A'}")
        summary_lines.append(f"  Engagement: {engagement_pct:.1f}%")
    
    summary_lines.append("\n" + "-" * 60)
    summary_lines.append("OVERALL STATISTICS")
    summary_lines.append("-" * 60)
    summary_lines.append(f"\nEmotion distribution: {dict(all_emotions.most_common())}")
    summary_lines.append(f"Action distribution: {dict(all_actions.most_common())}")
    
    avg_engagement = sum(per_student_engagement.values()) / len(per_student_engagement) if per_student_engagement else 0
    summary_lines.append(f"\nAverage class engagement: {avg_engagement:.1f}%")
    
    return "\n".join(summary_lines)

# ==================== POST-PROCESSING ====================
def clean_and_renumber_students(total_video_frames):
    """Remove ghost/phantom student entries and re-number surviving students
    with clean sequential IDs (1, 2, 3, ...).

    A student must have been tracked for at least MIN_FRAMES_FRACTION of the
    total video frames to be considered a real detection and kept in the output.
    Face images in FACES_DIR are also renamed to match the new IDs.
    """
    global json_data

    min_frames = int(total_video_frames * MIN_FRAMES_FRACTION)
    print(f"\n[CLEAN] Cleaning output — minimum frames threshold: {min_frames} "
          f"({MIN_FRAMES_FRACTION*100:.0f}% of {total_video_frames} frames)")

    # --- Step 1: Collect surviving students (sorted by original ID for stability) ---
    survivors = []
    removed = []
    for sid, data in sorted(json_data["students"].items(), key=lambda x: int(x[0])):
        frame_count = len(data["frames"])
        if frame_count >= min_frames:
            survivors.append((sid, data))
        else:
            removed.append(sid)
            print(f"   [CUT] Removed Student {sid} ({frame_count} frames -- below threshold)")

    print(f"   [OK] Kept {len(survivors)} real students, removed {len(removed)} ghosts")

    # --- Step 2: Build clean re-numbered students dict ---
    new_students = {}
    id_remap = {}  # old_id -> new_id string
    for new_idx, (old_sid, data) in enumerate(survivors, start=1):
        new_id = str(new_idx)
        id_remap[old_sid] = new_id
        new_students[new_id] = data

    # --- Step 3: Update global json_data ---
    json_data["students"] = new_students
    json_data["video_info"]["total_students"] = len(new_students)

    print(f"   \U0001f4ca Final student count in output: {len(new_students)}")
    return len(new_students)

# ==================== MAIN ====================
def main():
    import argparse
    global json_data

    parser = argparse.ArgumentParser(description="Classroom Engagement Analysis")
    parser.add_argument(
        "--video",
        default=os.path.join(BASE_DIR, "final_video1.mp4"),
        help="Path to input video file"
    )
    parser.add_argument("--output-json",    default=None, help="Path for output JSON file")
    parser.add_argument("--output-video",   default=None, help="Path for annotated output video")
    parser.add_argument("--output-summary", default=None, help="Path for summary text file")
    parser.add_argument("--faces-dir",      default=None, help="Directory to save face crops (unused in new pipeline)")
    parser.add_argument("--emotion-model",  default=None, help="Unused — kept for CLI compatibility")
    parser.add_argument("--class-map",      default=None, help="Unused — kept for CLI compatibility")
    args = parser.parse_args()
    video_path = args.video

    # Derive output paths — use CLI args if provided, else auto-derive from video stem
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path   = args.output_video   or os.path.join(BASE_DIR, "outputs", f"output_{video_stem}.mp4")
    output_json_path    = args.output_json    or os.path.join(BASE_DIR, "outputs", f"output_{video_stem}.json")
    output_summary_path = args.output_summary or os.path.join(BASE_DIR, "outputs", f"output_{video_stem}_summary.txt")

    init_models()

    if not os.path.exists(video_path):
        print(f"\u274c ERROR: Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\U0001f4f9 Video: {video_path}")
    print(f"   {w}x{h} @ {fps:.1f}fps, {total} frames")
    print(f"\U0001f4c1 Output video: {output_video_path}")
    print(f"\U0001f4c1 Output JSON:  {output_json_path}\n")
    
    json_data["video_info"] = {
        "video": video_path,
        "fps": fps,
        "width": w,
        "height": h,
        "total_frames": total
    }

    registration_phase(cap)

    json_data["video_info"]["total_students"] = next_student_id - 1

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    start = time.time()
    frame_id = 0
    last_assignments = []
    
    print("[RUN] Processing frames (OPTIMIZED)...\n")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            
            # Process every N frames or use cached results
            if frame_id % PROCESS_EVERY_N_FRAMES == 0:
                last_assignments = process_frame_optimized(frame, frame_id)
            else:
                # For skipped frames, use last known assignments but update JSON
                for track in tracks:
                    if track.confirmed:
                        student_id_str = str(track.student_id)
                        frame_id_str = str(frame_id)
                        with json_lock:
                            json_data["students"][student_id_str]["frames"][frame_id_str] = {
                                "emotion": track.current_emotion,
                                "action": track.current_action
                            }
            
            annotated = draw(frame.copy(), last_assignments, frame_id)
            out.write(annotated)
            
            if frame_id % 100 == 0:
                elapsed = time.time() - start
                speed = frame_id / elapsed
                print(f"   Frame {frame_id}/{total} | Students: {len(last_assignments)} | Speed: {speed:.1f} fps")
    
    finally:
        cap.release()
        out.release()
    
    elapsed = time.time() - start

    # ---- POST-PROCESSING: Remove ghost students, re-number clean IDs ----
    clean_count = clean_and_renumber_students(total)
    json_data["video_info"]["total_students"] = clean_count

    print(f"\n\U0001f4be Saving JSON to: {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    summary = generate_summary()
    print(f"\U0001f4be Saving summary to: {output_summary_path}")
    with open(output_summary_path, "w") as f:
        f.write(summary)
    
    print("\n" + summary)
    
    print("\n" + "=" * 60)
    print("\u2705 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\n\u23f1\ufe0f  Time: {elapsed:.1f}s ({total/elapsed:.1f} fps)")
    print(f"\U0001f465 Real students in output: {clean_count}")
    print(f"\n\U0001f4c1 Output video:   {output_video_path}")
    print(f"\U0001f4c1 Output JSON:    {output_json_path}")
    print(f"\U0001f4c1 Summary:        {output_summary_path}")

if __name__ == "__main__":
    main()
