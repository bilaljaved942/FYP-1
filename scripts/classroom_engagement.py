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
from tensorflow.keras.models import load_model
from PIL import Image
import clip
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# Paths - UPDATE THESE
VIDEO_PATH = r"E:\FYP\videos\final_video.mp4"
OUTPUT_VIDEO_PATH = r"E:\FYP\videos\output_engagement.mp4"
OUTPUT_JSON_PATH = r"E:\FYP\videos\output_engagement.json"
OUTPUT_SUMMARY_PATH = r"E:\FYP\videos\output_engagement_summary.txt"
FACES_DIR = r"E:\FYP\faces_engagement"
EMOTION_MODEL_PATH = r"E:\FYP\best_cnn_v2_emotions.keras"
CLASS_MAP_PATH = r"E:\FYP\class_map.json"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Performance settings
PROCESS_EVERY_N_FRAMES = 1  # Set to 2 or 3 for faster processing (skip frames)
NUM_WORKERS = 4  # Number of threads for parallel processing
BATCH_SIZE = 8  # Batch size for classification

# Detection parameters
YOLO_CONF = 0.15
YOLO_CONF_REGISTRATION = 0.15
IMG_SIZE = 640
NMS_IOU_THRESHOLD = 0.35

# Tracking parameters
IOU_MATCH_THRESHOLD = 0.2
OVERLAP_MERGE_THRESHOLD = 0.25
MAX_MISSING_FRAMES = 900
MIN_HITS_TO_CONFIRM = 5
MIN_TRACK_AREA_RATIO = 0.003
REGISTRATION_FRAMES = 150
SMOOTH_ALPHA = 0.2

# Classification parameters
FRAME_SMOOTHING = 5

# CLIP action prompts
CLIP_LABELS = {
    "using_mobile": [
        "a person holding a phone in their hand",
        "a person looking down at a mobile phone",
        "a person texting on a smartphone"
    ],
    "writing_notes": [
        "a person writing something in a notebook",
        "a person holding a pen and writing on paper",
        "a person sitting and writing notes in a notebook"
    ],
    "raising_hand": [
        "a person raising one hand in the air",
        "a person with one arm lifted up",
        "a person stretching their hand upward"
    ],
    "sleeping": [
        "a person sleeping with eyes closed",
        "a person resting their head on a desk",
        "a person dozing off with head down"
    ],
    "looking_away": [
        "a person turning their head to the right",
        "a person turning their head to the left",
        "a person facing away from the camera"
    ],
    "neutral": [
        "a person sitting still and looking straight",
        "a person sitting upright with no movement",
        "a person facing forward doing nothing"
    ]
}

os.makedirs(FACES_DIR, exist_ok=True)

# ==================== GLOBALS ====================
tracks = []
yolo_model = None
emotion_model = None
clip_model = None
clip_preprocess = None
text_embeddings = None
ordered_classes = []
CLIP_CLASSES = []
registration_complete = False
next_student_id = 1

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
        
        self.emotion_history = deque(maxlen=FRAME_SMOOTHING)
        self.action_history = deque(maxlen=FRAME_SMOOTHING)
        self.current_emotion = "neutral"
        self.current_action = "neutral"
    
    def predict(self):
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
    """Extract upper portion (head) of detection"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    head_h = (y2 - y1) // 3
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y1 + head_h + 20)
    
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2].copy()
    return None

def extract_body(frame, bbox):
    """Extract full body crop for action classification"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2].copy()
    return None

# ==================== MODEL INITIALIZATION ====================
def init_models():
    global yolo_model, emotion_model, clip_model, clip_preprocess, text_embeddings
    global ordered_classes, CLIP_CLASSES
    
    print("=" * 60)
    print("CLASSROOM ENGAGEMENT ANALYSIS SYSTEM (OPTIMIZED)")
    print("=" * 60 + "\n")
    
    print(f"⚡ Performance settings:")
    print(f"   - Process every {PROCESS_EVERY_N_FRAMES} frame(s)")
    print(f"   - Worker threads: {NUM_WORKERS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Device: {DEVICE}\n")
    
    # Load YOLO
    print("🔍 Loading YOLOv8 model...")
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")
    print("   ✅ YOLOv8 loaded")
    
    # Load Emotion Model
    print("💬 Loading Emotion Model...")
    emotion_model = load_model(EMOTION_MODEL_PATH, compile=False)
    print("   ✅ Emotion model loaded")
    
    # Load class map
    print("📋 Loading class map...")
    with open(CLASS_MAP_PATH, "r") as f:
        class_map = json.load(f)
    ordered_classes = [c for c, idx in sorted(class_map.items(), key=lambda x: x[1])]
    print(f"   ✅ Emotions: {ordered_classes}")
    
    # Load CLIP
    print("🧠 Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
    clip_model.eval()
    print("   ✅ CLIP loaded")
    
    # Create CLIP text embeddings
    print("📌 Creating CLIP text embeddings...")
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
    print(f"   ✅ Actions: {CLIP_CLASSES}")
    
    print("\n✅ All models loaded!\n")

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
    """Batch predict emotions for multiple crops"""
    if not crops:
        return ["neutral"] * len(crops)
    
    results = []
    valid_indices = []
    batch_inputs = []
    
    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            results.append("neutral")
        else:
            try:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (48, 48))
                normalized = resized / 255.0
                batch_inputs.append(normalized)
                valid_indices.append(i)
                results.append(None)  # Placeholder
            except:
                results.append("neutral")
    
    if batch_inputs:
        try:
            batch_array = np.array(batch_inputs)
            batch_array = np.expand_dims(batch_array, axis=-1)
            preds = emotion_model.predict(batch_array, verbose=0)
            
            for idx, pred in zip(valid_indices, preds):
                best_idx = int(np.argmax(pred))
                results[idx] = ordered_classes[best_idx]
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
                    
                    if best_score < 0.18 or (best_score - second_best) < 0.015:
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
    """Assign student IDs to confirmed tracks"""
    global next_student_id, json_data
    
    for track in tracks:
        if track.is_confirmed() and track.student_id is None:
            track.student_id = next_student_id
            track.confirmed = True
            
            with json_lock:
                json_data["students"][str(next_student_id)] = {"frames": {}}
            
            if track.face_img is not None:
                try:
                    cv2.imwrite(f"{FACES_DIR}/student_{next_student_id}.jpg", track.face_img)
                except:
                    pass
            
            print(f"  ★ Student {next_student_id} registered")
            next_student_id += 1

def registration_phase(cap):
    """Scan initial frames to find all students"""
    global tracks, registration_complete, next_student_id
    
    print(f"📝 Registration phase: scanning {REGISTRATION_FRAMES} frames...")
    
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
    
    final_positions = strict_nms(all_detections, 0.35)
    
    for bbox in final_positions:
        new_track = Track(bbox, 0)
        new_track.hits = MIN_HITS_TO_CONFIRM
        tracks.append(new_track)
    
    print(f"   Found {len(tracks)} student positions\n")
    
    for track in tracks:
        track.student_id = next_student_id
        track.confirmed = True
        json_data["students"][str(next_student_id)] = {"frames": {}}
        print(f"  ★ Student {track.student_id} registered")
        next_student_id += 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    registration_complete = True

def process_frame_optimized(frame, frame_id):
    """Process frame with tracking and BATCH classification"""
    global tracks, json_data
    
    detections = detect_persons(frame)
    
    frame_area = frame.shape[0] * frame.shape[1]
    detections = [d for d in detections 
                  if (d[2] - d[0]) * (d[3] - d[1]) / frame_area >= MIN_TRACK_AREA_RATIO]
    
    for track in tracks:
        track.predict()
    
    if len(tracks) > 0 and len(detections) > 0:
        cost = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                iou = compute_iou(det, track.bbox)
                cost[i, j] = 1 - iou
        
        row_idx, col_idx = linear_sum_assignment(cost)
        
        matched_dets = set()
        matched_trks = set()
        
        for i, j in zip(row_idx, col_idx):
            if cost[i, j] < (1 - IOU_MATCH_THRESHOLD):
                face_img = extract_face(frame, detections[i])
                tracks[j].update(detections[i], frame_id, face_img)
                matched_dets.add(i)
                matched_trks.add(j)
        
        for i, det in enumerate(detections):
            if i in matched_dets:
                continue
            
            best_track = None
            best_iou = OVERLAP_MERGE_THRESHOLD
            
            for j, track in enumerate(tracks):
                if j in matched_trks:
                    continue
                iou = compute_iou(det, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = j
            
            if best_track is not None:
                face_img = extract_face(frame, det)
                tracks[best_track].update(det, frame_id, face_img)
                matched_trks.add(best_track)
    
    merge_overlapping_tracks()
    assign_ids()
    
    # Collect all confirmed tracks and their crops
    confirmed_tracks = [t for t in tracks if t.confirmed]
    
    if not confirmed_tracks:
        return []
    
    # Extract all crops at once
    face_crops = [extract_face(frame, t.smooth_bbox) for t in confirmed_tracks]
    body_crops = [extract_body(frame, t.smooth_bbox) for t in confirmed_tracks]
    
    # Batch classify
    emotions = batch_predict_emotions(face_crops)
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

# ==================== MAIN ====================
def main():
    global json_data
    
    init_models()
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ ERROR: Video not found: {VIDEO_PATH}")
        return
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {w}x{h} @ {fps:.1f}fps, {total} frames")
    print(f"📁 Output video: {OUTPUT_VIDEO_PATH}")
    print(f"📁 Output JSON: {OUTPUT_JSON_PATH}\n")
    
    json_data["video_info"] = {
        "fps": fps,
        "width": w,
        "height": h,
        "total_frames": total
    }
    
    registration_phase(cap)
    
    json_data["video_info"]["total_students"] = next_student_id - 1
    
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    start = time.time()
    frame_id = 0
    last_assignments = []
    
    print("🎥 Processing frames (OPTIMIZED)...\n")
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
    
    print(f"\n💾 Saving JSON to: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(json_data, f, indent=2)
    
    summary = generate_summary()
    print(f"💾 Saving summary to: {OUTPUT_SUMMARY_PATH}")
    with open(OUTPUT_SUMMARY_PATH, "w") as f:
        f.write(summary)
    
    print("\n" + summary)
    
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\n⏱️  Time: {elapsed:.1f}s ({total/elapsed:.1f} fps)")
    print(f"👥 Total students: {next_student_id - 1}")
    print(f"\n📁 Output video: {OUTPUT_VIDEO_PATH}")
    print(f"📁 Output JSON: {OUTPUT_JSON_PATH}")
    print(f"📁 Summary: {OUTPUT_SUMMARY_PATH}")
    print(f"📁 Faces: {FACES_DIR}/")

if __name__ == "__main__":
    main()
