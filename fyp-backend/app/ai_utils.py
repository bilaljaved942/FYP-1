"""
AI Utilities — Data Transformation & Engagement Scoring

Reads the raw per-frame JSON produced by classroom_engagement.py and
transforms it into a per-student aggregated format suitable for the
frontend dashboard.

Target output structure (stored in AnalysisJob.ai_results):
{
  "students": [
    {
      "student_id": "1",
      "emotions":   {"angry": 4, "neutral": 120, ...},
      "actions":    {"sleeping": 10, "writing_notes": 50, ...},
      "engagement_over_time": [{"second": 1, "score": 85}, ...]
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


# ── Engagement scoring rules ──────────────────────────────────────
# Mirrors the proven logic from scripts/format_engagement_json.py

ENGAGED_ACTIONS = {"writing_notes", "raising_hand"}
DISENGAGED_ACTIONS = {"using_mobile", "sleeping", "looking_away"}
POSITIVE_EMOTIONS = {"happy", "neutral", "surprise"}
NEGATIVE_EMOTIONS = {"angry", "sad", "fear", "disgust"}


def is_engaged(emotion: str, action: str) -> bool:
    """
    Determine if a student is engaged for a single frame.

    Rules:
      1. Disengaged actions (mobile/sleeping/looking_away) → always False
      2. Negative emotion + non-engaged action → False
      3. Raising hand (any emotion) → True
      4. Writing notes + non-negative emotion → True
      5. Neutral action + positive emotion → True
      6. Everything else → False
    """
    if action in DISENGAGED_ACTIONS:
        return False
    if emotion in NEGATIVE_EMOTIONS and action not in ENGAGED_ACTIONS:
        return False
    if action == "raising_hand":
        return True
    if action == "writing_notes" and emotion not in NEGATIVE_EMOTIONS:
        return True
    if action == "neutral" and emotion in POSITIVE_EMOTIONS:
        return True
    return False


# ── Main transformer ─────────────────────────────────────────────


def transform_ai_output(raw_json_path: str) -> dict[str, Any]:
    """
    Read the raw per-frame AI output and aggregate into per-student stats.

    Parameters
    ----------
    raw_json_path : str
        Path to the ``output_engagement.json`` produced by the AI script.

    Returns
    -------
    dict
        ``{"students": [ {student_id, emotions, actions, engagement_over_time}, … ]}``
    """
    path = Path(raw_json_path)
    if not path.exists():
        raise FileNotFoundError(f"AI output not found: {raw_json_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data: dict[str, Any] = json.load(f)

    video_info = raw_data.get("video_info", {})
    fps: float = video_info.get("fps", 30.0)
    students_raw: dict[str, Any] = raw_data.get("students", {})

    students_list: list[dict[str, Any]] = []

    for student_id, student_data in students_raw.items():
        frames: dict[str, dict[str, str]] = student_data.get("frames", {})

        if not frames:
            # Still include the student but with empty data
            students_list.append(
                {
                    "student_id": str(student_id),
                    "emotions": {},
                    "actions": {},
                    "engagement_over_time": [],
                }
            )
            continue

        # ── Count emotion / action totals ──
        emotion_counter: Counter[str] = Counter()
        action_counter: Counter[str] = Counter()

        # ── Per-second engagement buckets ──
        #   {second: {"engaged": int, "total": int}}
        seconds_data: dict[int, dict[str, int]] = {}

        for frame_id_str, frame_data in frames.items():
            emotion = frame_data.get("emotion", "neutral")
            action = frame_data.get("action", "neutral")

            emotion_counter[emotion] += 1
            action_counter[action] += 1

            # Which second does this frame belong to? (1-indexed)
            frame_id = int(frame_id_str)
            second = math.ceil(frame_id / fps) if fps > 0 else 1

            if second not in seconds_data:
                seconds_data[second] = {"engaged": 0, "total": 0}

            seconds_data[second]["total"] += 1
            if is_engaged(emotion, action):
                seconds_data[second]["engaged"] += 1

        # ── Build engagement_over_time list ──
        engagement_over_time: list[dict[str, int]] = []
        for sec in sorted(seconds_data.keys()):
            bucket = seconds_data[sec]
            score = (
                int((bucket["engaged"] / bucket["total"]) * 100)
                if bucket["total"] > 0
                else 0
            )
            engagement_over_time.append({"second": sec, "score": score})

        students_list.append(
            {
                "student_id": str(student_id),
                "emotions": dict(emotion_counter),
                "actions": dict(action_counter),
                "engagement_over_time": engagement_over_time,
            }
        )

    return {"students": students_list}
