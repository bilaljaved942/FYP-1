"""
Unit tests for app.ai_utils — validates the data transformation
from raw per-frame AI output into per-student aggregated format.
"""

import json
import os
import tempfile

from app.ai_utils import is_engaged, transform_ai_output


# ── Engagement rule tests ─────────────────────────────────────────


class TestIsEngaged:
    """Verify the engagement scoring rules."""

    def test_disengaged_actions_always_false(self):
        for action in ("using_mobile", "sleeping", "looking_away"):
            assert is_engaged("happy", action) is False
            assert is_engaged("neutral", action) is False

    def test_raising_hand_always_engaged(self):
        for emotion in ("happy", "angry", "sad", "neutral", "fear"):
            assert is_engaged(emotion, "raising_hand") is True

    def test_writing_notes_with_positive_emotion(self):
        assert is_engaged("happy", "writing_notes") is True
        assert is_engaged("neutral", "writing_notes") is True
        assert is_engaged("surprise", "writing_notes") is True

    def test_writing_notes_with_negative_emotion(self):
        assert is_engaged("angry", "writing_notes") is False
        assert is_engaged("sad", "writing_notes") is False
        assert is_engaged("fear", "writing_notes") is False

    def test_neutral_action_positive_emotion(self):
        assert is_engaged("happy", "neutral") is True
        assert is_engaged("neutral", "neutral") is True
        assert is_engaged("surprise", "neutral") is True

    def test_neutral_action_negative_emotion(self):
        assert is_engaged("angry", "neutral") is False
        assert is_engaged("sad", "neutral") is False
        assert is_engaged("fear", "neutral") is False
        assert is_engaged("disgust", "neutral") is False


# ── Transformer tests ─────────────────────────────────────────────


def _make_raw_json(students: dict, fps: float = 30.0) -> str:
    """Write a synthetic raw JSON to a temp file and return the path."""
    data = {
        "video_info": {"fps": fps, "width": 640, "height": 480, "total_frames": 90},
        "students": students,
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return path


class TestTransformAiOutput:
    """Verify the transform_ai_output aggregation function."""

    def test_basic_single_student(self):
        raw_students = {
            "1": {
                "frames": {
                    "1": {"emotion": "happy", "action": "neutral"},
                    "2": {"emotion": "happy", "action": "neutral"},
                    "3": {"emotion": "angry", "action": "sleeping"},
                }
            }
        }
        path = _make_raw_json(raw_students)
        try:
            result = transform_ai_output(path)

            assert "students" in result
            assert len(result["students"]) == 1

            s = result["students"][0]
            assert s["student_id"] == "1"
            assert s["emotions"] == {"happy": 2, "angry": 1}
            assert s["actions"] == {"neutral": 2, "sleeping": 1}
            assert isinstance(s["engagement_over_time"], list)
            assert len(s["engagement_over_time"]) > 0
        finally:
            os.unlink(path)

    def test_multiple_students(self):
        raw_students = {
            "1": {
                "frames": {
                    "1": {"emotion": "neutral", "action": "neutral"},
                }
            },
            "2": {
                "frames": {
                    "1": {"emotion": "sad", "action": "sleeping"},
                }
            },
        }
        path = _make_raw_json(raw_students)
        try:
            result = transform_ai_output(path)
            assert len(result["students"]) == 2

            ids = {s["student_id"] for s in result["students"]}
            assert ids == {"1", "2"}
        finally:
            os.unlink(path)

    def test_empty_student_frames(self):
        """A student with zero frames should still appear in output."""
        raw_students = {"5": {"frames": {}}}
        path = _make_raw_json(raw_students)
        try:
            result = transform_ai_output(path)
            assert len(result["students"]) == 1

            s = result["students"][0]
            assert s["student_id"] == "5"
            assert s["emotions"] == {}
            assert s["actions"] == {}
            assert s["engagement_over_time"] == []
        finally:
            os.unlink(path)

    def test_engagement_score_bounds(self):
        """All scores must be between 0 and 100."""
        raw_students = {
            "1": {
                "frames": {
                    str(i): {"emotion": "happy", "action": "neutral"}
                    for i in range(1, 91)  # 90 frames = 3 seconds at 30fps
                }
            }
        }
        path = _make_raw_json(raw_students, fps=30.0)
        try:
            result = transform_ai_output(path)
            s = result["students"][0]
            for entry in s["engagement_over_time"]:
                assert 0 <= entry["score"] <= 100
                assert "second" in entry
        finally:
            os.unlink(path)

    def test_engagement_per_second_grouping(self):
        """Frames in the same second should be grouped together."""
        # At 2fps: frames 1-2 = second 1, frames 3-4 = second 2
        raw_students = {
            "1": {
                "frames": {
                    "1": {"emotion": "happy", "action": "neutral"},      # engaged, sec 1
                    "2": {"emotion": "angry", "action": "sleeping"},     # disengaged, sec 1
                    "3": {"emotion": "happy", "action": "writing_notes"},# engaged, sec 2
                    "4": {"emotion": "happy", "action": "writing_notes"},# engaged, sec 2
                }
            }
        }
        path = _make_raw_json(raw_students, fps=2.0)
        try:
            result = transform_ai_output(path)
            timeline = result["students"][0]["engagement_over_time"]

            # Should have 2 seconds
            assert len(timeline) == 2

            sec1 = next(e for e in timeline if e["second"] == 1)
            sec2 = next(e for e in timeline if e["second"] == 2)

            assert sec1["score"] == 50   # 1/2 engaged
            assert sec2["score"] == 100  # 2/2 engaged
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing input."""
        try:
            transform_ai_output("/nonexistent/path.json")
            assert False, "Should have raised"
        except FileNotFoundError:
            pass

    def test_output_matches_target_schema(self):
        """Verify the output matches the exact target JSON schema."""
        raw_students = {
            "1": {
                "frames": {
                    "1": {"emotion": "happy", "action": "raising_hand"},
                    "30": {"emotion": "neutral", "action": "neutral"},
                    "60": {"emotion": "angry", "action": "sleeping"},
                }
            }
        }
        path = _make_raw_json(raw_students, fps=30.0)
        try:
            result = transform_ai_output(path)
            s = result["students"][0]

            # Required keys
            assert set(s.keys()) == {"student_id", "emotions", "actions", "engagement_over_time"}
            assert isinstance(s["student_id"], str)
            assert isinstance(s["emotions"], dict)
            assert isinstance(s["actions"], dict)
            assert isinstance(s["engagement_over_time"], list)

            # Emotion/action values must be ints
            for v in s["emotions"].values():
                assert isinstance(v, int)
            for v in s["actions"].values():
                assert isinstance(v, int)

            # Timeline entries must have "second" and "score"
            for entry in s["engagement_over_time"]:
                assert set(entry.keys()) == {"second", "score"}
                assert isinstance(entry["second"], int)
                assert isinstance(entry["score"], int)
        finally:
            os.unlink(path)
