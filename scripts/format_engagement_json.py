"""
ENGAGEMENT JSON FORMATTER
Converts the raw classroom engagement JSON to a formatted summary

Input: output_engagement.json (student-centric with per-frame data)
Output: formatted_engagement.json (aggregated statistics)

Engagement Logic:
- ENGAGED: writing_notes + (neutral/happy/surprise)
          OR raising_hand + any emotion
          OR neutral action + (happy/neutral)
- DISENGAGED: using_mobile, sleeping, looking_away
             OR sad/angry/fear with any action
"""
import json
import os
from collections import Counter
import argparse

# ==================== CONFIGURATION ====================
INPUT_JSON_PATH = r"E:\FYP\videos\output_engagement.json"
OUTPUT_JSON_PATH = r"E:\FYP\videos\formatted_engagement.json"

# ==================== ENGAGEMENT LOGIC ====================
# Define which combinations of action + emotion = engaged/disengaged

# Actions that indicate engagement
ENGAGED_ACTIONS = {"writing_notes", "raising_hand"}

# Actions that indicate disengagement
DISENGAGED_ACTIONS = {"using_mobile", "sleeping", "looking_away"}

# Emotions that boost engagement
POSITIVE_EMOTIONS = {"happy", "neutral", "surprise"}

# Emotions that indicate disengagement
NEGATIVE_EMOTIONS = {"angry", "sad", "fear", "disgust"}


def is_engaged(emotion: str, action: str) -> bool:
    """
    Determine if a student is engaged based on emotion + action combination.
    
    Engagement Rules:
    1. Writing notes with positive emotion = ENGAGED
    2. Raising hand (any emotion) = ENGAGED
    3. Neutral action with positive emotion = ENGAGED
    4. Using mobile/sleeping/looking away = DISENGAGED
    5. Negative emotions override most actions = DISENGAGED
    
    Returns:
        True if engaged, False if disengaged
    """
    # Disengaged actions always mean disengagement
    if action in DISENGAGED_ACTIONS:
        return False
    
    # Negative emotions with non-engaged actions = disengaged
    if emotion in NEGATIVE_EMOTIONS and action not in ENGAGED_ACTIONS:
        return False
    
    # Raising hand is always engaged
    if action == "raising_hand":
        return True
    
    # Writing notes with non-negative emotions = engaged
    if action == "writing_notes" and emotion not in NEGATIVE_EMOTIONS:
        return True
    
    # Neutral action with positive emotions = engaged
    if action == "neutral" and emotion in POSITIVE_EMOTIONS:
        return True
    
    # Default to disengaged for unclear cases
    return False


def format_engagement_json(input_path: str, output_path: str) -> dict:
    """
    Process the raw engagement JSON and create formatted output.
    
    Args:
        input_path: Path to input JSON from classroom_engagement.py
        output_path: Path to save formatted JSON
        
    Returns:
        Formatted engagement dictionary
    """
    print("=" * 60)
    print("ENGAGEMENT JSON FORMATTER")
    print("=" * 60 + "\n")
    
    # Load input JSON
    print(f"📂 Loading: {input_path}")
    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    
    video_info = raw_data.get("video_info", {})
    students = raw_data.get("students", {})
    
    fps = video_info.get("fps", 30)
    total_frames = video_info.get("total_frames", 0)
    total_students = len(students)
    
    print(f"   Video: {total_frames} frames @ {fps} fps")
    print(f"   Students: {total_students}\n")
    
    # Initialize counters
    emotion_counts = Counter()  # How many students have each emotion (unique per student)
    action_counts = Counter()   # How many students have each action (unique per student)
    
    # Count unique students who are engaged/disengaged OVERALL
    students_engaged = 0
    students_disengaged = 0
    
    # Per-second engagement tracking
    seconds_data = {}  # {second: {"engaged": count, "total": count}}
    
    # Process each student
    print("📊 Processing students...")
    for student_id, student_data in students.items():
        frames = student_data.get("frames", {})
        
        if not frames:
            continue
        
        # Get unique emotions and actions for this student
        student_emotions = Counter([f["emotion"] for f in frames.values()])
        student_actions = Counter([f["action"] for f in frames.values()])
        
        # Count most common emotion/action for this student
        if student_emotions:
            top_emotion = student_emotions.most_common(1)[0][0]
            emotion_counts[top_emotion] += 1
        
        if student_actions:
            top_action = student_actions.most_common(1)[0][0]
            action_counts[top_action] += 1
        
        # Count engagement per frame for this student
        student_engaged_frames = 0
        student_total_frames = len(frames)
        
        # Process each frame for this student
        for frame_id_str, frame_data in frames.items():
            frame_id = int(frame_id_str)
            emotion = frame_data["emotion"]
            action = frame_data["action"]
            
            # Calculate which second this frame belongs to
            second = int(frame_id / fps) + 1
            
            if second not in seconds_data:
                seconds_data[second] = {"engaged": 0, "total": 0}
            
            seconds_data[second]["total"] += 1
            
            # Check engagement for this frame
            engaged = is_engaged(emotion, action)
            if engaged:
                student_engaged_frames += 1
                seconds_data[second]["engaged"] += 1
        
        # Determine if this STUDENT is overall engaged or disengaged
        # A student is "engaged" if they are engaged in majority of their frames (>50%)
        if student_total_frames > 0:
            engagement_ratio = student_engaged_frames / student_total_frames
            if engagement_ratio > 0.5:
                students_engaged += 1
            else:
                students_disengaged += 1
    
    # Calculate engagement over time (as percentage score)
    engagement_over_time = []
    for second in sorted(seconds_data.keys()):
        data = seconds_data[second]
        if data["total"] > 0:
            score = int((data["engaged"] / data["total"]) * 100)
        else:
            score = 0
        engagement_over_time.append({"second": second, "score": score})
    
    # Build formatted output
    formatted_output = {
        "emotions": dict(emotion_counts),
        "actions": dict(action_counts),
        "engagement_summary": {
            "engaged_count": students_engaged,
            "disengaged_count": students_disengaged
        },
        "engagement_over_time": engagement_over_time
    }
    
    # Save formatted JSON
    print(f"\n💾 Saving: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(formatted_output, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FORMATTED OUTPUT SUMMARY")
    print("=" * 60)
    
    print("\n📊 Emotions (students per emotion):")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"   {emotion}: {count}")
    
    print("\n🎬 Actions (students per action):")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"   {action}: {count}")
    
    print(f"\n✅ Engagement Summary (by student):")
    print(f"   Engaged students: {students_engaged}")
    print(f"   Disengaged students: {students_disengaged}")
    
    total_students_counted = students_engaged + students_disengaged
    if total_students_counted > 0:
        engagement_pct = (students_engaged / total_students_counted) * 100
        print(f"   Overall class engagement: {engagement_pct:.1f}%")
    
    print(f"\n📈 Engagement over time: {len(engagement_over_time)} seconds tracked")
    
    print("\n" + "=" * 60)
    print("✅ FORMATTING COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Output saved to: {output_path}")
    
    return formatted_output


def main():
    parser = argparse.ArgumentParser(description="Format engagement JSON")
    parser.add_argument("--input", "-i", default=INPUT_JSON_PATH, 
                        help="Input JSON path")
    parser.add_argument("--output", "-o", default=OUTPUT_JSON_PATH,
                        help="Output JSON path")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ ERROR: Input file not found: {args.input}")
        return
    
    format_engagement_json(args.input, args.output)


if __name__ == "__main__":
    main()
