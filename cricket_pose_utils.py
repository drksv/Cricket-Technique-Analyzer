import cv2
import mediapipe as mp
import urllib.request
import numpy as np
import os

mp_pose = mp.solutions.pose

# Joint index to human-readable mapping (based on MediaPipe Pose)
JOINT_NAMES = {
    0: 'Nose', 11: 'Left Shoulder', 12: 'Right Shoulder',
    13: 'Left Elbow', 14: 'Right Elbow', 15: 'Left Wrist', 16: 'Right Wrist',
    23: 'Left Hip', 24: 'Right Hip', 25: 'Left Knee', 26: 'Right Knee',
    27: 'Left Ankle', 28: 'Right Ankle'
}

def extract_pose_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    landmarks_list = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        frame_count += 1

        if result.pose_landmarks:
            landmarks = [(lm.x, lm.y) for lm in result.pose_landmarks.landmark]
            landmarks_list.append(landmarks)

    cap.release()
    pose.close()
    print(f"Processed {frame_count} frames, detected poses in {len(landmarks_list)} frames.")
    return landmarks_list

def compare_poses(user_landmarks, ideal_landmarks):
    total_score = 0
    frame_count = 0
    issues = []

    for user_frame, ideal_frame in zip(user_landmarks, ideal_landmarks):
        frame_score = 0
        frame_count += 1
        frame_issues = []

        for idx, (user_point, ideal_point) in enumerate(zip(user_frame, ideal_frame)):
            distance = np.linalg.norm(np.array(user_point) - np.array(ideal_point))
            frame_score += max(0, 10 - distance * 10)

            if distance > 0.1:  # threshold to flag
                joint_name = JOINT_NAMES.get(idx, f'Joint {idx}')
                frame_issues.append({
                    'joint': joint_name,
                    'issue': f'Off by {distance:.2f}'
                })

        total_score += frame_score
        if frame_issues:
            issues.append({
                'frame': frame_count,
                'problems': frame_issues
            })

    if frame_count == 0:
        return 0, [{'error': 'No valid frames processed. Check video quality or ensure body is visible.'}]

    final_score = max(0, min(100, total_score / frame_count))
    return final_score, issues

def analyze_video_vs_ideal(user_video_path, ideal_video_url):
    # Download ideal video temporarily
    ideal_video_path = "temp_ideal_video.mp4"
    urllib.request.urlretrieve(ideal_video_url, ideal_video_path)

    print("Extracting landmarks from user video...")
    user_landmarks = extract_pose_landmarks(user_video_path)
    print("Extracting landmarks from ideal video...")
    ideal_landmarks = extract_pose_landmarks(ideal_video_path)

    if not user_landmarks:
        return 0, [{'error': 'No landmarks detected in user video.'}]
    if not ideal_landmarks:
        return 0, [{'error': 'No landmarks detected in ideal video.'}]

    score, issues = compare_poses(user_landmarks, ideal_landmarks)

    os.remove(ideal_video_path)
    return score, issues
