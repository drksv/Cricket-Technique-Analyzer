import cv2
import mediapipe as mp
import urllib.request
import numpy as np

mp_pose = mp.solutions.pose

def video_from_url(url):
    tmp_video_path, _ = urllib.request.urlretrieve(url)
    cap = cv2.VideoCapture(tmp_video_path)
    return cap

def extract_pose_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False)
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            landmarks_list.append(landmarks)

    cap.release()
    return landmarks_list

def compare_poses(user_landmarks, ideal_landmarks):
    issues = []
    total_score = 0
    frame_count = min(len(user_landmarks), len(ideal_landmarks))

    for i in range(frame_count):
        user = user_landmarks[i]
        ideal = ideal_landmarks[i]

        deviations = np.linalg.norm(np.array(user) - np.array(ideal), axis=1)
        avg_deviation = np.mean(deviations)
        total_score += (100 - avg_deviation * 1000)  # crude scaling

        # Check key parts: elbow (13,14), feet (27,28), back (11,12)
        keypoints = {"Left Elbow": 13, "Right Elbow": 14, "Left Foot": 27, "Right Foot": 28, "Left Shoulder": 11, "Right Shoulder": 12}
        for part, idx in keypoints.items():
            if deviations[idx] > 0.05:
                issues.append(f"{part} off position at frame {i}")

    final_score = max(0, min(100, total_score / frame_count))
    return final_score, list(set(issues))

def analyze_video_vs_ideal(user_video_path, ideal_video_url):
    ideal_cap = video_from_url(ideal_video_url)

    # Save ideal video temporarily
    ideal_temp_path = "temp_ideal_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while True:
        ret, frame = ideal_cap.read()
        if not ret:
            break
        if out is None:
            h, w, _ = frame.shape
            out = cv2.VideoWriter(ideal_temp_path, fourcc, 20.0, (w, h))
        out.write(frame)
    ideal_cap.release()
    if out:
        out.release()

    user_landmarks = extract_pose_landmarks(user_video_path)
    ideal_landmarks = extract_pose_landmarks(ideal_temp_path)

    score, issues = compare_poses(user_landmarks, ideal_landmarks)

    return {
        "score": round(score, 2),
        "issues": issues
    }
