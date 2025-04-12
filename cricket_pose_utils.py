import cv2
import numpy as np
import mediapipe as mp
import os
import glob

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            pose_2d = [(lm.x, lm.y) for lm in landmarks]
            keypoints.append(pose_2d)
            break  # Use only the first frame for comparison

    cap.release()
    return keypoints if keypoints else None

def load_ideal_pose_from_videos(scenario_name, role="batting", base_dir="training"):
    scenario_path = os.path.join(base_dir, role, scenario_name)
    all_videos = glob.glob(os.path.join(scenario_path, "*.mp4")) + \
                 glob.glob(os.path.join(scenario_path, "*.mov")) + \
                 glob.glob(os.path.join(scenario_path, "*.avi"))

    all_keypoints = []
    for vid in all_videos:
        kps = extract_keypoints_from_video(vid)
        if kps:
            all_keypoints.append(kps[0])

    if not all_keypoints:
        return None

    avg_pose = []
    for joints in zip(*all_keypoints):
        joint_x = np.mean([j[0] for j in joints])
        joint_y = np.mean([j[1] for j in joints])
        avg_pose.append((joint_x, joint_y))

    return avg_pose

def get_body_part_mapping():
    return {
        "elbow": [13, 14],
        "shoulder": [11, 12],
        "knee": [25, 26],
        "ankle": [27, 28],
        "foot": [31, 32],
        "hip": [23, 24],
        "back": [11, 12, 23, 24],
        "feet": [27, 28, 31, 32]
    }

def compare_pose(user_pose, ideal_pose):
    if not user_pose or not ideal_pose:
        return 999, [], {}

    diffs = []
    part_mapping = get_body_part_mapping()
    part_diffs = {part: [] for part in part_mapping}

    for idx, ((ux, uy), (ix, iy)) in enumerate(zip(user_pose, ideal_pose)):
        dist = np.linalg.norm(np.array([ux, uy]) - np.array([ix, iy]))
        diffs.append(dist)
        for part, indices in part_mapping.items():
            if idx in indices:
                part_diffs[part].append(dist)

    part_diff_summary = {part: np.mean(vals) for part, vals in part_diffs.items() if vals}
    mean_diff = np.mean(diffs)
    return mean_diff, diffs, part_diff_summary
