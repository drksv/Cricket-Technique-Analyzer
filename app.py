import streamlit as st
import os
import json
from cricket_pose_utils import (
    extract_keypoints_from_video,
    compare_pose,
    load_ideal_pose_from_videos
)

# Load scenario mapping
with open("utils/scenario_mapping.json") as f:
    SCENARIOS = json.load(f)

st.set_page_config(page_title="Cricket Technique Analyzer", layout="centered")
st.title("🏏 AI Cricket Technique Analyzer")

st.subheader("📊 Analyze Your Technique")

role = st.selectbox("Choose Role", ["batting", "bowling"])
scenario = st.selectbox("Select Scenario", SCENARIOS[role])

video = st.file_uploader("🎥 Upload Your Cricket Video", type=["mp4", "mov", "avi"])

if video and st.button("Analyze Technique"):
    with open("user_video.mov", "wb") as out:
        out.write(video.read())

    st.video(video)
    st.info("🔎 Extracting pose...")

    user_pose_list = extract_keypoints_from_video("user_video.mov")
    ideal_pose = load_ideal_pose_from_videos(scenario_name=scenario, role=role)

    if not user_pose_list:
        st.error("❌ No pose detected in your video. Try better lighting or positioning.")
    elif not ideal_pose:
        st.error("❌ No training videos found for this scenario.")
    else:
        user_pose = user_pose_list[0]
        score, _, part_diff_summary = compare_pose(user_pose, ideal_pose)

        st.metric("Pose Similarity Score", f"{round(100 - score * 100, 2)} / 100")

        st.markdown("### 🧠 Body Part Feedback:")
        flagged = False
        for part, diff in part_diff_summary.items():
            if diff > 0.04:
                st.warning(f"⚠️ Adjust your **{part}** – noticeable deviation from ideal.")
                flagged = True

        if not flagged:
            st.success("✅ Great job! Your posture matches well.")
