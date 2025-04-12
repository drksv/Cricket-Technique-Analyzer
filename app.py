import streamlit as st
import json
from cricket_pose_utils import analyze_video_vs_ideal

# Load scenario mapping
with open('utils/scenario_mapping.json') as f:
    scenario_mapping = json.load(f)

st.title("🏏 Health Timeout Cricket Technique Analyzer")

scenario = st.selectbox("Select Scenario:", options=list(scenario_mapping.keys()))
option = st.selectbox("Select Option:", options=scenario_mapping[scenario])

uploaded_video = st.file_uploader("Upload your video (MP4 format)", type=["mp4"])

if st.button("Analyze") and uploaded_video is not None:
    with open("temp_user_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    ideal_video_url = scenario_mapping[scenario][option]

    st.info("Processing video — please wait, this may take a moment...")

    score, issues = analyze_video_vs_ideal("temp_user_video.mp4", ideal_video_url)

    st.subheader(f"Technique Score: {score:.2f}/100")

    if issues:
        for issue in issues:
            if 'error' in issue:
                st.error(issue['error'])
            else:
                st.write(f"📸 Frame {issue['frame']}:")
                for problem in issue['problems']:
                    st.write(f"🔴 {problem['joint']}: {problem['issue']}")

    else:
        st.success("No issues detected — great job!")

    st.info("Done analyzing.")
