import streamlit as st
import json
from cricket_pose_utils import analyze_video_vs_ideal

# Load scenario mapping
with open('utils/scenario_mapping.json') as f:
    scenario_mapping = json.load(f)

st.title("🏏 Health Timeout Technique Analyzer")

# Select activity type first
selected_category = st.selectbox("Choose activity type:", list(scenario_mapping.keys()))

# Then select scenario under that category
selected_scenario = st.selectbox(f"Select {selected_category} scenario:", list(scenario_mapping[selected_category].keys()))
ideal_video_url = scenario_mapping[selected_category][selected_scenario]

# Upload user video
user_video = st.file_uploader("Upload your cricket video (mp4)", type=['mp4'])

if st.button("Analyze") and user_video:
    with open("temp_user_video.mp4", "wb") as f:
        f.write(user_video.read())

    st.info("Analyzing... please wait ⏳")
    result = analyze_video_vs_ideal("temp_user_video.mp4", ideal_video_url)

    st.success(f"Pose Similarity Score: {result['score']} / 100")
    st.write("**Body Parts Needing Improvement:**")
    if result['issues']:
        for issue in result['issues']:
            st.write(f"- {issue}")
    else:
        st.write("✅ No major issues detected!")
