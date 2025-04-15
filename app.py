import streamlit as st
import json
from cricket_pose_utils import analyze_video_vs_ideal

# Load scenario mapping
with open('utils/scenario_mapping.json') as f:
    scenario_mapping = json.load(f)

st.title("🏏 Health Timeout Cricket Technique Analyzer")

st.sidebar.header("Select Scenario")
scenario_type = st.sidebar.selectbox("Type", list(scenario_mapping.keys()))
scenario_option = st.sidebar.selectbox("Scenario", scenario_mapping[scenario_type])

# Load ideal video URL from secrets
ideal_video_url = st.secrets["video_links"][scenario_option]

# Upload user video
uploaded_video = st.file_uploader("Upload your video (MP4)", type=["mp4"])

if uploaded_video is not None:
    with open("temp_user_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    st.success("Video uploaded. Running analysis...")

    result = analyze_video_vs_ideal("temp_user_video.mp4", ideal_video_url)

    st.write("### 📊 Result")
    st.write(f"**Score:** {result['score']}%")

    st.write("### ⚠️ Areas to Improve")
    for issue in result["issues"]:
        st.write(f"- {issue}")


