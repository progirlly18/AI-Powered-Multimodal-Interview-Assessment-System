import streamlit as st
import sys
from pathlib import Path

# ------------------------------------
# Backend Import
# ------------------------------------

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.services.interview_service import InterviewService

# ------------------------------------
# Page Config
# ------------------------------------

st.set_page_config(
    page_title="AI Interview Assessment",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 AI Interview Assessment System")
st.caption(
    "AI-powered interview evaluation using Computer Vision and Speech Analysis"
)

st.divider()

# ------------------------------------
# Candidate Details
# ------------------------------------

left, right = st.columns([1, 1])

with left:

    st.subheader("👤 Candidate Details")

    name = st.text_input(
        "Candidate Name",
        placeholder="Enter candidate name"
    )

    role = st.selectbox(
        "Job Role",
        [
            "Software Engineer",
            "AI Engineer",
            "Machine Learning Engineer",
            "Data Analyst",
            "HR Interview"
        ]
    )

    duration = st.slider(
        "Interview Duration (seconds)",
        min_value=10,
        max_value=60,
        value=20
    )

    start = st.button(
        "▶ Start Interview",
        use_container_width=True
    )

with right:

    st.subheader("📡 Interview Status")

    status = st.empty()
    camera = st.empty()
    microphone = st.empty()

    status.success("🟢 Ready")
    camera.info("📷 Waiting")
    microphone.info("🎤 Waiting")

st.divider()

# ------------------------------------
# Result Placeholders
# ------------------------------------

st.subheader("📊 Interview Results")

c1, c2, c3, c4 = st.columns(4)

emotion_box = c1.empty()
eye_box = c2.empty()
speech_box = c3.empty()
overall_box = c4.empty()

emotion_box.metric("😊 Emotion", "--")
eye_box.metric("👀 Eye Contact", "--")
speech_box.metric("🎤 Speech", "--")
overall_box.metric("⭐ Overall", "--")

recommendations_box = st.empty()
transcript_box = st.empty()
stats_box = st.empty()
# ------------------------------------
# Start Interview
# ------------------------------------

if start:

    if not name.strip():
        st.error("Please enter the candidate name.")
        st.stop()

    status.warning("🟡 Interview Running...")
    camera.success("📷 Camera Active")
    microphone.success("🎤 Recording Audio")

    st.write("Creating service...")

    service = InterviewService()

    st.write("Calling backend...")

    result = service.run(duration)

    st.write("Backend finished!")

    status.success("✅ Interview Completed")
    camera.info("📷 Camera Stopped")
    microphone.info("🎤 Recording Complete")

    # -------------------------------
    # Metrics
    # -------------------------------

    emotion_box.metric(
        "😊 Emotion",
        result["emotion"]
    )

    eye_box.metric(
        "👀 Eye Contact",
        result["eye_contact"]
    )

    speech_box.metric(
        "🎤 Speech Score",
        f'{result["speech_score"]}/100'
    )

    overall_box.metric(
        "⭐ Overall Score",
        f'{result["overall_score"]}/100'
    )

    # -------------------------------
    # Recommendations
    # -------------------------------

    recommendations_box.success(
        "### 📋 AI Recommendations\n\n"
        + "\n".join(
            f"• {item}"
            for item in result["recommendations"]
        )
    )

    # -------------------------------
    # Transcript
    # -------------------------------

    transcript_box.markdown("## 📝 Transcript")
    transcript_box.write(result["transcript"])

    # -------------------------------
    # Additional Statistics
    # -------------------------------

    with stats_box.container():

        st.markdown("## 📈 Speech Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🎤 WPM",
            result["wpm"]
        )

        col2.metric(
            "💬 Fillers",
            result["fillers"]
        )

        col3.metric(
            "🌧 Background",
            result["background"]
        )

    st.balloons()