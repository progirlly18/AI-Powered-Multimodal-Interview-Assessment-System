import streamlit as st
import time
import sys
from pathlib import Path

# -----------------------------
# Import backend
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from backend.services.interview_service import InterviewService

# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="AI Interview Assessment",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 AI Interview Assessment System")
st.caption("AI-powered interview evaluation using Computer Vision and Speech Analysis")

st.divider()

# -----------------------------
# Layout
# -----------------------------
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
        10,
        60,
        20
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

    progress = st.progress(0)

    status.success("🟢 Ready")

    camera.info("📷 Camera : Waiting")

    microphone.info("🎤 Microphone : Waiting")

st.divider()

st.subheader("📊 Interview Results")

c1, c2, c3, c4 = st.columns(4)

emotion = c1.empty()
eye = c2.empty()
speech = c3.empty()
overall = c4.empty()

emotion.metric("😊 Emotion", "--")
eye.metric("👀 Eye Contact", "--")
speech.metric("🎤 Speech", "--")
overall.metric("⭐ Overall", "--")

recommendations = st.empty()

# -----------------------------
# Start Interview
# -----------------------------

if start:

    status.warning("🟡 Interview Running")

    camera.success("📷 Camera Active")

    microphone.success("🎤 Recording Audio")

    for i in range(duration):

        progress.progress((i + 1) / duration)

        status.info(
            f"⏳ Time Remaining : {duration-i-1} sec"
        )

        time.sleep(1)

    status.info("🧠 Running AI Analysis...")

    # -------------------------
    # Backend
    # -------------------------

    service = InterviewService()

    result = service.run(duration)

    status.success("✅ Interview Completed")

    camera.info("📷 Camera Stopped")

    microphone.info("🎤 Recording Saved")

    emotion.metric(
        "😊 Emotion",
        result["emotion"]
    )

    eye.metric(
        "👀 Eye Contact",
        f'{result["eye_contact"]}%'
    )

    speech.metric(
        "🎤 Speech",
        f'{result["speech_score"]}/100'
    )

    overall.metric(
        "⭐ Overall",
        f'{result["overall_score"]}/100'
    )

    recommendations.success(
        "### 📋 AI Recommendations\n\n"
        + "\n".join(
            [f"• {x}" for x in result["recommendations"]]
        )
    )