import os
import tempfile
import streamlit as st

import torch

from utils.inference import (
    load_model,
    infer_video,
)
from utils.visualize import draw_predictions


# --------------------------------------------------
# App Config
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid Deepfake Detection",
    layout="centered",
)

st.title("🎭 Hybrid Deepfake Detection System")
st.write(
    "Upload a video to detect whether it is **REAL** or **FAKE** "
    "using a hybrid Xception + BiLSTM + Attention model."
)

# --------------------------------------------------
# Settings
# --------------------------------------------------

CHECKPOINT_PATH = "checkpoints/best_model.pth"
SEQ_LEN = 16
FAKE_THRESHOLD = 0.35  # update from threshold_search.py

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# --------------------------------------------------
# Load model (cached)
# --------------------------------------------------

@st.cache_resource
def load_cached_model():
    return load_model(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        seq_len=SEQ_LEN,
    )


model = load_cached_model()

st.success(f"Model loaded successfully on **{DEVICE.upper()}**")

# --------------------------------------------------
# File uploader
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"],
)

# --------------------------------------------------
# Inference
# --------------------------------------------------

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    with st.spinner("Analyzing video..."):
        label, fake_prob = infer_video(
            video_path=video_path,
            model=model,
            device=DEVICE,
            threshold=FAKE_THRESHOLD,
        )

    os.remove(video_path)

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    if label == 1:
        st.error("❌ FAKE VIDEO DETECTED")
    else:
        st.success("✅ REAL VIDEO")

    st.metric(
        label="Fake Probability",
        value=f"{fake_prob:.2f}",
    )

    st.progress(int(fake_prob * 100))

    st.info(
        f"Decision Threshold: **{FAKE_THRESHOLD}**\n\n"
        f"Prediction: **{'FAKE' if label else 'REAL'}**"
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")
st.caption(
    "Hybrid Deepfake Detection | Xception + BiLSTM + Attention | "
    "M.Tech CSE Project"
)
