import os
import tempfile
import streamlit as st
import torch
import pandas as pd

from utils.inference import load_model, infer_video
from database import init_db, insert_result, fetch_all_results, delete_all_results


# --------------------------------------------------
# Enterprise UI Styling
# --------------------------------------------------

st.set_page_config(page_title="Hybrid Deepfake Detection", layout="wide")

st.markdown("""
<style>
.big-title { font-size:30px !important; font-weight:700; }
.section-title { font-size:20px !important; font-weight:600; margin-top:20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🎭 Hybrid Deepfake Detection System</div>', unsafe_allow_html=True)

st.write(
    "Enterprise-ready AI system for detecting manipulated videos "
    "using Xception + BiLSTM + Attention architecture."
)

# --------------------------------------------------
# Initialize DB
# --------------------------------------------------

init_db()

# --------------------------------------------------
# Settings
# --------------------------------------------------

CHECKPOINT_PATH = "checkpoints/best_model.pth"
SEQ_LEN = 16
FAKE_THRESHOLD = 0.74

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_cached_model():
    return load_model(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        seq_len=SEQ_LEN,
    )

model = load_cached_model()

st.success(f"Model loaded on **{DEVICE.upper()}**")

# --------------------------------------------------
# Layout Columns
# --------------------------------------------------

col1, col2 = st.columns([2, 1])

# --------------------------------------------------
# Upload + Prediction Section
# --------------------------------------------------

with col1:
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

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

        prediction_text = "FAKE" if label == 1 else "REAL"

        # Save to DB
        insert_result(
            filename=uploaded_file.name,
            prediction=prediction_text,
            probability=float(fake_prob),
            threshold=FAKE_THRESHOLD
        )

        # Display result
        if prediction_text == "FAKE":
            st.error("❌ FAKE VIDEO DETECTED")
        else:
            st.success("✅ REAL VIDEO")

        st.metric("Fake Probability", f"{fake_prob:.2f}")
        st.progress(int(fake_prob * 100))

        st.info(
            f"Decision Threshold: **{FAKE_THRESHOLD}**\n\n"
            f"Prediction: **{prediction_text}**"
        )

# --------------------------------------------------
# Controls Section
# --------------------------------------------------

with col2:
    st.markdown('<div class="section-title">⚙ Controls</div>', unsafe_allow_html=True)

    if st.button("🗑 Delete History"):
        delete_all_results()
        st.success("Prediction history cleared.")

# --------------------------------------------------
# Prediction History Table
# --------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-title">📜 Prediction History</div>', unsafe_allow_html=True)

rows = fetch_all_results()

if rows:
    df = pd.DataFrame(rows, columns=[
        "ID", "Filename", "Prediction",
        "Probability", "Threshold", "Timestamp"
    ])

    st.dataframe(df.drop(columns=["ID"]), use_container_width=True)

    # CSV Download
    csv = df.drop(columns=["ID"]).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions yet.")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")
st.caption(
    "Hybrid Deepfake Detection | Xception + BiLSTM + Attention | "
    "Enterprise AI System | M.Tech CSE Project"
)
