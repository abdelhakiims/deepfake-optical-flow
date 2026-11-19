"""Web interface: upload a video, get a verdict.

The models are loaded once and the whole analysis runs in this process, with the
same preprocessing, weights, flow scale and threshold as in training.

Run from the repository root:
    streamlit run app.py -- --checkpoint runs/ffpp_two_stream_s42/best.pth
"""

import argparse
import os
import tempfile

import streamlit as st

from deepfake_flow.predict import DeepfakeDetector

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="best.pth")
parser.add_argument("--face_model", default=None)
parser.add_argument("--raft_dir", default=None)
parser.add_argument("--raft_weights", default=None)
cli, _ = parser.parse_known_args()


@st.cache_resource(show_spinner="Loading the models...")
def get_detector(checkpoint):
    return DeepfakeDetector(checkpoint, cli.face_model, cli.raft_dir, cli.raft_weights)


st.set_page_config(page_title="Deepfake detector")
st.title("Deepfake video detector")
st.write("Upload a video showing one person's face. The model analyses up to six short "
         "clips of the main face and averages their scores.")

if not os.path.isfile(cli.checkpoint):
    st.error(f"Model not found: {cli.checkpoint}")
    st.stop()

uploaded = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is not None and st.button("Analyse"):
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name
    try:
        with st.spinner("Analysing..."):
            result = get_detector(cli.checkpoint).predict(video_path)
        message = (f"probability of fake {result['p_fake']:.1%}, "
                   f"decision threshold {result['threshold']:.1%}")
        if result["verdict"] == "FAKE":
            st.error(f"Verdict: FAKE ({message})")
        else:
            st.success(f"Verdict: REAL ({message})")
        st.caption(f"{result['clips']} clips analysed. Probability per clip: "
                   f"{', '.join(f'{p:.0%}' for p in result['clip_p_fake'])}.")
        if result["people_seen"] > 1:
            st.caption("Several faces were visible; the main (largest frontal) face was analysed.")
    except Exception as exc:
        st.error(f"The analysis failed: {exc}")
    finally:
        os.remove(video_path)
