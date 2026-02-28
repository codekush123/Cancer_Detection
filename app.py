from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


st.set_page_config(page_title="Skin Lesion Classifier", page_icon=":microscope:", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_SIZE = (256, 256)


def discover_models() -> list[Path]:
    if not MODELS_DIR.exists():
        return []

    files = sorted(
        [*MODELS_DIR.glob("*.h5"), *MODELS_DIR.glob("*.keras")],
        key=lambda p: p.name.lower(),
    )
    return files


@st.cache_resource
def load_model_from_path(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def choose_preprocess(model_name: str, image_array: np.ndarray) -> np.ndarray:
    name = model_name.lower()

    if "efficientnet" in name:
        return preprocess_input(image_array)

    return image_array / 255.0


def preprocess_image(image: Image.Image, model_name: str) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return choose_preprocess(model_name, arr)


def predict(model: tf.keras.Model, image: Image.Image, model_name: str) -> tuple[str, float, float, float]:
    x = preprocess_image(image, model_name)
    pred = model.predict(x, verbose=0)

    if pred.ndim == 2 and pred.shape[1] == 1:
        malignant_prob = float(pred[0][0])
    elif pred.ndim == 2 and pred.shape[1] >= 2:
        probs = tf.nn.softmax(pred[0]).numpy()
        malignant_prob = float(probs[1])
    else:
        raise ValueError(f"Unexpected model output shape: {pred.shape}")

    benign_prob = 1.0 - malignant_prob
    label = "Malignant" if malignant_prob >= 0.5 else "Benign"
    confidence = malignant_prob if label == "Malignant" else benign_prob
    return label, confidence, benign_prob, malignant_prob


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

    .stApp {
        font-family: 'Manrope', sans-serif;
        background:
            radial-gradient(circle at 10% 15%, rgba(212, 175, 55, 0.12), transparent 34%),
            radial-gradient(circle at 85% 10%, rgba(120, 119, 198, 0.10), transparent 32%),
            linear-gradient(135deg, #0b1220 0%, #111827 52%, #1f2937 100%);
        color: #e5e7eb;
    }
    .hero {
        padding: 1.1rem 1.25rem;
        border-radius: 16px;
        border: 1px solid rgba(212, 175, 55, 0.25);
        background: linear-gradient(120deg, rgba(17, 24, 39, 0.84), rgba(30, 41, 59, 0.84));
        backdrop-filter: blur(6px);
        margin-bottom: 1rem;
        box-shadow: 0 12px 35px rgba(2, 6, 23, 0.35);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        color: #f8fafc;
        letter-spacing: 0.2px;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        color: #cbd5e1;
    }

    /* Text and widget colors tuned for dark premium background */
    .stMarkdown, .stText, label, .stSelectbox div, .stFileUploader label {
        color: #e5e7eb !important;
    }
    .stInfo {
        background: rgba(30, 41, 59, 0.75) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #d4af37, #b38728) !important;
        color: #111827 !important;
        border: 0 !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover {
        filter: brightness(1.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Skin Cancer Detection</h1>
      <p>Upload a dermoscopic image, select your trained model, and get a clear prediction with confidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

available_models = discover_models()

if not available_models:
    st.error("No model files found in the models folder. Add .h5 or .keras files under /models.")
    st.stop()

left, right = st.columns([1, 1.2], gap="large")

with left:
    selected_model_path = st.selectbox(
        "Choose trained model",
        options=available_models,
        format_func=lambda p: p.name,
        help="All .h5 and .keras files in the models folder are listed.",
    )

    st.write(f"Using model: `{selected_model_path.name}`")

    uploaded_file = st.file_uploader(
        "Upload lesion image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    run_prediction = st.button("Run Prediction", type="primary", use_container_width=True)

with right:
    if uploaded_file is None:
        st.info("Upload an image to preview and predict.")
        st.stop()

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if run_prediction:
        try:
            model = load_model_from_path(str(selected_model_path.resolve()))
            label, confidence, benign_prob, malignant_prob = predict(model, image, selected_model_path.name)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        if label == "Malignant":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{confidence * 100:.2f}%")
        m2.metric("Benign Prob.", f"{benign_prob * 100:.2f}%")
        m3.metric("Malignant Prob.", f"{malignant_prob * 100:.2f}%")

        st.progress(float(max(0.0, min(1.0, malignant_prob))), text="Malignant probability")
        st.caption("For educational/demo use only. Not a medical diagnosis.")
    else:
        st.info("Click Run Prediction to get results.")
