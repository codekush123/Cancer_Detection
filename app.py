from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


st.set_page_config(page_title="Skin Cancer Detector", page_icon=":test_tube:", layout="centered")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "efficientnet_best.h5",
    PROJECT_ROOT / "models" / "best_model.h5",
    PROJECT_ROOT / "efficientnet_best.h5",
    PROJECT_ROOT / "best_model.h5",
]

IMAGE_SIZE = (224, 224)


@st.cache_resource
def load_trained_model() -> tf.keras.Model:
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            return tf.keras.models.load_model(model_path)

    searched = "\n".join(f"- {p}" for p in MODEL_CANDIDATES)
    raise FileNotFoundError(
        "No trained model file found. Place your model in one of these locations:\n"
        f"{searched}"
    )


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


def predict(model: tf.keras.Model, image: Image.Image) -> tuple[str, float]:
    x = preprocess_image(image)
    pred = model.predict(x, verbose=0)

    if pred.ndim == 2 and pred.shape[1] == 1:
        malignant_prob = float(pred[0][0])
    elif pred.ndim == 2 and pred.shape[1] >= 2:
        probs = tf.nn.softmax(pred[0]).numpy()
        malignant_prob = float(probs[1])
    else:
        raise ValueError(f"Unexpected model output shape: {pred.shape}")

    label = "Malignant" if malignant_prob >= 0.5 else "Benign"
    confidence = malignant_prob if label == "Malignant" else 1.0 - malignant_prob
    return label, confidence


st.title("Skin Cancer Detection App")
st.write("Upload a dermoscopic image and the trained model will predict the lesion type.")

try:
    model = load_trained_model()
    st.success("Model loaded successfully.")
except Exception as exc:
    st.error(str(exc))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict Cancer Type", type="primary"):
        with st.spinner("Running prediction..."):
            label, confidence = predict(model, image)

        st.subheader(f"Prediction: {label}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.caption("Classes: Benign (non-cancerous), Malignant (cancerous)")
else:
    st.info("Please upload an image to get a prediction.")
