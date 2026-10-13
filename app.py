# app.py
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" / "face_mask_detector.keras"
IMG_SIZE = (128, 128)
LABELS = ["Wearing a mask", "Not wearing a mask"]   # index 0, index 1

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_detector.keras")

model = load_model()

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)  # cv2.resize defaults to bilinear
    x = np.asarray(img, dtype=np.float32)
    x = x[..., ::-1]            # RGB -> BGR, to match cv2.imread in the notebook
    x = x / 255.0
    return np.reshape(x, [1, 128, 128, 3])

st.title("Face Mask Detector")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)

    prediction = model.predict(preprocess(img), verbose=0)[0]
    idx = int(np.argmax(prediction))

    st.subheader(LABELS[idx])
    st.write(f"Confidence: {prediction[idx]:.1%}")