import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# === AYARLAR ===
MODEL_PATH = "models/cat_dog_model.h5"
IMG_SIZE = 128

# === MODELİ YÜKLE ===
@st.cache(allow_output_mutation=True)
def load_cnn_model():
    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# === BAŞLIK ===
st.title("🐶🐱 Cat vs Dog Sınıflandırıcı")
st.write("Bir görsel yükle, model tahmin etsin!")

# === GÖRSEL YÜKLE ===
uploaded_file = st.file_uploader("Bir resim seç (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Yüklenen Görsel", use_column_width=True)

    # === ÖN İŞLEME ===
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # === TAHMİN ===
    prediction = model.predict(img_array)[0][0]
    label = "🐶 Köpek" if prediction > 0.5 else "🐱 Kedi"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # === SONUÇ ===
    st.success(f"Tahmin: {label} ({confidence:.2%} güven)")
