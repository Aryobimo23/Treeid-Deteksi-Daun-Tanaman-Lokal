import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import io

# Load model
model = joblib.load('model/treeid_model.pkl')

# Ekstraksi fitur (sama dengan train_model.py)
def extract_features_from_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (100, 100))
    features = image.flatten()
    return features

# Tampilan aplikasi
st.title("TreeID 🌿 - Identifikasi Daun Pohon Lokal")

uploaded_file = st.file_uploader("Unggah gambar daun", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Ekstraksi dan klasifikasi
    features = extract_features_from_image(image).reshape(1, -1)
    prediction = model.predict(features)

    st.success(f"🌳 Jenis pohon terdeteksi: **{prediction[0]}**")
