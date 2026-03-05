import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model('image_classification_model.h5', compile=False)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("🚢 Ship Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1,128,128,3)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    st.success(f"Predicted Ship Type: {predicted_label}")
    st.info(f"Confidence: {confidence:.2f}%")