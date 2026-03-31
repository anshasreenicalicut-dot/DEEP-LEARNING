import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Object Detection App")
st.write("Upload an image for detection")

# load trained model
model = YOLO("C:\\Users\\Ansha TV\\Desktop\\Data Science\\Deep Learning\\deep learning\\runs\\detect\\train32\\weights\\best.pt")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Objects"):
        results = model(image)
        result_image = results[0].plot()

        st.image(result_image, caption="Detection Result", use_container_width=True)