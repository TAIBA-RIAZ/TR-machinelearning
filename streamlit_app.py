import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("wheat_leaf_disease_destection_model.keras")
    return model

model = load_model()

# Preprocess function
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# UI Design
st.title("🌾 Wheat Leaf Disease Detection")
st.write("Upload an image of a wheat leaf to detect its disease.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    
    class_names = ["Crown and Root Rot", "Healthy Wheat", "Leaf Rust", "Wheat Loose Smut"]
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### Prediction: **{predicted_class}**")
