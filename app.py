# basic_fruit_classifier_app.py
# Simplified and lightweight fruit classifier web interface using Streamlit

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os


# Load the model and labels
# Updated MODEL_PATH and CLASS_INDEX_PATH using os.path.join for better path handling
MODEL_PATH = os.path.join(os.path.expanduser('~'), 'Downloads', 'c++', 'Desktop Application Development', 'the proper app', 'ML Project', 'saved_model', 'fruit_quality_cnn_model.h5')
CLASS_INDEX_PATH = os.path.join(os.path.expanduser('~'), 'Downloads', 'c++', 'Desktop Application Development', 'the proper app', 'ML Project', 'saved_model', 'class_indices.json')

# App title and instructions
st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("🍎 Fruit Image Classifier")
st.markdown("Upload a fruit image to identify its category.")

# Validate existence of model files
if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDEX_PATH):
    st.error("Model or class label file not found in 'saved_model/' directory.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDEX_PATH, 'r') as f:
    class_indices = json.load(f)
labels = list(class_indices.keys())

# File uploader
uploaded_file = st.file_uploader("Upload a fruit image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_data = Image.open(uploaded_file).convert('RGB')
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    resized = image_data.resize((150, 150))
    array = np.expand_dims(np.array(resized) / 255.0, axis=0)

    # Prediction
    preds = model.predict(array)[0]
    class_id = np.argmax(preds)
    label = labels[class_id]
    confidence = preds[class_id] * 100

    st.subheader("Prediction")
    st.success(f"Detected: **{label}** with {confidence:.2f}% confidence")


