import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model_path = r"C:\Users\HP\Documents\breast cancer\model.h5"
model = tf.keras.models.load_model(model_path)

# Correct category order
categories = ["Normal", "Benign", "Malignant"]  

st.title("Breast Cancer Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an ultrasound image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Load as GRAYSCALE

    # Resize image to match model input size
    img_size = 128  # Ensure this matches the model's expected input size
    image = cv2.resize(image, (img_size, img_size))

    # Normalize image
    image = image.astype(np.float32) / 255.0

    # Reshape to match model input shape
    image = image.reshape(1, img_size, img_size, 1)

    # Debugging: Print processed image shape
    st.write(f"✅ Processed image shape: {image.shape}")

    # Predict
    prediction = model.predict(image)
    class_index = np.argmax(prediction)

    # Display result
    st.success(f"### Prediction: {categories[class_index]}")


