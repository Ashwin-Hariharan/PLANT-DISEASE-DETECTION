import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
MODEL_PATH = r'C:\Users\ashwi\GUVI_Projects\PlantDisease\EfficientNetB0_best.keras'
model = load_model(MODEL_PATH)

# Load class labels
class_names = ['Apple Scab', 'Apple Black Rot', 'Apple Rust', 'Healthy Apple',
               'Blueberry Healthy', 'Cherry Powdery Mildew', 'Cherry Healthy',
               'Corn Gray Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight',
               'Corn Healthy', 'Grape Black Rot', 'Grape Esca', 'Grape Leaf Blight',
               'Grape Healthy', 'Orange Huanglongbing', 'Peach Bacterial Spot',
               'Peach Healthy', 'Pepper Bacterial Spot', 'Pepper Healthy',
               'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
               'Raspberry Healthy', 'Soybean Healthy', 'Squash Powdery Mildew',
               'Strawberry Leaf Scorch', 'Strawberry Healthy', 'Tomato Bacterial Spot',
               'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold',
               'Tomato Septoria Leaf Spot', 'Tomato Spider Mites', 'Tomato Target Spot',
               'Tomato Mosaic Virus', 'Tomato Yellow Leaf Curl Virus', 'Tomato Healthy']

image_size = (96, 96)  # Ensure this matches the model's expected input size

# Streamlit App UI
st.title(" Plant Disease Detection App")
st.write("Upload an image of a plant leaf, and the model will predict the disease!")

# Upload image
uploaded_file = st.file_uploader(" Upload an Image ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to RGB format (3 channels)
    image = Image.open(uploaded_file).convert("RGB")

    # Resize the image to match model input size
    image = image.resize(image_size)

    # Convert to NumPy array and preprocess
    plant_image = np.array(image)  # Convert PIL image to NumPy array
    plant_image = img_to_array(plant_image)  # Convert to Tensor
    plant_image = np.expand_dims(plant_image, axis=0)  # Add batch dimension
    plant_image /= 255.0  # Normalize pixel values

    # Model Prediction
    predictions = model.predict(plant_image)
    predicted_disease = class_names[np.argmax(predictions)]  # Get highest probability class

    # Display Image & Prediction
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"###  Predicted Disease: **{predicted_disease}**")
