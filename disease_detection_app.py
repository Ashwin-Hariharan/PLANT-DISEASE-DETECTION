import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image

# Load the trained model
MODEL_PATH = r'C:\Users\ashwi\GUVI_Projects\PlantDisease\SavedModels\ResNet50.h5'
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

image_size = (128, 128)  # ResNet50 expected input size

# Streamlit App UI
st.title("Plant Disease Detection App")
st.write("Upload an image of a plant leaf, and the model will predict the disease!")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to RGB format (3 channels)
    image = Image.open(uploaded_file).convert("RGB")

    # Resize the image to match model input size
    image = image.resize(image_size)

    # Convert to NumPy array and preprocess
    plant_image = img_to_array(image)  # Convert to Tensor
    plant_image = np.expand_dims(plant_image, axis=0)  # Add batch dimension
    plant_image = preprocess_input(plant_image)  # Apply ResNet preprocessing

    # Model Prediction
    predictions = model.predict(plant_image)[0]  # Get first batch result
    predicted_class_index = np.argmax(predictions)
    predicted_disease = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100  # Convert to percentage

    # Display Image & Prediction
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write(f"### Predicted Disease: **{predicted_disease}** ({confidence:.2f}% confidence)")

    # ========== Data Analysis Graphs ========== #

    # 1️⃣ **Prediction Probability Bar Chart** 📊
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=class_names, y=predictions, palette="viridis")
    ax1.set_xlabel("Diseases")
    ax1.set_ylabel("Prediction Probability")
    ax1.set_title("Model Confidence for Each Disease")
    plt.xticks(rotation=90)
    st.pyplot(fig1)
