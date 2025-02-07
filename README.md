Plant Disease Detection Using Deep Learning
A deep learning-based approach for detecting plant diseases using transfer learning and CNN models.

** Project Overview**
This project uses Convolutional Neural Networks (CNNs) and transfer learning to classify plant diseases from leaf images. The model is trained using EfficientNetB0, Xception, and DenseNet121, with EfficientNetB0 achieving the highest accuracy of 96%.

The trained model is deployed as a Streamlit web application, where users can upload images of leaves to detect possible plant diseases.

** Dataset Information**
Dataset: New Plant Diseases Dataset (Augmented)
Classes: Multiple plant diseases + healthy leaves
Images: High-resolution images of leaves
Preprocessing: Resized to 96×96, normalized, and augmented

** Key Features**
Transfer Learning: Uses pre-trained CNNs (EfficientNetB0, Xception, DenseNet121)
High Accuracy: EfficientNetB0 achieves 96% validation accuracy
Real-Time Prediction: Fast inference using optimized models
Web App: Streamlit-based GUI for easy use
Data Augmentation: Enhances model generalization

**Web Application Usage**
Upload an image of a plant leaf.
The model analyzes the image and predicts the disease.
The app displays the predicted disease class and confidence score.
