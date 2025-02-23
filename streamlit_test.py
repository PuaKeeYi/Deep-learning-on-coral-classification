import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the CNN model
model = load_model('CNN_Final_Model.h5')

# Function to preprocess the image
def preprocess_image(image):
    size = (225, 225)  # Change size based on your model input
    image = ImageOps.fit(image, size)
    image = np.asarray(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict image class
def predict_image_class(image, model, class_names):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = class_names[np.argmax(prediction)]
    prediction_probs = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    return predicted_class, prediction_probs

# Streamlit app
st.title('Image Classification with CNN')

uploaded_files = st.file_uploader('Choose images...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
if uploaded_files:
    class_names = ['bleached coral', 'healthy coral']  # Replace with your actual class names
    
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
            st.write('Classifying...')

            predicted_class, prediction_probs = predict_image_class(image, model, class_names)

            st.write(f'Prediction: {predicted_class}')
            st.write('Prediction Probabilities:')
            st.write(prediction_probs)
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")