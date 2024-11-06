import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_option_menu import option_menu

# Load the model
model = load_model('first_mode.h5', compile=False)

# Define the class names (replace with your actual class names)
class_names = ['Disgust', 'Surprise', 'Fear', 'Happy', 'Neutral', 'Sad', 'Anger']  # Update with the real class names

# Set Streamlit to use centered layout
st.set_page_config(layout="centered", initial_sidebar_state="auto", page_title="Face Emotions Classification", page_icon="😊")

# Display logo image above sidebar
st.sidebar.image("logo.png", use_column_width=True)

# Sidebar navigation using Streamlit Option Menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Try the Model", "Project Overview"], 
                          icons=['camera', 'info'], menu_icon="cast", default_index=0)

# Page routing logic based on selected menu
if selected == "Try the Model":
    # Streamlit interface for image classification
    st.title('🖼️ Face Emotions with Pre-trained Model')

    # File uploader for an image
    uploaded_file = st.file_uploader("📂 Choose an image to classify", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        
        # Preprocess the image to match the input shape of the model
        image = image.resize((224, 224))  # Update to match your model's expected input size
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        if len(image_array.shape) == 2:  # If grayscale, convert to RGB
            image_array = np.stack((image_array,)*3, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(image_array)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_label]

        # Display the result side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)
        with col2:
            st.markdown(f"<h1 style='font-size: 32px;'>🔮 Predicted Class: {predicted_class}</h1>", unsafe_allow_html=True)

elif selected == "Project Overview":
    # Streamlit interface for project overview
    st.title('📋 Project Overview')
    st.markdown("""
    ### Project Introduction
    This project aims to identify emotions in facial images by leveraging a pre-trained deep learning model. The model processes an input image to detect the expressed emotion, providing insights into the mood or feelings of the person in the image.
    
    ### How It Works
    - Pre-trained Model: We use a pre-trained MobileNet model as the base, fine-tuned on a dataset of facial expressions. This allows the model to accurately classify different emotions, such as happiness, sadness, anger, and surprise.

    - Image Processing: When a user uploads an image, it’s automatically resized and normalized to meet the input requirements of the model, ensuring that the emotion is correctly detected.

    - Emotion Prediction: The model outputs a prediction, displaying the detected emotion along with a confidence score. This result is then shown on the screen, helping users understand the emotion conveyed in the image.
    """)

    st.markdown("""
    ### Sample from Prediction
    """)

    try:
            x =  Image.open("smaple from pridecton.jpg")
            st.image(x, caption="Multi Sample from Prediction", use_column_width=True)
    except FileNotFoundError:
        st.write("smaple could not be found.")
    
    # Display team members in two columns, aligned for consistency
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Team Members
        - **Abdulelah Bin Elias**: 
        GD Computer Science, Data Scientist
        """)
    with col2:
        st.markdown("""
        ###
        - **Abdullah Bin Ahmed**: 
        GD Computer Science, Data Scientist
        """)
    

    
    # Display business cards if available in the directory
    try:
        col1, col2 = st.columns(2)
        with col1:
            card1 = Image.open("First_BusCard.png")
            st.image(card1, caption="Eng. Abdulelah", use_column_width=True)
        with col2:
            card2 = Image.open("Second_BusCard.png")
            st.image(card2, caption="Eng. Abdullalh", use_column_width=True)
    except FileNotFoundError:
        st.write("Business cards could not be found.")
