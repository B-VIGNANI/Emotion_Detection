import streamlit as st
import requests
from streamlit_lottie import st_lottie

# Set page configuration
st.set_page_config(page_title="Emotion Recognition App", layout="wide")

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation for Emotion Recognition
lottie_emotion = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_1pxqjqps.json")

# Set a custom font style for the entire app
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Create two columns: one for the title and one for the animation
col1, col2 = st.columns([4, 1])

# Title in the center and animation on the left side near the title
with col1:
    st.markdown("""
        <h1 style="text-align: center; color: #4CAF50;">Emotion Recognition App</h1>
    """, unsafe_allow_html=True)

with col2:
    st_lottie(lottie_emotion, speed=1, width=200, height=200, key="emotion")  # Size is approximately 3 inches (200x200 px)

# Add space between sections
st.write("")  

# Colorful Box with About the App
with st.container():
    st.markdown("""
    <div style="border: 2px solid #FF5722; background-color: #FFE0B2; padding: 20px; border-radius: 10px;">
        <h3 style="color: #FF5722;">About the App</h3>
        <p style="color: #000000;">The Emotion Recognition app leverages state-of-the-art technologies to analyze and detect emotions from facial expressions, speech, and EEG signals. It aims to help users understand their emotional states in real-time, providing a deeper connection with their mental well-being. By using advanced AI and machine learning models, this app offers emotion-based GIFs and personalized feedback to improve emotional health.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Colorful Box with Inspiration
with st.container():
    st.markdown("""
    <div style="border: 2px solid #2196F3; background-color: #BBDEFB; padding: 20px; border-radius: 10px;">
        <h3 style="color: #2196F3;">Inspiration 💡</h3>
        <p style="color: #000000;">The app was inspired by the need to bridge the gap between technology and emotional intelligence. By integrating various methods of emotion recognition, such as facial, speech, and EEG signal analysis, we aim to provide users with a comprehensive emotional experience, enhancing self-awareness and emotional understanding.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Colorful Box with Vision and Mission
with st.container():
    st.markdown("""
    <div style="border: 2px solid #009688; background-color: #B2DFDB; padding: 20px; border-radius: 10px;">
        <h3 style="color: #009688;">Vision and Mission 🎯</h3>
        <p style="color: #000000;">The vision is to revolutionize emotional intelligence through technology, offering individuals real-time tools to monitor and improve their emotional well-being. The mission is to create an easy-to-use platform that can detect emotions using facial expressions, speech patterns, and EEG signals, helping users gain insights into their emotional states and fostering mental health awareness.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Colorful Box with Motivation
with st.container():
    st.markdown("""
    <div style="border: 2px solid #FF9800; background-color: #FFCC80; padding: 20px; border-radius: 10px;">
        <h3 style="color: #FF9800;">Motivation 💪</h3>
        <p style="color: #000000;">Our motivation comes from the growing need to address mental health in the digital age. By providing immediate emotional insights, we hope to empower individuals to better manage their emotions, build healthier habits, and ultimately lead more balanced lives.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Colorful Box with Use Case
with st.container():
    st.markdown("""
    <div style="border: 2px solid #3F51B5; background-color: #C5CAE9; padding: 20px; border-radius: 10px;">
        <h3 style="color: #3F51B5;">Use Case 📱</h3>
        <p style="color: #000000;">Mental Health helps individuals track their emotions and receive personalized recommendations to improve emotional well-being.</p>
        <p style="color: #000000;">Healthcare assists professionals in monitoring patients' emotional states, particularly in mental health care.</p>
        <p style="color: #000000;">Customer Service enhances user experience by analyzing customer emotions during interactions and adjusting responses accordingly.</p>
        <p style="color: #000000;">Personal Development provides users with insights into their emotional trends over time, helping them build emotional resilience.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Colorful Box with Key Features
with st.container():
    st.markdown("""
    <div style="border: 2px solid #9C27B0; background-color: #E1BEE7; padding: 20px; border-radius: 10px;">
        <h3 style="color: #9C27B0;">Key Features 🔑</h3>
        <p style="color: #000000;">Real-time Emotion Recognition detects emotions through facial expressions, speech, and EEG signals.</p>
        <p style="color: #000000;">Emotion-Based GIFs display a GIF that matches the detected emotion to provide emotional feedback.</p>
        <p style="color: #000000;">User-Friendly Interface offers a simple and intuitive design for ease of use.</p>
        <p style="color: #000000;">Multiple Input Modes support facial recognition, speech input, and EEG signals for emotion detection.</p>
        <p style="color: #000000;">Personalized Feedback offers suggestions based on detected emotional state for self-improvement.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Colorful Box with Technologies Used
with st.container():
    st.markdown("""
    <div style="border: 2px solid #FF9800; background-color: #FFE082; padding: 20px; border-radius: 10px;">
        <h3 style="color: #FF9800;">Technologies Used ⚙️</h3>
        <p style="color: #000000;">DeepFace is used for facial recognition to detect emotions through facial expressions.</p>
        <p style="color: #000000;">OpenCV is used for real-time image processing.</p>
        <p style="color: #000000;">TensorFlow is the deep learning framework used for building and training emotion detection models.</p>
        <p style="color: #000000;">MLP and CNN are used for processing and classifying speech emotion data.</p>
        <p style="color: #000000;">Streamlit is the framework used for developing the real-time emotion recognition app.</p>
        <p style="color: #000000;">Python is the main programming language for development.</p>
        <p style="color: #000000;">scikit-learn is used for machine learning tasks such as emotion classification.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Colorful Box with User Guide
with st.container():
    st.markdown("""
    <div style="border: 2px solid #607D8B; background-color: #B0BEC5; padding: 20px; border-radius: 10px;">
        <h3 style="color: #607D8B;">User Guide 📚</h3>
        <p style="color: #000000;">For Facial Emotion Recognition, upload or enable your webcam to detect emotions from facial expressions. The app will display a corresponding GIF based on the detected emotion.</p>
        <p style="color: #000000;">For Speech Emotion Recognition, upload an audio file or speak directly to the app to detect emotion from your voice.</p>
        <p style="color: #000000;">For EEG Signal Emotion Recognition, connect the EEG device and process the signals to detect your emotional state.</p>
        <p style="color: #000000;">For Interpretation, view the emotion classification result and related feedback, including suggestions to improve emotional health.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Colorful Box with Privacy and Data Security
with st.container():
    st.markdown("""
    <div style="border: 2px solid #8BC34A; background-color: #C8E6C9; padding: 20px; border-radius: 10px;">
        <h3 style="color: #8BC34A;">Privacy and Data Security 🔒</h3>
        <p style="color: #000000;">Data storage is temporary. No personal data or media is stored permanently. The data used for emotion detection is processed temporarily for the duration of the session.</p>
        <p style="color: #000000;">Data is anonymized to ensure user privacy.</p>
        <p style="color: #000000;">The app requests user consent before collecting or processing any data, ensuring full transparency.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Colorful Box with Performance Metrics
with st.container():
    st.markdown("""
    <div style="border: 2px solid #F44336; background-color: #FFCDD2; padding: 20px; border-radius: 10px;">
        <h3 style="color: #F44336;">Performance Metrics 📊</h3>
        <p style="color: #000000;">The system has an accuracy rate of 85% for emotion detection from facial expressions, speech, and EEG signals.</p>
        <p style="color: #000000;">Real-time performance is optimized for smooth interaction, ensuring minimal latency.</p>
        <p style="color: #000000;">Emotion classification results are delivered in under 1 second for faster feedback.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Colorful Box with Future Enhancements
with st.container():
    st.markdown("""
    <div style="border: 2px solid #00BCD4; background-color: #B2EBF2; padding: 20px; border-radius: 10px;">
        <h3 style="color: #00BCD4;">Future Enhancements 🚀</h3>
        <p style="color: #000000;">Integrating more detailed emotion recognition, such as detecting mixed emotions and providing more tailored feedback.</p>
        <p style="color: #000000;">Adding more advanced features like sentiment analysis for text-based emotion recognition.</p>
        <p style="color: #000000;">Enhancing the system with multi-modal emotion detection (combining speech, facial, and EEG data).</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")  # Add space between boxes

# Developed By Box with updated info
with st.container():
    st.markdown("""
    <div style="border: 2px solid #e1e1e1; background-color: #E8F5E9; padding: 20px; border-radius: 10px;">
        <h3 style="color: #FF5722; font-size: 18px;">Developed By 💻</h3>
        <p style="color: #000000; font-size: 14px;">ADHIL M</p>
        <p style="color: #000000; font-size: 14px;">mohammedadhil0408@gmail.com</p>
        <p style="color: #000000; font-size: 14px;">github.com/adhil48</p>
    </div>
    """, unsafe_allow_html=True)
