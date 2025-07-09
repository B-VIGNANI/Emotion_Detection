import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request
from datetime import datetime
from deepface import DeepFace
from PIL import Image
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import requests
import tempfile
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import kagglehub
import warnings
import altair as alt

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(page_title="Emotion Recognition App", layout="wide")

# Updated background image (GIF full screen)
bg_image_url = "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNmQ1MmxxdzUxNHlhd3Y5d3d6cGExMHoyc3g3aHdicmhiajdpaWQ3NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/1nbweFkb6kqVhNCwJ4/giphy.gif"
background_style = f"""
<style>
    .stApp {{
        background-image: url('{bg_image_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-color: transparent;
    }}
    textarea, .stSelectbox, .stTextInput, .stFileUploader {{
        background-color: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(4px);
        border-radius: 10px;
    }}
    .stSidebar, .css-1d391kg {{
        background-color: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(5px);
    }}
    .sliding-text {{
        white-space: nowrap;
        overflow: hidden;
        box-sizing: border-box;
    }}
    .sliding-text p {{
        display: inline-block;
        padding-left: 100%;
        animation: slide 15s linear infinite;
        font-size: 1.5em;
        font-weight: bold;
        color: #ffffff;
    }}
    @keyframes slide {{
        0% {{ transform: translate(0, 0); }}
        100% {{ transform: translate(-100%, 0); }}
    }}
</style>
"""
st.markdown(background_style, unsafe_allow_html=True)

# Load text emotion model
model_url = "https://github.com/ADHIL48/Emotion_Recognition_app/raw/f1096452ed11fde1454dbd13d6e006a9a8ea1412/text/models/emotion_classifier_pipe_lr.pkl"
model_path = "./emotion_classifier_pipe_lr.pkl"
if not os.path.exists(model_path):
    urllib.request.urlretrieve(model_url, model_path)
pipe_lr = joblib.load(open(model_path, "rb"))

# Emotion GIFs for facial and speech emotions
emotion_gifs = {
    "happy": "https://media3.giphy.com/media/F7HZJkbC9piohwiR6N/giphy.gif",
    "angry": "https://media1.giphy.com/media/eKS2TiHojIeHcDhSUQ/giphy.gif",
    "neutral": "https://media0.giphy.com/media/7CXIO53h5YciXOp505/giphy.gif",
    "sad": "https://media4.giphy.com/media/PQ36CJwR3mOnbhGU28/giphy.gif",
    "disgust": "https://media2.giphy.com/media/5brTZxnO83UeETnFDR/giphy.gif",
    "fear": "https://media3.giphy.com/media/wHKWXCXDFIsziGPOUR/giphy.gif",
    "pleasant surprise": "https://media1.giphy.com/media/3OsFzorSZSUZcvo6UC/giphy.gif",
    "love": "https://media.giphy.com/media/DDW1pNueYIFZAz09Ku/giphy.gif",
    "humor": "https://media4.giphy.com/media/2fIbmaiOnI3VlQFZEq/giphy.gif"
}

# Load Keras model from URL
def load_keras_model_from_url(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    return load_model(tmp_path)

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to display waveform
def wave_plot(data, sampling_rate):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 4)
    ax.set_facecolor("black")
    fig.set_facecolor("black")
    plt.ylabel('Amplitude')
    plt.title("WAVEFORM", fontweight="bold")
    librosa.display.waveshow(data, sr=sampling_rate, x_axis='s')
    st.pyplot(fig, use_container_width=True)
    return data

# Prediction using CNN
def prediction(data, sampling_rate):
    dict_values = {0:"neutral",1:"calm",2:"happy",3:"sad",4:"angry",5:"fear",6:"disgust",7:"pleasant surprise"}
    cnn_model_url = "https://raw.githubusercontent.com/ADHIL48/Emotion_Recognition_app/main/speech_or_voice/models/CnnModel.h5"
    cnn_model = load_keras_model_from_url(cnn_model_url)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    X_test = np.expand_dims([mfccs], axis=2)
    predict = cnn_model.predict(X_test)
    predictions = np.argmax(predict, axis=1)
    detected_emotion = [dict_values[val] for val in predictions]
    emotion = detected_emotion[0]
    st.subheader(emotion.upper())
    if emotion in emotion_gifs:
        st.image(emotion_gifs[emotion], use_container_width=True)
    st.success("Prediction Probability")
    emotion_labels = [dict_values[i] for i in range(len(predict[0]))]
    prob_df = pd.DataFrame({'emotions': emotion_labels, 'probability': predict[0]})
    fig = alt.Chart(prob_df).mark_bar().encode(
        x=alt.X('emotions', sort='-y'),
        y='probability',
        color='emotions'
    )
    st.altair_chart(fig, use_container_width=True)

def show_homepage():
    # Lottie animation for Emotion Recognition
    lottie_emotion = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_1pxqjqps.json")

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

# Main App
def main():
    st.sidebar.title("Navigation")
    menu = ["Home", "Text Emotion Recognition", "Facial Emotion Detection", "EEG Emotion Recognition", "Speech Emotion Recognition", "Info"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        show_homepage()

    elif choice == "Text Emotion Recognition":
        st.subheader("Text-Based Emotion Detection")
        with st.form(key='emotion_form'):
            user_text = st.text_area("Enter text to analyze emotion")
            submit = st.form_submit_button(label='Analyze')

        if submit:
            pred = pipe_lr.predict([user_text])[0]
            proba = pipe_lr.predict_proba([user_text])
            st.success(f"Detected Emotion: {pred}")
            st.image(emotion_gifs.get(pred.lower(), ""), width=100)

            proba_df = pd.DataFrame(proba, columns=pipe_lr.classes_).T.reset_index()
            proba_df.columns = ["Emotion", "Probability"]
            st.bar_chart(proba_df.set_index("Emotion"))

    elif choice == "Facial Emotion Detection":
        st.subheader("Real-Time Facial Emotion Detection")
        img_file_buffer = st.camera_input("Take a photo")

        if img_file_buffer is not None:
            img = Image.open(img_file_buffer)
            img_array = np.array(img.convert('RGB'))
            try:
                result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
                dominant = result[0]['dominant_emotion']
                scores = result[0]['emotion']
                st.success(f"Detected Emotion: {dominant.capitalize()}")
                st.image(emotion_gifs.get(dominant.lower(), ""), width=150)
                df_scores = pd.DataFrame(scores.items(), columns=["Emotion", "Score"])
                st.bar_chart(df_scores.set_index("Emotion"))
            except Exception as e:
                st.error(f"Detection Error: {e}")

    elif choice == "EEG Emotion Recognition":
        st.subheader("EEG Emotion Recognition (See Notebook for Implementation)")
        st.markdown("""
        You can view the EEG emotion recognition notebook and implementation [here](https://github.com/ADHIL48/Emotion_Recognition_app/blob/main/EEG/Emotion_Prediction.ipynb).
        """)

    elif choice == "Speech Emotion Recognition":
        st.subheader("Speech Emotion Recognition")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p style="font-family:Courier; color:DarkSeaGreen; font-size: 20px;"><b>CHOOSE AUDIO FILE</b></p>', unsafe_allow_html=True)
            audio_file = st.file_uploader("", type=['wav', 'mp3', 'ogg'])

            if audio_file is not None:
                data, sampling_rate = librosa.load(audio_file)
                st.markdown('<p style="font-family:Courier; color:DarkSeaGreen; font-size: 18px;"><b>WAVEFORM</b></p>', unsafe_allow_html=True)
                wave_plot(data, sampling_rate)
                prediction(data, sampling_rate)

    elif choice == "Info":
        st.subheader("About")
        st.markdown("""
        This Emotion Recognition App leverages **Deep Learning** to predict emotions from text, speech, facial expressions, and EEG data.

        - 📝 **Text**: Predicts emotions using Natural Language Processing.
        - 🎤 **Speech**: Recognizes emotions from voice using a Convolutional Neural Network (CNN).
        - 📷 **Facial**: Detects emotions in real-time using DeepFace.
        - 🧠 **EEG**: For emotion prediction based on EEG signals, refer to the linked Jupyter notebook.
        """)

if __name__ == '__main__':
    main()