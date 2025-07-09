import streamlit as st
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
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import kagglehub
import warnings
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

# Emoji GIF URLs
emoji_url_map = {
    'angry': "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGN3aGR2dm5icGN6OXI4NXo5dGoxZG9hbW81bTZrYmYzbTd2ajV3diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/eKS2TiHojIeHcDhSUQ/giphy.gif",
    'disgust': "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3J3MHB4aHE0cW9sc281bnc2cDd6MGF5cm45dTM5dDdsbTZseHptdSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5brTZxnO83UeETnFDR/giphy.gif",
    'fear': "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExOGEwcHF4MGZtY3Y2c3JsbDh6OGVmeWt3cWI5aGt6MDllMWJnOTVuYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wHKWXCXDFIsziGPOUR/giphy.gif",
    'happy': "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExOW40YXNpc3dqMDlrMGZraDlpdmxkOGtkeGJtdGg2Mjl6Zm1wZ2VsNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/F7HZJkbC9piohwiR6N/giphy.gif",
    'sad': "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjJoN2YxeHEya2x1ZGVlbzZwMnh4MGJhMXNkcjM2YWduOXNxanVqMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/PQ36CJwR3mOnbhGU28/giphy.gif",
    'surprise': "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcWV2M2M5cHRza2FsbGFscnI5cWxteHMydjd4bXdpb2M2Y3JtcnR1eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3OsFzorSZSUZcvo6UC/giphy.gif",
    'neutral': "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcW44c2x6cDM4a3pyMTF2NHZ1aTBzdHF1cjc1bWE1ZXk1cHRlaGRtciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7CXIO53h5YciXOp505/giphy.gif",
    'love': "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYTI3eHl2MnBxMHV6MzJ6OGdpa3M3c3JtcGRkbmRubDJ4MDcxdHJjNCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/DDW1pNueYIFZAz09Ku/giphy.gif",
    'humor': "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNTlmOWs4Zm9ndDY1YXc3aHF2NGQyMTMwaHNkNzRsOWZ0eDdpZ2lucSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/2fIbmaiOnI3VlQFZEq/giphy.gif"
}

# Load text emotion model
model_url = "https://github.com/ADHIL48/Emotion_Recognition_app/raw/f1096452ed11fde1454dbd13d6e006a9a8ea1412/text/models/emotion_classifier_pipe_lr.pkl"
model_path = "./emotion_classifier_pipe_lr.pkl"
if not os.path.exists(model_path):
    urllib.request.urlretrieve(model_url, model_path)
pipe_lr = joblib.load(open(model_path, "rb"))

# Load speech emotion model
speech_model_url = "https://github.com/Bhagyansh-garg/Speech_Emotion_recognition/raw/master/emotion_model.pkl"
speech_model_path = "emotion_model.pkl"
if not os.path.exists(speech_model_path):
    urllib.request.urlretrieve(speech_model_url, speech_model_path)
speech_model = joblib.load(open(speech_model_path, "rb"))

# Feature extractor for speech model
def extract_features_from_audio(file):
    audio, sample_rate = librosa.load(file, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

@st.cache_data
def predict_speech_emotion(audio_path):
    features = extract_features_from_audio(audio_path).reshape(1, -1)
    prediction = speech_model.predict(features)[0]
    return prediction

# Main App
def main():
    st.title("🧠 Emotion Recognition App")
    menu = ["Home", "Text Analysis", "Facial Emotion Detection", "Speech Emotion Recognition", "EEG Emotion Recognition", "Info"]
    choice = st.sidebar.selectbox("Select Feature", menu)

    if choice == "Home":
        pass  # No text or content on home page

    elif choice == "Text Analysis":
        st.subheader("Text-Based Emotion Detection")
        with st.form(key='emotion_form'):
            user_text = st.text_area("Enter text to analyze emotion")
            submit = st.form_submit_button(label='Analyze')

        if submit:
            pred = pipe_lr.predict([user_text])[0]
            proba = pipe_lr.predict_proba([user_text])
            st.success(f"Detected Emotion: {pred}")
            st.image(emoji_url_map.get(pred.lower(), ""), width=100)

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
                st.image(emoji_url_map.get(dominant.lower(), ""), width=150)

                df_scores = pd.DataFrame(scores.items(), columns=["Emotion", "Score"])
                st.bar_chart(df_scores.set_index("Emotion"))
            except Exception as e:
                st.error(f"Detection Error: {e}")

    elif choice == "Speech Emotion Recognition":
        st.subheader("Upload Speech Audio")
        uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.read())

            pred = predict_speech_emotion("temp_audio.wav")
            st.success(f"Detected Emotion: {pred}")
            st.image(emoji_url_map.get(pred.lower(), ""), width=150)

    elif choice == "EEG Emotion Recognition":
        st.subheader("EEG Emotion Recognition (See Notebook for Implementation)")
        st.markdown("""
        You can view the EEG emotion recognition notebook and implementation [here](https://github.com/ADHIL48/Emotion_Recognition_app/blob/main/EEG/Emotion_Prediction.ipynb).
        """)
        
    else:
        st.subheader("📌 About the App")
        st.markdown("""
        This application detects human emotions using text, facial expressions, speech, and EEG signals.

        💡 *Use Case*
        Useful in psychology, entertainment, sentiment analysis, and user experience design.

        🚀 *Features*
        - Text-based emotion recognition using NLP
        - Real-time facial emotion detection using DeepFace
        - Speech emotion recognition using audio clips
        - EEG emotion recognition through the SEED dataset
        - Emojis and GIFs for visual feedback

        👨‍💻 *Developed By*
        Adhil M  
        📧 mohammedadhil0408@gmail.com  
        🌐 GitHub: [adhil48](https://github.com/ADHIL48)
        """)

if __name__ == "__main__":
    main()
