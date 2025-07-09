import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import requests
import joblib
import tempfile
from tensorflow.keras.models import load_model

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

# Load Keras model from URL
def load_keras_model_from_url(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    return load_model(tmp_path)

# Load Pickle model from URL
def load_pickle_model_from_url(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    return joblib.load(tmp_path)

# Emotion GIFs
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

    st.markdown('<p style="font-family:Courier; color:DarkSeaGreen; font-size: 18px;"><b>EMOTION DETECTED</b></p>', unsafe_allow_html=True)
    st.subheader(emotion.upper())

    if emotion in emotion_gifs:
        st.image(emotion_gifs[emotion], use_container_width=True)

    # Emotion Probability Bar Graph
    st.success("Prediction Probability")
    emotion_labels = [dict_values[i] for i in range(len(predict[0]))]
    prob_df = pd.DataFrame({'emotions': emotion_labels, 'probability': predict[0]})

    fig = alt.Chart(prob_df).mark_bar().encode(
        x=alt.X('emotions', sort='-y'),
        y='probability',
        color='emotions'
    )
    st.altair_chart(fig, use_container_width=True)

# Prediction using MLP
def prediction_mlp(data, sampling_rate):
    dict_values = {0:"neutral",1:"calm",2:"happy",3:"sad",4:"angry",5:"fear",6:"disgust",7:"pleasant surprise"}

    mlp_model_url = "https://raw.githubusercontent.com/ADHIL48/Emotion_Recognition_app/main/speech_or_voice/models/MLP_model.pkl"
    MLP_model = load_pickle_model_from_url(mlp_model_url)

    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    predict = MLP_model.predict([mfccs])
    detected_emotion = [dict_values[val] for val in predict]
    emotion = detected_emotion[0]

    st.markdown('<p style="font-family:Courier; color:DarkSeaGreen; font-size: 18px;"><b>EMOTION DETECTED</b></p>', unsafe_allow_html=True)
    st.subheader(emotion.upper())

    if emotion in emotion_gifs:
        st.image(emotion_gifs[emotion], use_container_width=True)

# Main Streamlit app
def main():
    st.title("🎙️ Speech Emotion Classifier App")
    menu = ["CNN Model", "MLP Model", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    col1, col2 = st.columns(2)

    if choice in ["CNN Model", "MLP Model"]:
        with col1:
            st.markdown('<p style="font-family:Courier; color:DarkSeaGreen; font-size: 20px;"><b>CHOOSE AUDIO FILE</b></p>', unsafe_allow_html=True)
            audio_file = st.file_uploader("", type=['wav', 'mp3', 'ogg'])

            if audio_file is not None:
                data, sampling_rate = librosa.load(audio_file)

                st.markdown('<p style="font-family:Courier; color:DarkSeaGreen; font-size: 18px;"><b>WAVE FORM</b></p>', unsafe_allow_html=True)
                data = wave_plot(data, sampling_rate)

                with col2:
                    st.markdown('<p style="font-family:Courier; color:DarkSeaGreen; font-size: 18px;"><b>PLAY AUDIO</b></p>', unsafe_allow_html=True)
                    st.audio(audio_file, format='audio/wav', start_time=0)

                if choice == "CNN Model":
                    prediction(data, sampling_rate)
                else:
                    prediction_mlp(data, sampling_rate)

    else:
        st.markdown('<p style="font-family:Courier; color:Teal; font-size: 18px;"><b>Speech Emotion Recognition (SER) extracts emotional state of the speaker from speech. This app uses CNN and MLP models to classify your emotions from speech audio files.</b></p>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Courier; color:Teal; font-size: 25px;"><b>CREATED BY</b></p>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Courier; color:Teal; font-size: 20px;"><b>ADHIL M</b></p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
