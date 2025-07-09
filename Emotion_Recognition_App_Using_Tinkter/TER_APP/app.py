import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import requests
import tempfile
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST

# Load Model from GitHub
@st.cache_resource
def load_model_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(response.content)
            model = joblib.load(tmp_file.name)
        return model
    else:
        st.error("Failed to load the model from GitHub.")
        return None

model_url = "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/text/models/emotion_classifier_pipe_lr.pkl"
pipe_lr = load_model_from_github(model_url)

# Prediction Functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Emotion dictionaries
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮",
    "love": "❤️", "humor": "😄"
}

emotion_gif_dict = {
    "happy": "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExOW40YXNpc3dqMDlrMGZraDlpdmxkOGtkeGJtdGg2Mjl6Zm1wZ2VsNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/F7HZJkbC9piohwiR6N/giphy.gif",
    "anger": "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGN3aGR2dm5icGN6OXI4NXo5dGoxZG9hbW81bTZrYmYzbTd2ajV3diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/eKS2TiHojIeHcDhSUQ/giphy.gif",
    "neutral": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcW44c2x6cDM4a3pyMTF2NHZ1aTBzdHF1cjc1bWE1ZXk1cHRlaGRtciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7CXIO53h5YciXOp505/giphy.gif",
    "sad": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjJoN2YxeHEya2x1ZGVlbzZwMnh4MGJhMXNkcjM2YWduOXNxanVqMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/PQ36CJwR3mOnbhGU28/giphy.gif",
    "disgust": "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3J3MHB4aHE0cW9sc281bnc2cDd6MGF5cm45dTM5dDdsbTZseHptdSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5brTZxnO83UeETnFDR/giphy.gif",
    "fear": "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExOGEwcHF4MGZtY3Y2c3JsbDh6OGVmeWt3cWI5aGt6MDllMWJnOTVuYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wHKWXCXDFIsziGPOUR/giphy.gif",
    "surprise": "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcWV2M2M5cHRza2FsbGFscnI5cWxteHMydjd4bXdpb2M2Y3JtcnR1eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3OsFzorSZSUZcvo6UC/giphy.gif",
    "love": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYTI3eHl2MnBxMHV6MzJ6OGdpa3M3c3JtcGRkbmRubDJ4MDcxdHJjNCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/DDW1pNueYIFZAz09Ku/giphy.gif",
    "humor": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNTlmOWs4Zm9ndDY1YXc3aHF2NGQyMTMwaHNkNzRsOWZ0eDdpZ2lucSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/2fIbmaiOnI3VlQFZEq/giphy.gif"
}

# Main App
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    create_page_visited_table()
    create_emotionclf_table()

    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))
        st.subheader("Emotion Detection in Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "")
                st.write(f"{prediction}: {emoji_icon}")
                st.write(f"Confidence: {np.max(probability):.2f}")

                # Show Emotion GIF
                gif_url = emotion_gif_dict.get(prediction, None)
                if gif_url:
                    st.image(gif_url, caption=f"{prediction.title()} GIF", use_container_width=True)

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Page Name', y='Counts', color='Page Name')
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)

    else:
        add_page_visited_details("About", datetime.now(IST))
        st.subheader("About This App")

        st.write("Welcome to the Emotion Detection in Text App! This application utilizes NLP and ML to analyze and identify emotions in text.")

        st.subheader("Mission")
        st.write("We aim to provide a user-friendly and efficient tool to uncover emotional insights from written language.")

        st.subheader("How It Works")
        st.write("Enter text, and the model analyzes it to return the detected emotion along with a confidence score and visual feedback.")

        st.subheader("Key Features:")
        st.markdown("##### 1. Real-time Emotion Detection")
        st.markdown("##### 2. Confidence Score")
        st.markdown("##### 3. User-friendly Interface")

        st.subheader("Applications")
        st.markdown("""
        - Social media sentiment analysis  
        - Customer feedback analysis  
        - Market research  
        - Brand monitoring  
        - Content recommendation
        """)

if __name__ == '__main__':
    main()
