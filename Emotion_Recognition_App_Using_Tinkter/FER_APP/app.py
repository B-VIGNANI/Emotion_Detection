import streamlit as st
import cv2
from deepface import DeepFace
import tempfile
import os
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="Facial Emotion Recognition",
    layout="centered",
    initial_sidebar_state="auto"
)

# Emojis for display
emotion_emoji_dict = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

# Emotion GIFs dictionary
emotion_gif_dict = {
    "angry": "https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif",
    "disgust": "https://media.giphy.com/media/jt7bAtEijhurm/giphy.gif",
    "fear": "https://media.giphy.com/media/3o6ZtpxSZbQRRnwCKQ/giphy.gif",
    "happy": "https://media.giphy.com/media/5GoVLqeAOo6PK/giphy.gif",
    "sad": "https://media.giphy.com/media/9Y5BbDSkSTiY8/giphy.gif",
    "surprise": "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif",
    "neutral": "https://media.giphy.com/media/l0MYydaQ2bErcsCVS/giphy.gif"
}

# Title
st.title("Facial Emotion Recognition App 😊😢😠")

st.markdown("Detect your facial **emotion in real-time** using `DeepFace`, `OpenCV`, and fun emotion-based GIFs!")

# Upload image option
uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

# Webcam option
use_webcam = st.checkbox("Use Webcam")

def detect_emotion(img_path):
    try:
        result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        st.error(f"Error during emotion detection: {e}")
        return None

def capture_webcam_frame():
    cap = cv2.VideoCapture(0)
    st.info("⏳ Opening webcam... Press 'q' to capture.")
    captured = False
    img_path = None

    while cap.isOpened() and not captured:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from webcam.")
            break

        cv2.imshow("Press 'q' to capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Save frame temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            img_path = temp_file.name
            cv2.imwrite(img_path, frame)
            captured = True

    cap.release()
    cv2.destroyAllWindows()
    return img_path

def main():
    image_path = None

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file.name)
        image_path = temp_file.name

    elif use_webcam:
        image_path = capture_webcam_frame()
        if image_path:
            st.image(image_path, caption="Captured Image", use_column_width=True)

    # If image is available
    if image_path:
        with st.spinner("Analyzing Emotion..."):
            pred = detect_emotion(image_path)
            time.sleep(1)

        if pred:
            st.success(f"Detected Emotion: **{pred.capitalize()}** {emotion_emoji_dict.get(pred.lower(), '')}")
            gif_url = emotion_gif_dict.get(pred.lower(), "")
            if gif_url:
                st.image(gif_url, width=150)
            else:
                st.warning("No GIF found for this emotion.")
        else:
            st.warning("Emotion detection failed. Try another image or retake photo.")

if __name__ == "__main__":
    main()
