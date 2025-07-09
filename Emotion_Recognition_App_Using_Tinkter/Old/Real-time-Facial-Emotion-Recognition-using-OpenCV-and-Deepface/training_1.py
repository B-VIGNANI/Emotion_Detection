import os
import sys
import cv2
import imageio
import numpy as np
from deepface import DeepFace
from datetime import datetime

# Optional: Enable UTF-8 encoding for terminals that support it
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass  # Fallback if not supported

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# === Path Configuration ===
emoji_dir = r"C:\Users\moham\Downloads\Emotion_Recognition_Project\Real-time-Facial-Emotion-Recognition-using-OpenCV-and-Deepface\Emoji"
save_path = r"C:\Users\moham\Downloads\Emotion_Captures"
os.makedirs(save_path, exist_ok=True)

# === Emotion to Emoji Mapping ===
emotion_to_emoji = {
    'angry': 'Angry.gif',
    'disgust': 'Disgust.gif',
    'fear': 'Fear.gif',
    'happy': 'Smile.gif',
    'sad': 'Sad face.gif',
    'surprise': 'Surprise.gif',
    'neutral': 'Neutral.gif'
}

# === Load Emoji GIFs (for top-left) and Static Emoji Icons (for bar graph) ===
emoji_gifs = {}
emoji_icons = {}

for emotion, filename in emotion_to_emoji.items():
    path = os.path.join(emoji_dir, filename)
    try:
        frames = imageio.mimread(path)
        emoji_gifs[emotion] = [cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) for frame in frames]
        emoji_icons[emotion] = cv2.resize(emoji_gifs[emotion][0], (18, 18))  # Static icon
    except Exception as e:
        print(f"[WARNING] Error loading emoji for '{emotion}': {e}")

# === Load Haar Cascade Face Detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Start Webcam ===
cap = cv2.VideoCapture(0)
emoji_frame_index = 0
dominant_emotion = None
emotion_scores = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_roi = rgb[y:y + h, x:x + w]

        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion'].lower()
            emotion_scores = result[0]['emotion']

            # Draw face rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, dominant_emotion.upper(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print("[WARNING] Emotion detection error:", e)
        break  # Only process the first face

    # === 1. Emoji GIF (Top-Left Corner) ===
    top_left_x = 10
    top_left_y = 10
    emoji_size = 144

    if dominant_emotion in emoji_gifs:
        emoji_frames = emoji_gifs[dominant_emotion]
        emoji_img = emoji_frames[emoji_frame_index % len(emoji_frames)]
        emoji_frame_index += 1

        emoji_resized = cv2.resize(emoji_img, (emoji_size, emoji_size))
        frame[top_left_y:top_left_y + emoji_size, top_left_x:top_left_x + emoji_size] = emoji_resized

        # Add emotion label under the emoji
        cv2.putText(frame, dominant_emotion.upper(),
                    (top_left_x, top_left_y + emoji_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # === 2. Compact Emotion Bar Graph with Emotion Names ===
    if emotion_scores:
        bar_x = top_left_x
        bar_y_start = top_left_y + emoji_size + 50
        bar_height = 8
        spacing = 4
        max_width = 60
        icon_size = 18

        for i, (emo, val) in enumerate(emotion_scores.items()):
            y = bar_y_start + i * (bar_height + spacing + 2)
            bar_len = int((val / 100) * max_width)

            # Draw bar background and filled portion
            cv2.rectangle(frame, (bar_x, y), (bar_x + max_width, y + bar_height), (220, 220, 220), -1)
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_len, y + bar_height), (255, 0, 0), -1)

            # Add emotion name next to the bar
            cv2.putText(frame, f"{emo.capitalize()}: {int(val)}%", (bar_x + max_width + 8, y + bar_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 80), 1)

    # === Display Frame ===
    cv2.imshow("Real-Time Emotion Detection", frame)
    key = cv2.waitKey(100) & 0xFF

    # === W = Save Frame ===
    if key == ord('w'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_path, f"emotion_capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[INFO] Image saved: {filename}")

    # === Q = Quit ===
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
