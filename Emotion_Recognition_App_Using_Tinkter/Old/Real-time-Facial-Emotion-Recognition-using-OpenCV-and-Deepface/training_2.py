import os
import sys
import cv2
import numpy as np
import imageio
from deepface import DeepFace
from datetime import datetime

# === Emoji & Save Paths ===
emoji_dir = r"C:\Users\moham\Downloads\Emotion_Recognition_Project\Real-time-Facial-Emotion-Recognition-using-OpenCV-and-Deepface\Emoji"
save_path = r"C:\Users\moham\Downloads\Emotion_Captures"
os.makedirs(save_path, exist_ok=True)

# === Emoji Mapping ===
emotion_to_emoji = {
    'angry': 'Angry.gif',
    'disgust': 'Disgust.gif',
    'fear': 'Fear.gif',
    'happy': 'Smile.gif',
    'sad': 'Sad face.gif',
    'surprise': 'Surprise.gif',
    'neutral': 'Neutral.gif'
}

# === Load Emojis ===
emoji_gifs = {}
emoji_icons = {}

for emotion, filename in emotion_to_emoji.items():
    path = os.path.join(emoji_dir, filename)
    try:
        frames = imageio.mimread(path)
        emoji_gifs[emotion] = [cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) for frame in frames]
        emoji_icons[emotion] = cv2.resize(emoji_gifs[emotion][0], (24, 24))  # Static icon
    except Exception as e:
        print(f"[WARNING] Could not load emoji '{emotion}': {e}")

# === Load Face Detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Webcam Init ===
cap = cv2.VideoCapture(0)
emoji_frame_index = 0
dominant_emotion = None
emotion_scores = None
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    face_count = len(faces)

    for (x, y, w, h) in faces:
        face_roi = rgb[y:y + h, x:x + w]

        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion'].lower()
            emotion_scores = result[0]['emotion']

            # Draw green box with emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion.upper(), (x, y - 10), font, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print("[WARNING] Emotion detection error:", e)
        break  # Only process the first face

    # === LEFT PANEL CREATION ===
    panel_width = 240
    panel = np.ones((frame.shape[0], panel_width, 3), dtype=np.uint8) * 255

    # Top emoji gif
    if dominant_emotion in emoji_gifs:
        emoji_frames = emoji_gifs[dominant_emotion]
        emoji_img = emoji_frames[emoji_frame_index % len(emoji_frames)]
        emoji_frame_index += 1
        emoji_resized = cv2.resize(emoji_img, (144, 144))
        panel[10:10 + 144, 48:48 + 144] = emoji_resized
        cv2.putText(panel, dominant_emotion.upper(), (65, 170), font, 0.6, (0, 0, 255), 2)

    # Bar chart with emotion values
    if emotion_scores:
        sorted_emotions = sorted(emotion_scores.items(), key=lambda item: -item[1])
        bar_start_y = 200
        bar_height = 18
        spacing = 6
        max_bar_width = 100

        for idx, (emo, val) in enumerate(sorted_emotions):
            y = bar_start_y + idx * (bar_height + spacing)

            # Draw icon
            if emo in emoji_icons:
                panel[y:y + 24, 10:34] = emoji_icons[emo]

            # Draw emotion label and bar
            bar_len = int((val / 100) * max_bar_width)
            cv2.rectangle(panel, (40, y + 5), (140, y + 5 + bar_height), (230, 230, 230), -1)
            cv2.rectangle(panel, (40, y + 5), (40 + bar_len, y + 5 + bar_height), (0, 100, 255), -1)
            cv2.putText(panel, f"{emo.upper()} : {int(val)}%", (145, y + 20), font, 0.4, (0, 0, 0), 1)

    # Face Count
    cv2.putText(frame, f"Faces: {face_count}", (10, 30), font, 0.7, (255, 255, 255), 2)

    # === Combine Left Panel and Webcam Frame ===
    combined = np.hstack((panel, frame))
    cv2.imshow("Real-Time Emotion Detection", combined)

    key = cv2.waitKey(100) & 0xFF

    if key == ord('w'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_path, f"emotion_capture_{timestamp}.jpg")
        cv2.imwrite(filename, combined)
        print(f"[INFO] Saved: {filename}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
