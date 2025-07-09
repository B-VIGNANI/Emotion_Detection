import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.layers import TimeDistributed
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ==== CONFIG ====
DATASET_PATH = r"C:\Users\moham\Downloads\Emotion_Recognition_Project\Speech_Emotion_Recognition\TESS Toronto emotional speech set data"
EMOJI_PATH = r"C:\Users\moham\Downloads\Emotion_Recognition_Project\Speech_Emotion_Recognition\Emotion Emojis"

# ==== FEATURE EXTRACTION ====
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    return features

# ==== LOAD DATA ====
data, labels = [], []
for folder_name in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder_name)
    for file_path in glob.glob(os.path.join(folder_path, '*.wav')):
        features = extract_features(file_path)
        data.append(features)
        labels.append(folder_name)

# ==== ENCODE + SPLIT ====
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)
X_train = np.array(X_train)[:, np.newaxis, :]
X_test = np.array(X_test)[:, np.newaxis, :]

# ==== BUILD MODEL ====
model = Sequential()
model.add(TimeDistributed(Dense(256, activation='relu'), input_shape=(1, X_train.shape[2])))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# ==== EMOTION PREDICTION ====
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    features = features[np.newaxis, np.newaxis, :]
    predicted_probabilities = model.predict(features)
    predicted_label_index = np.argmax(predicted_probabilities)
    predicted_emotion = label_encoder.classes_[predicted_label_index]
    emotion_mapping = {
        'YAF_angry': 'ANGRY', 'YAF_disgust': 'DISGUST', 'YAF_fear': 'FEAR',
        'YAF_happy': 'HAPPY', 'YAF_neutral': 'NEUTRAL', 'YAF_pleasant_surprised': 'SURPRISED',
        'YAF_sad': 'SAD', 'OAF_angry': 'ANGRY', 'OAF_disgust': 'DISGUST', 'OAF_Fear': 'FEAR',
        'OAF_happy': 'HAPPY', 'OAF_neutral': 'NEUTRAL', 'OAF_Pleasant_surprised': 'SURPRISED',
        'OAF_Sad': 'SAD',
    }
    return emotion_mapping.get(predicted_emotion, "UNKNOWN")

# ==== GUI APP ====
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Prediction App")
        self.root.geometry("600x600")
        self.root.configure(bg='lightyellow')

        self.emotion_to_emoji = {
            "HAPPY": "happy.png", "SAD": "sad.png", "ANGRY": "angry.png",
            "SURPRISED": "surprised.png", "NEUTRAL": "neutral.png",
            "FEAR": "fear.png", "DISGUST": "disgust.png"
        }

        self.emoji_image = None
        self.prediction_history = []

        self.show_home_page()

    def show_home_page(self):
        self.clear_window()
        tk.Label(self.root, text="🎤 Emotion Recognition", font=('Helvetica', 20, 'bold')).pack(pady=30)
        tk.Button(self.root, text="🎧 Upload Audio", command=self.show_audio_page, bg='orange', width=20).pack(pady=10)
        tk.Button(self.root, text="📜 Prediction History", command=self.show_history_page, bg='lightgreen', width=20).pack(pady=10)
        tk.Button(self.root, text="ℹ️ About", command=self.show_about_page, bg='lightblue', width=20).pack(pady=10)

    def show_audio_page(self):
        self.clear_window()
        tk.Label(self.root, text="Upload a .wav File", font=('Helvetica', 16)).pack(pady=10)

        def upload_audio():
            file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
            if file_path:
                emotion = predict_emotion(file_path)
                self.prediction_history.append((os.path.basename(file_path), emotion))
                emoji_file = os.path.join(EMOJI_PATH, self.emotion_to_emoji.get(emotion, 'neutral.png'))
                emoji = Image.open(emoji_file).resize((100, 100))
                self.emoji_image = ImageTk.PhotoImage(emoji)
                emoji_label.config(image=self.emoji_image)
                result_label.config(text=f"Predicted Emotion: {emotion}", font=('Helvetica', 14, 'bold'))

        tk.Button(self.root, text="Browse .wav File", command=upload_audio, bg='orange').pack(pady=20)
        result_label = tk.Label(self.root, text="Emotion will appear here", font=('Helvetica', 14))
        result_label.pack(pady=10)
        emoji_label = tk.Label(self.root)
        emoji_label.pack(pady=10)
        tk.Button(self.root, text="⬅ Back", command=self.show_home_page).pack(pady=20)

    def show_history_page(self):
        self.clear_window()
        tk.Label(self.root, text="Prediction History", font=('Helvetica', 16, 'bold')).pack(pady=20)
        if not self.prediction_history:
            tk.Label(self.root, text="No predictions yet!").pack()
        else:
            for i, (file, emotion) in enumerate(self.prediction_history, start=1):
                tk.Label(self.root, text=f"{i}. {file} → {emotion}").pack(anchor='w', padx=20)
        tk.Button(self.root, text="⬅ Back", command=self.show_home_page).pack(pady=20)

    def show_about_page(self):
        self.clear_window()
        tk.Label(self.root, text="About This App", font=('Helvetica', 16, 'bold')).pack(pady=20)
        about = (
            "This software recognizes emotions from voice using machine learning.\n\n"
            "All inputs must be .wav audio files. It uses the TESS dataset to train the model.\n\n"
            "Thanks to University of Toronto and the developers!"
        )
        tk.Label(self.root, text=about, wraplength=500, justify="left").pack(pady=10)
        tk.Button(self.root, text="⬅ Back", command=self.show_home_page).pack(pady=20)

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

# ==== RUN ====
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
