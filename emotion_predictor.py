import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Get base path dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models using full path
face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"))
emotion_model = load_model(os.path.join(BASE_DIR, "emotion_model.h5"))

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_face_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Face not detected"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        prediction = emotion_model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        return emotion

    return "Face not detected"
