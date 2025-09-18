import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from django.shortcuts import render
import joblib
from facial_analysis.emotion_predictor import detect_face_emotion


# Load the trained model and vectorizer
model = joblib.load(os.path.join(os.path.dirname(__file__), "naive_bayes_model.pkl"))
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl"))


def chatbot(request):
    result = ''
    user_input = ''
    face_emotion = ''

    if request.method == 'POST':
        # Get message from form
        user_input = request.POST.get('message')

        # Text-based depression detection
        vect_text = vectorizer.transform([user_input])
        prediction = model.predict(vect_text)[0]
        result = 'Depressed 😟' if prediction == 1 else 'Not Depressed 😊'

        # Facial emotion prediction from webcam
        face_emotion = detect_face_emotion()

    # Pass all results to the template
    return render(request, 'chat/index.html', {
        'result': result,
        'user_input': user_input,
        'face_emotion': face_emotion
    })
