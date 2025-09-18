# depression-detection

This project is a machine learning-based chatbot that detects signs of depression through text analysis and facial emotion detection. It combines Natural Language Processing (NLP) with Computer Vision (CV) to provide an adaptive, intelligent response system.

📂 Project Structure
capstone/
│── chatbot/                # Django chatbot app
│── facial_analysis/        # Facial emotion detection (OpenCV + Keras)
│── models/                 # Trained ML models saved here
│── data/                   # Dataset folder (not included in repo)
│   ├── train.csv
│   ├── test.csv
│── requirements.txt
│── manage.py
│── README.md

📊 Dataset

The dataset used for training is not included in this repository (due to size restrictions).

🔹 Depression Detection (Text)

We used a dataset of text responses labeled with depressed / not depressed.

Place files inside the data/ folder:

data/train.csv
data/test.csv

🔹 Facial Emotion Detection

For emotion recognition, we used the FER-2013 dataset (available on Kaggle
).

Download the dataset and place it under:

data/fer2013.csv

⚙️ Installation

Clone the repo:

git clone https://github.com/abollavaram/depression-detection.git
cd capstone


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)


Install dependencies:

pip install -r requirements.txt

🚀 Usage
Run the Chatbot (Text-based Detection)
python manage.py runserver

Run Facial Emotion Detection (Webcam)
python facial_analysis/emotion_detector.py

📌 Features

Text-based depression detection using ML classifiers (Naive Bayes, Logistic Regression).

Facial emotion recognition with CNN trained on FER-2013 dataset.

Django chatbot interface for real-time interaction.

Combines NLP + CV for adaptive learning.
