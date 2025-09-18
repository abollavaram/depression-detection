import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
df = pd.read_csv("depression_clean.csv")  # Make sure this file is present

# Preprocess text
lemmatizer = WordNetLemmatizer()
corpus = []

for text in df['message']:
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [lemmatizer.lemmatize(word) for word in text]
    corpus.append(" ".join(text))

# Vectorize
vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus)
y = df['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
