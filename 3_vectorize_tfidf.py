import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv("depression_clean_processed.csv")

# Separate inputs and labels
X = df['cleaned_message']
y = df['label']

# TF-IDF vectorizer setup
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')

# Convert text to numerical features
X_vect = vectorizer.fit_transform(X)

# Split into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.25, random_state=42)

print("TF-IDF transformation complete.")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Optional: Save for next step
import joblib
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump((X_train, y_train, X_test, y_test), "vectorized_data.pkl")
