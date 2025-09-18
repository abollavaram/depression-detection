import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load data
df = pd.read_csv("depression_clean.csv")

# Set up tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Cleaned data will go here
cleaned_messages = []

for msg in df['message']:
    # Remove punctuation, numbers, and symbols
    msg = re.sub('[^a-zA-Z]', ' ', msg)
    
    # Convert to lowercase
    msg = msg.lower()
    
    # Tokenize into words
    words = nltk.word_tokenize(msg)
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize each word
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Rejoin into cleaned sentence
    cleaned_msg = ' '.join(words)
    cleaned_messages.append(cleaned_msg)

# Create new DataFrame with cleaned messages
df_cleaned = pd.DataFrame({'cleaned_message': cleaned_messages, 'label': df['label']})

# Show first few rows
print("Sample cleaned messages:\n")
print(df_cleaned.head())

# Optional: Save to CSV (if you want)
df_cleaned.to_csv("depression_clean_processed.csv", index=False)
