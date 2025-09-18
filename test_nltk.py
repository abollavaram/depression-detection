from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Input text to test
text = "I am happily running through the park with my friends."

# Tokenize into words
words = word_tokenize(text)
print("Tokenized:", words)

# Remove stopwords (like 'the', 'am', 'with')
filtered = [word for word in words if word.lower() not in stopwords.words('english')]
print("After removing stopwords:", filtered)

# Lemmatize (e.g., 'running' → 'run')
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
print("Lemmatized:", lemmatized)
