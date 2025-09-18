import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# Load vectorized data
X_train, y_train, X_test, y_test = joblib.load("vectorized_data.pkl")

# Show original class distribution
print("Before Oversampling - Class distribution in y_train:")
print(np.bincount(y_train))

# Apply RandomOverSampler instead of SMOTE (works better for small datasets)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Show new class distribution
print("\nAfter Oversampling - Class distribution in y_resampled:")
print(np.bincount(y_resampled))

# -------------------------
# Train Naive Bayes
# -------------------------
print("\n--- Naive Bayes Classifier ---")
nb_model = MultinomialNB()
nb_model.fit(X_resampled, y_resampled)
nb_pred = nb_model.predict(X_test)

print(classification_report(y_test, nb_pred))

# -------------------------
# Train Logistic Regression
# -------------------------
print("\n--- Logistic Regression Classifier ---")
lr_model = LogisticRegression(solver='lbfgs')
lr_model.fit(X_resampled, y_resampled)
lr_pred = lr_model.predict(X_test)

print(classification_report(y_test, lr_pred))

# -------------------------
# Save models for chatbot use
# -------------------------
joblib.dump(nb_model, "naive_bayes_model.pkl")
joblib.dump(lr_model, "logistic_regression_model.pkl")

print("\nModels saved successfully!")
