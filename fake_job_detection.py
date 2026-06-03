# Fake Job Detection Using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("fake_job_postings.csv")

# Select required columns
df = df[['title', 'description', 'fraudulent']]

# Handle missing values
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')

# Combine text columns
df['text'] = df['title'] + " " + df['description']

# Features and Target
X = df['text']
y = df['fraudulent']

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Custom Prediction
while True:
    job_text = input("\nEnter Job Description (or 'exit'): ")

    if job_text.lower() == "exit":
        break

    job_vector = vectorizer.transform([job_text])
    prediction = model.predict(job_vector)

    if prediction[0] == 1:
        print("⚠️ Fake Job Posting")
    else:
        print("✅ Real Job Posting")
