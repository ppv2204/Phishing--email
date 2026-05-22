import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt


# Step 2: Load Dataset

df = pd.read_csv('/content/phishing_email.csv')


# Step 3: Display Dataset

print("First 5 Rows of Dataset:")
print(df.head())

print("\nDataset Columns:")
print(df.columns)


# Step 4: Convert Labels

df['Email Type'] = df['Email Type'].map({
    'Phishing Email': 1,
    'Safe Email': 0
})


# Step 5: Define Features and Labels

X = df['Email Text']
y = df['Email Type']


# Step 6: Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 7: Convert Text to Numerical Features

vectorizer = TfidfVectorizer()

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)


# Step 8: Train Model

model = MultinomialNB()

model.fit(X_train_features, y_train)


# Step 9: Predictions

y_pred = model.predict(X_test_features)


# Step 10: Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)


# Step 11: Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 12: Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Step 13: Plot Confusion Matrix

plt.figure(figsize=(6,4))

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Safe', 'Phishing'],
            yticklabels=['Safe', 'Phishing'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# Step 14: Test with Custom Email

sample_email = [
    "Congratulations! You won a free iPhone. Click the link now to claim your reward."
]

sample_vector = vectorizer.transform(sample_email)

prediction = model.predict(sample_vector)

print("\nCustom Email Prediction:")

if prediction[0] == 1:
    print("Phishing Email")
else:
    print("Safe Email")