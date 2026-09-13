import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load cleaned data
df = pd.read_csv("results_full.csv")

# TF-IDF + Logistic Regression baseline
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["intent_pred"]  # using DistilBERT labels for comparison

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

y_pred = clf.predict(X)

print("Baseline Evaluation:")
print(classification_report(y, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
