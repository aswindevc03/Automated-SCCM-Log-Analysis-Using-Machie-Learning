import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load CSV
df = pd.read_csv("sccm_error_codes.csv")


# Combine text for ML
df["text"] = df["ErrorCode"].astype(str) + " " + df["Issue"] + " " + df["Fix"]

# Target column
y = df["Issue"]  # or df["Fix"] depending on what you want to predict


# Create and train pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(df["text"], y)

# Save model
with open("sccm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("ML model trained and saved as sccm_model.pkl")
