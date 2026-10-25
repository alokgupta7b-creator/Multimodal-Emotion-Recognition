
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "meld_dataset"

# Find MELD CSV files, even if they are inside a subfolder
def find_csv(filename):
    matches = list(DATA_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {DATA_DIR}"
        )
    return matches[0]


train_df = pd.read_csv(find_csv("train_sent_emo.csv"))
dev_df = pd.read_csv(find_csv("dev_sent_emo.csv"))
test_df = pd.read_csv(find_csv("test_sent_emo.csv"))

# Remove rows without text or labels
for df in (train_df, dev_df, test_df):
    df.dropna(subset=["Utterance", "Emotion"], inplace=True)
    df["Utterance"] = df["Utterance"].astype(str).str.strip()
    df["Emotion"] = df["Emotion"].astype(str).str.strip().str.lower()
    df.drop(df[df["Utterance"] == ""].index, inplace=True)

# Train only on the official training split
X_train = train_df["Utterance"]
y_train = train_df["Emotion"]

# Evaluate on validation and test splits
X_dev, y_dev = dev_df["Utterance"], dev_df["Emotion"]
X_test, y_test = test_df["Utterance"], test_df["Emotion"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("Training text emotion model on MELD...")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_dev)}")
print(f"Test samples: {len(X_test)}")

model.fit(X_train, y_train)

# Validation evaluation
dev_predictions = model.predict(X_dev)
print("\n--- MELD Validation Results ---")
print(f"Accuracy: {accuracy_score(y_dev, dev_predictions):.4f}")
print(classification_report(y_dev, dev_predictions, zero_division=0))

# Final test evaluation
test_predictions = model.predict(X_test)
print("\n--- MELD Test Results ---")
print(f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
print(classification_report(y_test, test_predictions, zero_division=0))

# Save the trained pipeline
model_path = ROOT / "text_emotion_model.joblib"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")
