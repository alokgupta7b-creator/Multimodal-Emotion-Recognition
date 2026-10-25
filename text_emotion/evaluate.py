
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "meld_dataset"
MODEL_PATH = ROOT / "text_emotion_model.joblib"


def find_csv(filename):
    matches = list(DATA_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Cannot find {filename} inside {DATA_DIR}"
        )
    return matches[0]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python -m text_emotion.train first."
        )

    test_df = pd.read_csv(find_csv("test_sent_emo.csv"))
    test_df = test_df.dropna(subset=["Utterance", "Emotion"]).copy()
    test_df["Utterance"] = test_df["Utterance"].astype(str).str.strip()
    test_df["Emotion"] = test_df["Emotion"].astype(str).str.strip().str.lower()
    test_df = test_df[test_df["Utterance"] != ""]

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(test_df["Utterance"])

    print("MELD Text Emotion Model Evaluation")
    print(f"Test samples: {len(test_df)}")
    print(f"Accuracy: {accuracy_score(test_df['Emotion'], predictions):.4f}")

    print("\nClassification Report:")
    print(classification_report(
        test_df["Emotion"],
        predictions,
        zero_division=0
    ))

    labels = sorted(test_df["Emotion"].unique())
    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(labels)
    print(confusion_matrix(
        test_df["Emotion"],
        predictions,
        labels=labels
    ))


if __name__ == "__main__":
    main()
