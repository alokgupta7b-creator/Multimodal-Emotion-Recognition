
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from project_config import TEXT_MODEL_PATH, meld_annotation_path
from text_emotion.model_io import load_text_model
from text_emotion.preprocessing import preprocess_text


def main():
    test_df = pd.read_csv(meld_annotation_path("test"))
    test_df = test_df.dropna(subset=["Utterance", "Emotion"]).copy()
    test_df["Utterance"] = test_df["Utterance"].map(preprocess_text)
    test_df["Emotion"] = test_df["Emotion"].astype(str).str.strip().str.lower()
    test_df = test_df[test_df["Utterance"] != ""]

    model = load_text_model(TEXT_MODEL_PATH)

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
