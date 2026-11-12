
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from project_config import MELD_SPLITS, TEXT_MODEL_PATH, meld_annotation_path, validate_meld_dataset
from text_emotion.model_io import write_model_metadata
from text_emotion.preprocessing import preprocess_text

# Project paths
ROOT = Path(__file__).resolve().parent.parent


def load_split(split: str) -> pd.DataFrame:
    frame = pd.read_csv(meld_annotation_path(split))
    frame = frame.dropna(subset=["Utterance", "Emotion"]).copy()
    frame["Utterance"] = frame["Utterance"].map(preprocess_text)
    frame["Emotion"] = frame["Emotion"].astype(str).str.strip().str.lower()
    return frame[frame["Utterance"] != ""].copy()


from sklearn.pipeline import FeatureUnion

model = Pipeline([
    ("features", FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=40000,
            sublinear_tf=True
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=30000,
            sublinear_tf=True,
            min_df=2
        ))
    ])),
    ("classifier", LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=1.0,
        random_state=42
    ))
])
def main() -> None:
    validate_meld_dataset()
    frames = {split: load_split(split) for split in MELD_SPLITS}
    X_train, y_train = frames["train"]["Utterance"], frames["train"]["Emotion"]
    X_dev, y_dev = frames["dev"]["Utterance"], frames["dev"]["Emotion"]
    X_test, y_test = frames["test"]["Utterance"], frames["test"]["Emotion"]

    print("Training text emotion model on MELD...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_dev)}")
    print(f"Test samples: {len(X_test)}")
    model.fit(X_train, y_train)

    dev_predictions = model.predict(X_dev)
    print(f"Validation Macro F1: {f1_score(y_dev, dev_predictions, average='macro', zero_division=0):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_dev, dev_predictions):.4f}")
    print(classification_report(y_dev, dev_predictions, zero_division=0))

    test_predictions = model.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
    print(classification_report(y_test, test_predictions, zero_division=0))

    model_path = TEXT_MODEL_PATH
    import joblib
    joblib.dump(model, model_path)
    write_model_metadata(model, model_path, {split: len(frame) for split, frame in frames.items()})
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
