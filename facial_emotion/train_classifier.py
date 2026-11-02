
"""Train and evaluate the MELD facial emotion classifier."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from facial_emotion.config import EMOTION_LABELS


ROOT = Path(r"D:\MELD_Facial_Output")
MODEL_DIR = ROOT / "models"
FUSION_DIR = ROOT / "fusion"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FUSION_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split):
    """Load all successful feature vectors for one split."""

    folder = ROOT / split
    files = sorted(
        file for file in folder.glob("batch_*.npz")
        if file.stem.removeprefix("batch_").isdigit()
    )

    features = []
    labels = []
    sample_ids = []

    for file in files:
        with np.load(file, allow_pickle=False) as data:
            features.append(data["features"])
            labels.extend(data["labels"].tolist())
            sample_ids.extend(data["sample_ids"].tolist())

    if not features:
        raise ValueError(f"No features found for {split}")

    X = np.concatenate(features, axis=0)
    y = np.asarray(labels, dtype=str)
    ids = np.asarray(sample_ids, dtype=str)

    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"Duplicate sample IDs in {split}")

    if not np.isfinite(X).all():
        raise ValueError(f"Invalid feature values in {split}")

    unknown = set(y) - set(EMOTION_LABELS)
    if unknown:
        raise ValueError(f"Unknown emotion labels: {unknown}")

    print(f"{split}: {X.shape[0]} samples, {X.shape[1]} features")

    return X, y, ids


def evaluate(name, y_true, probabilities, classes):
    """Print classification metrics."""

    predictions = classes[np.argmax(probabilities, axis=1)]

    print(f"\n{name} RESULTS")
    print("Accuracy:", round(accuracy_score(y_true, predictions), 4))
    print("Macro F1:", round(
        f1_score(y_true, predictions, average="macro", zero_division=0),
        4,
    ))
    print("Weighted F1:", round(
        f1_score(y_true, predictions, average="weighted", zero_division=0),
        4,
    ))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        predictions,
        labels=EMOTION_LABELS,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_true,
        predictions,
        labels=EMOTION_LABELS,
    ))


def save_probabilities(path, ids, probabilities, classes):
    """Save probabilities using the multimodal fusion schema."""

    output = pd.DataFrame(probabilities, columns=classes)
    output.insert(0, "sample_id", ids)

    output.to_csv(path, index=False)
    print("Saved:", path)


def main():
    X_train, y_train, _ = load_split("train")
    X_dev, y_dev, dev_ids = load_split("dev")
    X_test, y_test, test_ids = load_split("test")

    print("\nTraining feature scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Logistic Regression...")
    classifier = LogisticRegression(
        class_weight="balanced",
        C=0.1,
        max_iter=3000,
        solver="lbfgs",
        random_state=42,
    )

    classifier.fit(X_train_scaled, y_train)

    classes = classifier.classes_

    if tuple(classes) != tuple(EMOTION_LABELS):
        raise ValueError(
            f"Unexpected class order: {classes}"
        )

    print("\nEvaluating development split...")
    dev_probabilities = classifier.predict_proba(X_dev_scaled)
    evaluate("DEVELOPMENT", y_dev, dev_probabilities, classes)

    print("\nEvaluating test split...")
    test_probabilities = classifier.predict_proba(X_test_scaled)
    evaluate("TEST", y_test, test_probabilities, classes)

    model_path = MODEL_DIR / "facial_emotion_logistic_regression.joblib"

    joblib.dump(
        {
            "scaler": scaler,
            "classifier": classifier,
            "emotion_labels": list(classes),
            "feature_count": X_train.shape[1],
        },
        model_path,
    )

    print("\nSaved model:", model_path)

    save_probabilities(
        FUSION_DIR / "visual_dev_probabilities.csv",
        dev_ids,
        dev_probabilities,
        classes,
    )

    save_probabilities(
        FUSION_DIR / "visual_test_probabilities.csv",
        test_ids,
        test_probabilities,
        classes,
    )

    print("\nFacial classifier training completed.")


if __name__ == "__main__":
    main()
