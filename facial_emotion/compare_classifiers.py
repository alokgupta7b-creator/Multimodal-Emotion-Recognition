
"""Compare facial classifiers using the MELD development split."""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from facial_emotion.train_classifier import load_split


def main():
    X_train, y_train, _ = load_split("train")
    X_dev, y_dev, _ = load_split("dev")

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            C=0.1,
            max_iter=3000,
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "RBF SVM": SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            probability=False,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining: {name}")

        if name == "Random Forest" or name == "Extra Trees":
            model.fit(X_train, y_train)
            predictions = model.predict(X_dev)
        else:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_dev_scaled)

        accuracy = accuracy_score(y_dev, predictions)
        macro_f1 = f1_score(
            y_dev,
            predictions,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(
            y_dev,
            predictions,
            average="weighted",
            zero_division=0,
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "Macro F1", ascending=False
    )

    print("\n========== DEVELOPMENT COMPARISON ==========")
    print(results_df.to_string(index=False))

    output_path = (
        r"D:\MELD_Facial_Output\models\classifier_comparison_dev.csv"
    )
    results_df.to_csv(output_path, index=False)
    print("\nSaved comparison:", output_path)


if __name__ == "__main__":
    main()
