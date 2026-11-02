
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "MELD"
MODEL_PATH = ROOT / "text_emotion_model.joblib"
OUTPUT_DIR = ROOT / "features" / "fusion"

EMOTIONS = [
    "anger", "disgust", "fear", "joy",
    "neutral", "sadness", "surprise"
]


def export_split(split):
    annotation_path = DATA_DIR / "annotations" / f"{split}_sent_emo.csv"
    manifest_path = DATA_DIR / "manifests" / f"{split}_manifest.csv"

    annotations = pd.read_csv(annotation_path)
    manifest = pd.read_csv(manifest_path)

    # Standardize annotation column names
    annotations = annotations.rename(columns={
        "Dialogue_ID": "dialogue_id",
        "Utterance_ID": "utterance_id"
    })

    keys = ["dialogue_id", "utterance_id"]

    # Convert IDs to integers
    for df in (annotations, manifest):
        for key in keys:
            df[key] = pd.to_numeric(
                df[key], errors="raise"
            ).astype(int)

    # Check duplicate IDs
    if annotations.duplicated(keys).any():
        raise ValueError(f"Duplicate annotation IDs in {split}")

    if manifest.duplicated(keys).any():
        raise ValueError(f"Duplicate manifest IDs in {split}")

    # Match text with manifest sample IDs
    data = manifest[["sample_id"] + keys].merge(
        annotations[keys + ["Utterance"]],
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False
    )

    if data["Utterance"].isna().any():
        raise ValueError(f"Some manifest IDs have no matching text in {split}")

    # Match train.py preprocessing: strip whitespace only
    data["Utterance"] = data["Utterance"].astype(str).str.strip()
    data = data[data["Utterance"] != ""].copy()

    # Use raw text; do not call preprocess_text()
    probabilities = model.predict_proba(data["Utterance"].tolist())

    result = pd.DataFrame(probabilities, columns=model.classes_)
    result.insert(0, "sample_id", data["sample_id"].values)
    result = result[["sample_id"] + EMOTIONS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"text_{split}_probabilities.csv"
    result.to_csv(output_path, index=False)

    # Validate output
    assert result["sample_id"].is_unique
    assert result.columns.tolist() == ["sample_id"] + EMOTIONS
    assert result[EMOTIONS].notna().all().all()
    assert ((result[EMOTIONS] >= 0) & (result[EMOTIONS] <= 1)).all().all()
    assert (result[EMOTIONS].sum(axis=1).sub(1).abs() < 1e-5).all()

    print(f"\n{split.upper()} probabilities saved: {output_path}")
    print(f"Rows: {len(result)}")
    print(result.head(3).to_string(index=False))


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

if list(model.classes_) != EMOTIONS:
    raise ValueError(f"Unexpected model classes: {list(model.classes_)}")

classifier = model.named_steps["classifier"]

print("Model: Logistic Regression")
print("class_weight:", classifier.class_weight)
print("C:", classifier.C)

if classifier.class_weight != "balanced" or classifier.C != 1.0:
    raise ValueError("Model configuration does not match required settings.")

export_split("dev")
export_split("test")

print("\nBoth probability files generated and validated successfully.")
