
import pandas as pd
from project_config import MELD_MANIFESTS, MELD_ROOT, TEXT_MODEL_PATH, REPOSITORY_ROOT
from text_emotion.model_io import load_text_model
from text_emotion.preprocessing import preprocess_text

OUTPUT_DIR = REPOSITORY_ROOT / "features" / "fusion"

EMOTIONS = [
    "anger", "disgust", "fear", "joy",
    "neutral", "sadness", "surprise"
]


def export_split(split):
    annotation_path = MELD_ROOT / "annotations" / f"{split}_sent_emo.csv"
    manifest_path = MELD_MANIFESTS / f"{split}_manifest.csv"

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

    data["Utterance"] = data["Utterance"].map(preprocess_text)
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


model = load_text_model(TEXT_MODEL_PATH)

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
