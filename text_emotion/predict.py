
from pathlib import Path

from project_config import TEXT_MODEL_PATH
from text_emotion.model_io import load_text_model
from text_emotion.preprocessing import preprocess_text

ROOT = Path(__file__).resolve().parent.parent
def predict_text(text: str) -> dict[str, object]:
    model = load_text_model(TEXT_MODEL_PATH)
    cleaned_text = preprocess_text(text)
    if not cleaned_text:
        raise ValueError("Text input cannot be empty after preprocessing.")
    probabilities = model.predict_proba([cleaned_text])[0]
    return {
        "predicted_label": str(model.classes_[probabilities.argmax()]),
        "confidence": float(probabilities.max()),
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        },
    }


def main() -> None:
    print("MELD Text Emotion Predictor")
    print("Supported emotions: anger, disgust, fear, joy, neutral, sadness, surprise")
    print("Type 'exit' to quit.\n")
    while True:
        text = input("Enter a sentence: ").strip()
        if text.lower() == "exit":
            print("Goodbye!")
            return
        if not text:
            print("Please enter a sentence.")
            continue
        print(f"Prediction: {predict_text(text)}\n")


if __name__ == "__main__":
    main()
