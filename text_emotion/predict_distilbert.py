"""CLI entry point for DistilBERT text inference."""

from text_emotion.distilbert_model import DistilBertTextPredictor


if __name__ == "__main__":
    predictor = DistilBertTextPredictor()
    while True:
        value = input("Enter text (or 'exit'): ").strip()
        if value.lower() == "exit":
            break
        print(predictor.predict(value))
