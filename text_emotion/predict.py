
from pathlib import Path

import joblib
from text_emotion.preprocessing import preprocess_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "text_emotion_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run python -m text_emotion.train first."
    )

model = joblib.load(MODEL_PATH)

print("MELD Text Emotion Predictor")
print("Supported emotions: anger, disgust, fear, joy, neutral, sadness, surprise")
print("Type 'exit' to quit.\n")

while True:
    text = input("Enter a sentence: ").strip()

    if text.lower() == "exit":
        print("Goodbye!")
        break

    if not text:
        print("Please enter a sentence.")
        continue

    cleaned_text = preprocess_text(text)
    prediction = model.predict([cleaned_text])[0]

    print(f"Predicted emotion: {prediction}\n")
