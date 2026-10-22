
import joblib
from text_emotion.preprocessing import preprocess_text

# Load the trained model
model = joblib.load("text_emotion_model.joblib")

# Emotion labels from the dataset
emotion_labels = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

# Test your own sentences
while True:
    text = input("\nEnter a sentence (or type 'exit' to stop): ")

    if text.lower() == "exit":
        break

    cleaned_text = preprocess_text(text)
    prediction = model.predict([cleaned_text])[0]

    print("Predicted Emotion:", emotion_labels.get(
        int(prediction), "Unknown"
    ))
