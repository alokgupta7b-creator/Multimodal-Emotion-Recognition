
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

from text_emotion.preprocessing import preprocess_text

# Load the emotion dataset
dataset = load_dataset("dair-ai/emotion")

# Prepare training and testing data
X_train = [preprocess_text(text) for text in dataset["train"]["text"]]
y_train = dataset["train"]["label"]

X_test = [preprocess_text(text) for text in dataset["test"]["text"]]
y_test = dataset["test"]["label"]

# Create and train the model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training the text emotion model...")
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save the trained model
joblib.dump(model, "text_emotion_model.joblib")
print("\nModel saved as text_emotion_model.joblib")
