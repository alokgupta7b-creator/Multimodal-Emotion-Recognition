
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from project_config import meld_annotation_path
from text_emotion.preprocessing import preprocess_text


train_df = pd.read_csv(meld_annotation_path("train"))
dev_df = pd.read_csv(meld_annotation_path("dev"))


def clean(df):
    df = df.dropna(subset=["Utterance", "Emotion"]).copy()
    df["Utterance"] = df["Utterance"].map(preprocess_text)
    df["Emotion"] = df["Emotion"].astype(str).str.strip().str.lower()
    return df[df["Utterance"] != ""]


train_df = clean(train_df)
dev_df = clean(dev_df)

X_train, y_train = train_df["Utterance"], train_df["Emotion"]
X_dev, y_dev = dev_df["Utterance"], dev_df["Emotion"]

for c_value in [1.0, 2.0, 3.0, 4.0]:
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
            C=c_value,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)

    macro_f1 = f1_score(
        y_dev, predictions, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_dev, predictions)

    print(
        f"C={c_value:.1f} | "
        f"Validation Accuracy={accuracy:.4f} | "
        f"Macro F1={macro_f1:.4f}"
    )
