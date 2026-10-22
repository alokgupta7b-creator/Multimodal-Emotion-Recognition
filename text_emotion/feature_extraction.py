
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer():
    """
    Convert text into numerical TF-IDF features.
    Uses single words and two-word phrases.
    """

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    return vectorizer


if __name__ == "__main__":
    texts = [
        "I am very happy",
        "I am very sad",
        "I am not happy",
        "Today is a happy day"
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1
    )

    features = vectorizer.fit_transform(texts)

    print("Feature matrix shape:", features.shape)
    print("\nExtracted features:")
    print(vectorizer.get_feature_names_out())
    print("\nTF-IDF matrix:")
    print(features.toarray())
