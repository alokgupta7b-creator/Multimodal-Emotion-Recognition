
import re
import unicodedata


def preprocess_text(text: str) -> str:
    """
    Clean text before emotion recognition.
    Preserves meaningful words such as 'not', 'no', and 'never'.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.\w+\b",
        " ",
        text
    )

    # Convert text to lowercase
    text = text.lower()

    # Keep letters, numbers, and basic punctuation
    text = re.sub(r"[^\w\s!?']", " ", text)

    # Remove unnecessary spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_texts(texts):
    """Clean multiple text samples."""

    if isinstance(texts, str):
        raise TypeError("Expected a list of text samples.")

    return [preprocess_text(text) for text in texts]


if __name__ == "__main__":
    samples = [
        "I am SO HAPPY!!!",
        "I am NOT happy.",
        "Visit https://example.com today!",
        "I feel sad "
    ]

    for sample in samples:
        print("Original:", sample)
        print("Cleaned: ", preprocess_text(sample))
        print("-" * 40)
