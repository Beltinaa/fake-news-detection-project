import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text