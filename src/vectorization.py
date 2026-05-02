from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer():
    return TfidfVectorizer(
        stop_words="english",
        max_df=0.7
    )