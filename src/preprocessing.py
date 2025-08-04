import re
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords", quiet=True)

STOPWORDS = set(nltk_stopwords.words("english"))
# keep "not" to preserve meaning (optional but helpful)
STOPWORDS.discard("not")

STEMMER = SnowballStemmer("english")


def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in STOPWORDS)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", " i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stemming(text: str) -> str:
    return " ".join(STEMMER.stem(w) for w in text.split())


# Full pipeline for TF-IDF baselines
def preprocess_for_tfidf(text: str) -> str:
    text = clean_text(text)
    text = remove_stopwords(text)
    text = stemming(text)  # if recall drops, comment this out
    return text


# Minimal cleanup for BERT
def preprocess_for_bert(text: str) -> str:
    return str(text).strip()
