# scripts/_common_multi.py
import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_for_tfidf
from src.constants import LABELS

TRAIN_PATH = os.getenv("TRAIN_CSV", "data/raw/train.csv")


def load_xy_multi():
    df = pd.read_csv(TRAIN_PATH).dropna(subset=["comment_text"])

    # sanity check: labels must exist in train.csv
    missing = [c for c in LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in {TRAIN_PATH}: {missing}")

    # simple stratification by 'toxic' to preserve class balance roughly
    stratify_col = df["toxic"] if "toxic" in df.columns else None

    train, valid = train_test_split(
        df, test_size=0.2, random_state=42, stratify=stratify_col
    )

    Xtr = train["comment_text"].apply(preprocess_for_tfidf)
    Xte = valid["comment_text"].apply(preprocess_for_tfidf)
    Ytr = train[LABELS].astype(int).values
    Yte = valid[LABELS].astype(int).values
    return Xtr, Ytr, Xte, Yte


def mlflow_setup():
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("toxic_comment_multilabel")
