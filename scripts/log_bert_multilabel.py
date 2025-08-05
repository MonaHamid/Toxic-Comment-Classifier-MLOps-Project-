# scripts/log_bert_multilabel.py
import os

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch

from src.constants import LABELS
from src.preprocessing import preprocess_for_bert

# Tunables (override via env)
MAX_LEN = int(os.getenv("MAX_LEN", "256"))  # 256 faster than 512 on CPU
BATCH = int(os.getenv("BATCH_SIZE", "32"))
THRESH = float(os.getenv("BERT_THRESHOLD", "0.5"))
TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
N_VAL = os.getenv("N_VAL")  # e.g., "300" for quick run; None = full val split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    df = pd.read_csv("data/raw/train.csv").dropna(subset=["comment_text"])
    stratify_col = df["toxic"] if "toxic" in df.columns else None
    _, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=stratify_col
    )

    X_val = val_df["comment_text"].apply(preprocess_for_bert).tolist()
    Y_true = val_df[LABELS].astype(int).values

    if N_VAL:
        n = int(N_VAL)
        X_val = X_val[:n]
        Y_true = Y_true[:n]

    tokenizer = AutoTokenizer.from_pretrained("saved_model", model_max_length=MAX_LEN)
    model = AutoModelForSequenceClassification.from_pretrained("saved_model")
    model.eval().to("cpu")

    probs_all = []
    with torch.inference_mode():
        for i in range(0, len(X_val), BATCH):
            if i % (BATCH * 10) == 0:
                print(f"...{i}/{len(X_val)}")
            batch = X_val[i : i + BATCH]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            for k in enc:
                enc[k] = enc[k].to("cpu")

            outputs = model(**enc)
            logits = outputs.logits.detach().cpu().numpy()
            probs = sigmoid(logits)
            probs_all.append(probs)

    S = np.vstack(probs_all)
    Y_pred = (S >= THRESH).astype(int)

    f1_micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    f1_per = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    per_label = {f"f1_{lbl}": float(v) for lbl, v in zip(LABELS, f1_per)}

    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment("toxic_comment_multilabel")

    class HFMultilabelWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.max_len = int(os.getenv("MAX_LEN", "256"))
            self.thresh = float(os.getenv("BERT_THRESHOLD", "0.5"))
            self.tokenizer = AutoTokenizer.from_pretrained(
                "saved_model", model_max_length=self.max_len
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "saved_model"
            )
            self.model.eval().to("cpu")

        def predict(self, context, model_input):
            texts = model_input["text"].tolist()
            preds = []
            with torch.inference_mode():
                for i in range(0, len(texts), 32):
                    batch = texts[i : i + 32]
                    enc = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_len,
                        return_tensors="pt",
                    )
                    outputs = self.model(**enc)
                    logits = outputs.logits.detach().numpy()
                    probs = 1 / (1 + np.exp(-logits))
                    preds.append((probs >= self.thresh).astype(int))
            return np.vstack(preds)

    with mlflow.start_run(run_name="bert_saved_model_multilabel_fast") as run:
        mlflow.log_param("model", "bert_saved_model_multilabel_fast")
        mlflow.log_param("threshold", THRESH)
        mlflow.log_param("max_len", MAX_LEN)
        mlflow.log_param("batch_size", BATCH)
        if N_VAL:
            mlflow.log_param("n_val", int(N_VAL))

        mlflow.log_metric("f1_micro", f1_micro)
        mlflow.log_metric("f1_macro", f1_macro)
        for k, v in per_label.items():
            mlflow.log_metric(k, v)

        signature = mlflow.models.infer_signature(
            model_input=pd.DataFrame({"text": ["sample"]}),
            model_output=np.zeros((1, len(LABELS)), dtype=int),
        )
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=HFMultilabelWrapper(),
            signature=signature,
            input_example=pd.DataFrame({"text": ["you are awesome!"]}),
        )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "ToxicCommentClassifierMulti")

    print(
        f"✅ Logged BERT (fast, CPU). f1_micro={f1_micro:.4f}  f1_macro={f1_macro:.4f}"
    )


if __name__ == "__main__":
    main()
