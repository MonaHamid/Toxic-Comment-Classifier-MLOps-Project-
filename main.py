import json
import os
from typing import Dict, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()
MODEL_DIR = "models/toxic_bert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

with open(os.path.join(MODEL_DIR, "labels.json")) as f:
    LABELS = json.load(f)


class PredictIn(BaseModel):
    texts: List[str]


class PredictOut(BaseModel):
    predictions: List[Dict[str, float]]


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    enc = tokenizer(
        payload.texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256,
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()
    result = [{LABELS[i]: float(p[i]) for i in range(len(LABELS))} for p in probs]
    return {"predictions": result}
