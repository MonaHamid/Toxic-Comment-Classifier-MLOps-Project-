import os

import mlflow.pyfunc
import pandas as pd

os.environ["MLFLOW_TRACKING_URI"] = os.getenv(
    "MLFLOW_TRACKING_URI", "http://localhost:5050"
)

model = mlflow.pyfunc.load_model("models:/ToxicCommentClassifierMulti/Production")
preds = model.predict(pd.DataFrame({"text": ["You are awesome!", "I hate you."]}))
print(preds)  # (N, 6) array of 0/1
