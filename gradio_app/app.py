# gradio_app/app.py
import os, pandas as pd, gradio as gr, mlflow.pyfunc
from src.constants import LABELS

os.environ["MLFLOW_TRACKING_URI"] = os.getenv(
    "MLFLOW_TRACKING_URI", "http://localhost:5050"
)
MODEL_URI = "models:/ToxicCommentClassifierMulti/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)


def predict_fn(text: str):
    y = model.predict(pd.DataFrame({"text": [text]}))[0]  # array of 0/1, len=6
    return {lbl: int(v) for lbl, v in zip(LABELS, y)}


demo = gr.Interface(
    fn=predict_fn,
    inputs=gr.Textbox(lines=5, label="Comment"),
    outputs=gr.Label(num_top_classes=6, label="Predicted labels (1=yes)"),
    title="Toxic Comment Classifier (Multi-label via MLflow Production)",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
