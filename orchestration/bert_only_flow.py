from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from sklearn.model_selection import train_test_split
import os, subprocess, pandas as pd

DATA_TRAIN = "data/raw/train.csv"


def run(cmd, extra_env=None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(cmd, env=env).wait()


@task(name="ingest_data")
def ingest_data() -> dict:
    log = get_run_logger()
    if not os.path.exists(DATA_TRAIN):
        raise FileNotFoundError(f"{DATA_TRAIN} not found")
    df = pd.read_csv(DATA_TRAIN)
    if "comment_text" not in df.columns:
        raise ValueError("Expected 'comment_text' column")
    log.info(f"Loaded {DATA_TRAIN}: shape={df.shape}")
    return {"rows": len(df)}


@task(name="split_data")
def split_data() -> dict:
    # quick split (no files written)
    df = pd.read_csv(DATA_TRAIN).dropna(subset=["comment_text"])
    strat = df["toxic"] if "toxic" in df.columns else None
    _, _val = train_test_split(df, test_size=0.2, random_state=42, stratify=strat)
    return {"val_rows": len(_val)}


@task(name="log_bert")
def log_bert() -> int:
    # runs your fast CPU logger that logs + registers in MLflow
    return run(
        ["python", "-m", "scripts.log_bert_multilabel_fast_proba"],
        {
            "N_VAL": "300",
            "MAX_LEN": "96",
            "BATCH_SIZE": "64",
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI", "http://localhost:5050"
            ),
        },
    )


@task(name="evaluate")
def evaluate() -> str:
    get_run_logger().info("Evaluation complete – see MLflow for metrics.")
    return "ok"


@task(name="register_production")
def register_production():
    from mlflow.tracking import MlflowClient

    log = get_run_logger()
    client = MlflowClient(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    )
    name = "ToxicCommentClassifierMulti"
    infos = client.search_model_versions(f"name='{name}'")
    if not infos:
        log.warning("No versions found to promote.")
        return "none"
    latest = max(infos, key=lambda i: int(i.version))
    client.transition_model_version_stage(
        name, latest.version, "Production", archive_existing_versions=True
    )
    log.info(f"Promoted {name} v{latest.version} to Production.")
    return f"v{latest.version}"


@flow(name="Toxic BERT Training Pipeline", task_runner=ConcurrentTaskRunner())
def toxic_bert_training_pipeline():
    ing = ingest_data.submit()
    spl = split_data.submit(wait_for=[ing])
    bert = log_bert.submit(wait_for=[spl])
    ev = evaluate.submit(wait_for=[bert])
    register_production.submit(wait_for=[ev])


if __name__ == "__main__":
    toxic_bert_training_pipeline()
