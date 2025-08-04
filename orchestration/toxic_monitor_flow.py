# orchestration/toxic_monitor_flow.py
from prefect import flow, task, get_run_logger
import pandas as pd
import os, datetime as dt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow
import subprocess

# Config
REF_CSV = "data/raw/train.csv"
CURRENT_CSV = "data/raw/current.csv"
THRESHOLD = 0.3  # retraining trigger threshold


@task
def compute_drift():
    log = get_run_logger()
    ref = pd.read_csv(REF_CSV)[["comment_text", "toxic"]].rename(
        columns={"comment_text": "text", "toxic": "target"}
    )

    if os.path.exists(CURRENT_CSV):
        cur = pd.read_csv(CURRENT_CSV)[["comment_text"]].rename(
            columns={"comment_text": "text"}
        )
        log.info(f"Loaded current data: {CURRENT_CSV}")
    else:
        cur = ref.sample(frac=0.2, random_state=42)
        log.warning(f"{CURRENT_CSV} not found, using sample from reference for demo.")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    result = report.as_dict()

    drift_score = result["metrics"][0]["result"]["dataset_drift"]["drift_share"]
    drifted_features = result["metrics"][0]["result"]["number_of_drifted_columns"]

    log.info(f"Drift score: {drift_score}, Drifted features: {drifted_features}")

    # Log to MLflow
    mlflow.set_tracking_uri("http://localhost:5050")
    mlflow.set_experiment("monitoring")
    with mlflow.start_run(run_name=f"drift-check-{dt.datetime.now()}"):
        mlflow.log_metric("drift_score", drift_score)
        mlflow.log_metric("drifted_features", drifted_features)

    return drift_score


@task
def trigger_retraining():
    log = get_run_logger()
    log.info("🚀 Drift detected! Triggering retraining...")
    subprocess.run(["python", "orchestration/bert_only_flow.py"])
    log.info("✅ Retraining completed.")


@flow(name="Monitoring Pipeline")
def toxic_monitoring_pipeline():
    drift = compute_drift()
    if drift.result() > THRESHOLD:
        trigger_retraining()
    else:
        get_run_logger().info("No retraining needed.")


if __name__ == "__main__":
    toxic_monitoring_pipeline()
