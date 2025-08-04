from prefect import flow, task, get_run_logger
import pandas as pd, psycopg2, datetime as dt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, MissingValuesPreset


@task
def compute_metrics():
    ref = pd.read_csv("data/raw/train.csv")[["comment_text", "toxic"]].rename(
        columns={"comment_text": "text", "toxic": "target"}
    )
    cur = (
        pd.read_csv("data/raw/train.csv")
        .sample(frac=0.2, random_state=42)[["comment_text", "toxic"]]
        .rename(columns={"comment_text": "text", "toxic": "target"})
    )
    rep = Report(metrics=[DataDriftPreset(), MissingValuesPreset()])
    rep.run(reference_data=ref, current_data=cur)
    d = rep.as_dict()
    return {
        "drift_score": d["metrics"][0]["result"]["dataset_drift"]["drift_share"],
        "missing_share": d["metrics"][1]["result"]["current"][
            "share_of_missing_values"
        ],
        "drifted_features": d["metrics"][0]["result"]["number_of_drifted_columns"],
    }


@task
def write_to_postgres(m):
    conn = psycopg2.connect(
        dbname="monitoring",
        user="evidently",
        password="evidently",
        host="postgres",
        port=5432,
    )
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS drift_results
                   (ts TIMESTAMP, drift_score DOUBLE PRECISION,
                    missing_share DOUBLE PRECISION, drifted_features INT)"""
    )
    cur.execute(
        "INSERT INTO drift_results VALUES (%s,%s,%s,%s)",
        (
            dt.datetime.utcnow(),
            m["drift_score"],
            m["missing_share"],
            m["drifted_features"],
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


@flow(name="monitoring_flow")
def monitoring_flow():
    log = get_run_logger()
    log.info("Running Evidently monitoring…")
    m = compute_metrics()
    write_to_postgres(m)
    log.info("Saved metrics to Postgres")


if __name__ == "__main__":
    monitoring_flow()
