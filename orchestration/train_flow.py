# orchestration/train_flow.py
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
import os, subprocess

MLFLOW = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")

def run(cmd_list, extra_env=None) -> int:
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW
    if extra_env:
        env.update(extra_env)
    p = subprocess.Popen(cmd_list, env=env)
    return p.wait()

@task(name="train_logreg")
def train_logreg():
    return run(["python", "-m", "scripts.train_logreg_multi"])

@task(name="train_nb")
def train_nb():
    return run(["python", "-m", "scripts.train_nb_multi"])

@task(name="train_svm")
def train_svm():
    return run(["python", "-m", "scripts.train_svm_multi"])

@task(name="log_bert")
def log_bert():
    return run(
        ["python", "-m", "scripts.log_bert_multilabel],
        extra_env={"N_VAL": "300", "MAX_LEN": "96", "BATCH_SIZE": "64"},
    )

@task(name="finalize")
def finalize(a: int, b: int, c: int, d: int):
    get_run_logger().info(f"Exit codes -> logreg={a}, nb={b}, svm={c}, bert={d}")

# Run tasks in parallel for the side-by-side bars
@flow(name="train_all_models_multilabel", task_runner=ConcurrentTaskRunner(max_workers=4))
def train_all_models_multilabel():
    f1 = train_logreg.submit()
    f2 = train_nb.submit()
    f3 = train_svm.submit()
    f4 = log_bert.submit()
    finalize.submit(f1, f2, f3, f4)

if __name__ == "__main__":
    train_all_models_multilabel()
