import os

from mlflow.tracking import MlflowClient

tracking = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
client = MlflowClient(tracking_uri=tracking)

name = "ToxicCommentClassifierMulti"
version = "2"  # <-- set the version you want to promote
client.transition_model_version_stage(
    name=name,
    version=version,
    stage="Production",
    archive_existing_versions=True,
)
print(f"Promoted {name} v{version} to Production.")
