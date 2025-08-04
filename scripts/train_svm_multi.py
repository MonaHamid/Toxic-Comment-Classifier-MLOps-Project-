from scripts._common_multi import load_xy_multi, mlflow_setup
from scripts.metrics_ml import multilabel_f1
import mlflow, mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

# Load data + set MLflow
Xtr, Ytr, Xte, Yte = load_xy_multi()
mlflow_setup()

# Pipeline: TF-IDF + One-vs-Rest LinearSVC
pipe = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", OneVsRestClassifier(LinearSVC())),
    ]
)

# Train & predict
pipe.fit(Xtr, Ytr)
Yhat = pipe.predict(Xte)

# Metrics
f1_micro, f1_macro, per_label = multilabel_f1(Yte, Yhat)

# Log to MLflow
with mlflow.start_run(run_name="svm_linearsvc_ovr_multilabel"):
    mlflow.log_param("model", "linearsvc_ovr")
    mlflow.log_param("vectorizer", "tfidf_50k_1-2gram")
    mlflow.log_metric("f1_micro", f1_micro)
    mlflow.log_metric("f1_macro", f1_macro)
    for k, v in per_label.items():
        mlflow.log_metric(k, v)
    mlflow.sklearn.log_model(pipe, "model")

print(f"SVM OVR f1_micro={f1_micro:.4f}  f1_macro={f1_macro:.4f}")
