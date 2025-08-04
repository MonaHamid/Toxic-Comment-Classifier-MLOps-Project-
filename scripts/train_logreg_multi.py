from scripts._common_multi import load_xy_multi, mlflow_setup
from scripts.metrics_ml import multilabel_f1
import mlflow, mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

Xtr, Ytr, Xte, Yte = load_xy_multi()
mlflow_setup()

pipe = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000))),
    ]
)

pipe.fit(Xtr, Ytr)
Yhat = pipe.predict(Xte)

f1_micro, f1_macro, per_label = multilabel_f1(Yte, Yhat)

with mlflow.start_run(run_name="logreg_ovr_multilabel"):
    mlflow.log_param("model", "logreg_ovr")
    mlflow.log_param("vectorizer", "tfidf_50k_1-2gram")
    mlflow.log_metric("f1_micro", f1_micro)
    mlflow.log_metric("f1_macro", f1_macro)
    for k, v in per_label.items():
        mlflow.log_metric(k, v)
    mlflow.sklearn.log_model(pipe, "model")
print(f"LogReg OVR f1_micro={f1_micro:.4f} f1_macro={f1_macro:.4f}")
