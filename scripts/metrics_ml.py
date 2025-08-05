from sklearn.metrics import f1_score

from src.constants import LABELS


def multilabel_f1(y_true, y_pred):
    # y_true, y_pred shape: (n_samples, n_labels), values in {0,1}
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_label = {f"f1_{lbl}": float(s) for lbl, s in zip(LABELS, f1_per_label)}
    return f1_micro, f1_macro, per_label
