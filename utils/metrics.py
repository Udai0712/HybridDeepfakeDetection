import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(
    y_true,
    y_pred,
    y_prob=None,
    average: str = "binary",
):
    """
    Compute classification metrics for deepfake detection.

    Args:
        y_true: list or np.array of true labels (0/1)
        y_pred: list or np.array of predicted labels (0/1)
        y_prob: list or np.array of predicted probabilities for FAKE class
        average: 'binary' (default)

    Returns:
        metrics: dict containing accuracy, precision, recall, f1, auc
    """

    metrics = {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["f1"] = f1_score(
        y_true, y_pred, average=average, zero_division=0
    )

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    return metrics


def print_metrics(metrics: dict):
    """
    Pretty print metrics.
    """
    print("\n===== Evaluation Metrics =====")
    for k, v in metrics.items():
        if v is None:
            print(f"{k.upper():10s}: N/A")
        else:
            print(f"{k.upper():10s}: {v:.4f}")
    print("==============================\n")


def compute_confusion(y_true, y_pred):
    """
    Compute confusion matrix.

    Returns:
        [[TN, FP],
         [FN, TP]]
    """
    return confusion_matrix(y_true, y_pred)
