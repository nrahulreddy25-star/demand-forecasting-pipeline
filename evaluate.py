"""
evaluate.py
-----------
Common forecast accuracy metrics used across all models so comparisons in
MLflow and the README are apples-to-apples.
"""
import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred, eps: float = 1.0):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def smape(y_true, y_pred, eps: float = 1.0):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def improvement_over_baseline(baseline_metric: float, model_metric: float) -> float:
    """% improvement of model over baseline for an error metric (lower=better)."""
    if baseline_metric == 0:
        return 0.0
    return float((baseline_metric - model_metric) / baseline_metric * 100)


def summary(y_true, y_pred) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
