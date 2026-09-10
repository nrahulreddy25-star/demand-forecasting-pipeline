"""
train.py
--------
End-to-end training & experiment-tracking entry point.

Trains a naive seasonal baseline, XGBoost, Prophet and an LSTM on the
retail sales panel, logs params/metrics/artifacts for each run to MLflow,
picks the best model by validation RMSE, and saves it to models/ for the
FastAPI service to load.

Usage:
    python src/train.py --data data/retail_sales.csv --store_id 1 --item_id 1
"""
import argparse
import json
import os
import pickle
import sys

import mlflow
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import build_feature_frame, FEATURE_COLUMNS
from src.evaluate import summary, improvement_over_baseline
from src.models import NaiveSeasonalBaseline, XGBoostForecaster

TEST_DAYS = 60
VAL_DAYS = 60


def time_split(df: pd.DataFrame, test_days=TEST_DAYS, val_days=VAL_DAYS):
    df = df.sort_values("date")
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    return train, val, test


def run_naive(train, val, test, experiment):
    with mlflow.start_run(run_name="naive_seasonal_baseline"):
        mlflow.log_param("model_type", "naive_seasonal")
        mlflow.log_param("season_length", 7)

        model = NaiveSeasonalBaseline(season_length=7)
        preds_val = model.predict(val)
        preds_test = model.predict(test)

        val_metrics = summary(val["sales"], preds_val)
        test_metrics = summary(test["sales"], preds_test)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        return model, val_metrics, test_metrics


def run_xgboost(train, val, test, experiment):
    with mlflow.start_run(run_name="xgboost"):
        params = dict(n_estimators=400, max_depth=6, learning_rate=0.05)
        mlflow.log_param("model_type", "xgboost")
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = XGBoostForecaster(**params)
        model.fit(train[FEATURE_COLUMNS], train["sales"], val[FEATURE_COLUMNS], val["sales"])

        preds_val = model.predict(val[FEATURE_COLUMNS])
        preds_test = model.predict(test[FEATURE_COLUMNS])

        val_metrics = summary(val["sales"], preds_val)
        test_metrics = summary(test["sales"], preds_test)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        importances = model.feature_importances(FEATURE_COLUMNS)
        importances.to_csv("models/xgboost_feature_importance.csv")
        mlflow.log_artifact("models/xgboost_feature_importance.csv")
        mlflow.xgboost.log_model(model.model, artifact_path="model")

        return model, val_metrics, test_metrics


def run_prophet(train, val, test, experiment):
    try:
        from src.models import ProphetForecaster
    except ImportError:
        print("Prophet not installed — skipping. `pip install prophet` to enable.")
        return None, None, None

    with mlflow.start_run(run_name="prophet"):
        mlflow.log_param("model_type", "prophet")
        mlflow.log_param("seasonality_mode", "multiplicative")

        model = ProphetForecaster()
        model.fit(train[["date", "sales", "promo"]])

        preds_val = model.predict(val[["date", "promo"]])
        preds_test = model.predict(test[["date", "promo"]])

        val_metrics = summary(val["sales"], preds_val)
        test_metrics = summary(test["sales"], preds_test)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        return model, val_metrics, test_metrics


def run_lstm(train, val, test, experiment):
    try:
        from src.models import LSTMForecaster
    except ImportError:
        print("TensorFlow not installed — skipping. `pip install tensorflow` to enable.")
        return None, None, None

    lookback = 28
    with mlflow.start_run(run_name="lstm"):
        mlflow.log_param("model_type", "lstm")
        mlflow.log_param("lookback", lookback)
        mlflow.log_param("epochs", 20)

        full_series = pd.concat([train, val])["sales"].values.astype(float)
        model = LSTMForecaster(lookback=lookback, epochs=20)
        model.fit(full_series)

        val_series = pd.concat([train.tail(lookback), val])["sales"].values.astype(float)
        preds_val = model.predict(val_series)
        test_series = pd.concat([val.tail(lookback), test])["sales"].values.astype(float)
        preds_test = model.predict(test_series)

        val_metrics = summary(val["sales"].values, preds_val)
        test_metrics = summary(test["sales"].values, preds_test)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        return model, val_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/retail_sales.csv")
    parser.add_argument("--store_id", type=int, default=1)
    parser.add_argument("--item_id", type=int, default=1)
    parser.add_argument("--experiment", default="demand-forecasting")
    parser.add_argument("--skip_prophet", action="store_true")
    parser.add_argument("--skip_lstm", action="store_true")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment(args.experiment)

    raw = pd.read_csv(args.data, parse_dates=["date"])
    raw = raw[(raw["store_id"] == args.store_id) & (raw["item_id"] == args.item_id)]

    feat = build_feature_frame(raw)
    train, val, test = time_split(feat)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    results = {}

    model, val_m, test_m = run_naive(train, val, test, args.experiment)
    results["naive"] = {"model": model, "val": val_m, "test": test_m}

    model, val_m, test_m = run_xgboost(train, val, test, args.experiment)
    results["xgboost"] = {"model": model, "val": val_m, "test": test_m}

    if not args.skip_prophet:
        model, val_m, test_m = run_prophet(train, val, test, args.experiment)
        if model is not None:
            results["prophet"] = {"model": model, "val": val_m, "test": test_m}

    if not args.skip_lstm:
        model, val_m, test_m = run_lstm(train, val, test, args.experiment)
        if model is not None:
            results["lstm"] = {"model": model, "val": val_m, "test": test_m}

    # Pick best model by validation RMSE
    best_name = min(results, key=lambda k: results[k]["val"]["rmse"])
    best = results[best_name]
    baseline_rmse = results["naive"]["test"]["rmse"]
    uplift = improvement_over_baseline(baseline_rmse, best["test"]["rmse"])

    print("\n=== Model comparison (test set) ===")
    for name, r in results.items():
        print(f"{name:>10s}: RMSE={r['test']['rmse']:.2f}  MAE={r['test']['mae']:.2f}  MAPE={r['test']['mape']:.1f}%")
    print(f"\nBest model: {best_name} — {uplift:.1f}% RMSE improvement over naive baseline")

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump({"name": best_name, "model": best["model"]}, f)

    with open("models/results_summary.json", "w") as f:
        json.dump(
            {k: {"val": v["val"], "test": v["test"]} for k, v in results.items()},
            f, indent=2,
        )

    print("\nSaved best model to models/best_model.pkl")


if __name__ == "__main__":
    main()
