import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import build_feature_frame, add_calendar_features, FEATURE_COLUMNS
from src.evaluate import mae, rmse, mape, summary, improvement_over_baseline


def _toy_df(n=100):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": dates,
        "store_id": 1,
        "item_id": 1,
        "sales": rng.integers(5, 50, size=n),
        "promo": rng.integers(0, 2, size=n),
    })


def test_calendar_features_add_expected_columns():
    df = _toy_df(10)
    out = add_calendar_features(df)
    for col in ["day_of_week", "month", "is_weekend", "month_sin", "doy_cos"]:
        assert col in out.columns


def test_build_feature_frame_drops_insufficient_history_rows():
    df = _toy_df(60)
    feat = build_feature_frame(df)
    # First rows (fewer than max lag=28 days of history) must be dropped
    assert feat["date"].min() > df["date"].min()
    assert not feat[["lag_1", "lag_7", "lag_14", "lag_28"]].isna().any().any()


def test_build_feature_frame_has_all_expected_feature_columns():
    df = _toy_df(60)
    feat = build_feature_frame(df)
    for col in FEATURE_COLUMNS:
        assert col in feat.columns, f"missing feature column {col}"


def test_no_target_leakage_in_rolling_features():
    df = _toy_df(60)
    feat = build_feature_frame(df)
    # roll_mean_7 for a row must not depend on that row's own sales value —
    # check by construction: shifting happens before rolling in features.py
    row = feat.iloc[10]
    window_dates = pd.date_range(row["date"] - pd.Timedelta(days=7), row["date"] - pd.Timedelta(days=1))
    manual_mean = df[df["date"].isin(window_dates)]["sales"].mean()
    assert abs(row["roll_mean_7"] - manual_mean) < 1e-6


def test_metrics_basic_values():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    assert mae(y_true, y_pred) == np.mean([2, 2, 3])
    assert rmse(y_true, y_pred) > 0
    s = summary(y_true, y_pred)
    assert set(s.keys()) == {"mae", "rmse", "mape", "smape"}


def test_improvement_over_baseline():
    assert improvement_over_baseline(10, 8) == 20.0
    assert improvement_over_baseline(10, 10) == 0.0
    assert improvement_over_baseline(0, 5) == 0.0
