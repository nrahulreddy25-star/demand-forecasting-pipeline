"""
features.py
------------
Feature engineering for the retail demand forecasting pipeline:
lag features, rolling-window statistics, and calendar/seasonality features.
Designed to work per (store_id, item_id) group so features never leak
across independent series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical encodings so the model sees seasonality as continuous, not categorical
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_lag_features(df: pd.DataFrame, group_cols, target_col: str = "sales",
                      lags=LAGS) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(group_cols)[target_col]
    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, group_cols, target_col: str = "sales",
                          windows=ROLLING_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    # Shift by 1 first so rolling stats never see the current day's own sales
    shifted = df.groupby(group_cols)[target_col].shift(1)
    for window in windows:
        df[f"roll_mean_{window}"] = (
            shifted.groupby([df[c] for c in group_cols]).transform(
                lambda s: s.rolling(window, min_periods=max(1, window // 2)).mean()
            )
        )
        df[f"roll_std_{window}"] = (
            shifted.groupby([df[c] for c in group_cols]).transform(
                lambda s: s.rolling(window, min_periods=max(1, window // 2)).std()
            )
        )
    return df


def build_feature_frame(df: pd.DataFrame, group_cols=("store_id", "item_id"),
                         target_col: str = "sales") -> pd.DataFrame:
    """Full feature pipeline: calendar + lag + rolling features.

    Rows without enough history to compute the longest lag are dropped,
    matching standard practice for lag-based forecasting models.
    """
    group_cols = list(group_cols)
    df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df, group_cols, target_col)
    df = add_rolling_features(df, group_cols, target_col)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS = (
    ["store_id", "item_id", "promo", "day_of_week", "day_of_month", "week_of_year",
     "month", "quarter", "year", "is_weekend",
     "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]
    + [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
)
