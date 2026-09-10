"""
FastAPI service for the demand forecasting model.

Run:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health                       liveness check
    GET  /model-info                   which model is loaded + metrics
    POST /predict                      forecast sales for a store/item over N days
"""
import os
import pickle
import sys
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_feature_frame, FEATURE_COLUMNS

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "data/retail_sales.csv")

app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Serves forecasts from the best-performing model "
                 "(XGBoost / Prophet / LSTM, selected by validation RMSE).",
    version="1.0.0",
)

_state = {"model": None, "model_name": None, "history": None}


class PredictRequest(BaseModel):
    store_id: int = Field(..., example=1)
    item_id: int = Field(..., example=1)
    horizon_days: int = Field(14, ge=1, le=90, example=14)


class PredictResponse(BaseModel):
    store_id: int
    item_id: int
    model_used: str
    forecast: List[dict]


def _load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["name"]


def _load_history():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@app.on_event("startup")
def startup():
    _state["model"], _state["model_name"] = _load_model()
    _state["history"] = _load_history()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    if _state["model"] is None:
        raise HTTPException(503, "No trained model found. Run `python src/train.py` first.")
    info = {"model_name": _state["model_name"]}
    results_path = "models/results_summary.json"
    if os.path.exists(results_path):
        import json
        with open(results_path) as f:
            info["results_summary"] = json.load(f)
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _state["model"] is None:
        raise HTTPException(503, "No trained model found. Run `python src/train.py` first.")
    if _state["history"] is None:
        raise HTTPException(503, f"History data not found at {DATA_PATH}.")

    hist = _state["history"]
    series = hist[(hist["store_id"] == req.store_id) & (hist["item_id"] == req.item_id)]
    if series.empty:
        raise HTTPException(404, f"No history for store_id={req.store_id}, item_id={req.item_id}")

    series = series.sort_values("date").reset_index(drop=True)
    last_date = series["date"].max()

    # Iteratively roll the feature-based model forward day by day, feeding
    # each prediction back in as the new "actual" so lag/rolling features
    # stay valid across the forecast horizon.
    working = series.copy()
    model_name = _state["model_name"]
    forecasts = []

    for step in range(1, req.horizon_days + 1):
        next_date = last_date + timedelta(days=step)
        new_row = pd.DataFrame([{
            "date": next_date,
            "store_id": req.store_id,
            "item_id": req.item_id,
            "sales": 0,     # placeholder, unused for prediction row after feature build
            "promo": 0,
        }])
        candidate = pd.concat([working, new_row], ignore_index=True)
        feat = build_feature_frame(candidate)
        target_row = feat[feat["date"] == next_date]

        if target_row.empty:
            # Not enough history to build lag features (shouldn't happen once
            # working grows) — fall back to last known value.
            pred = float(working["sales"].iloc[-1])
        elif model_name == "xgboost":
            pred = float(_state["model"].predict(target_row[FEATURE_COLUMNS])[0])
        elif model_name == "naive":
            pred = float(target_row["lag_7"].iloc[0])
        elif model_name == "prophet":
            pred = float(_state["model"].predict(target_row[["date", "promo"]])[0])
        elif model_name == "lstm":
            recent = working["sales"].values.astype(float)[-_state["model"].lookback:]
            pred = float(_state["model"].predict(
                pd_concat_series(recent, 0)
            )[0]) if len(recent) == _state["model"].lookback else float(working["sales"].iloc[-1])
        else:
            pred = float(working["sales"].iloc[-1])

        pred = max(0.0, round(pred, 2))
        forecasts.append({"date": next_date.strftime("%Y-%m-%d"), "predicted_sales": pred})

        working = pd.concat([
            working,
            pd.DataFrame([{"date": next_date, "store_id": req.store_id,
                            "item_id": req.item_id, "sales": pred, "promo": 0}])
        ], ignore_index=True)

    return PredictResponse(
        store_id=req.store_id,
        item_id=req.item_id,
        model_used=model_name,
        forecast=forecasts,
    )


def pd_concat_series(recent, extra_placeholder):
    # Helper kept separate so LSTM's expected 1-step-ahead window shape is clear.
    import numpy as np
    return np.array(recent)
