"""
Streamlit dashboard for the retail demand forecasting pipeline.

Run:
    streamlit run dashboard/app.py

Lets a user pick a store/item, view historical sales, and fetch a live
forecast from the FastAPI service (or fall back to the local model file
if the API isn't running).
"""
import os
import sys
import json
import pickle
from datetime import timedelta

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = "data/retail_sales.csv"
MODEL_PATH = "models/best_model.pkl"
RESULTS_PATH = "models/results_summary.json"

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide", page_icon="📈")


@st.cache_data
def load_history():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@st.cache_data
def load_results_summary():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


def call_api_forecast(store_id, item_id, horizon):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"store_id": store_id, "item_id": item_id, "horizon_days": horizon},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def local_fallback_forecast(history, store_id, item_id, horizon):
    """Naive fallback if the API isn't reachable: seasonal-naive using last
    7 days, purely so the dashboard remains usable standalone."""
    series = history[(history.store_id == store_id) & (history.item_id == item_id)].sort_values("date")
    if series.empty:
        return None
    last_week = series["sales"].tail(7).values
    last_date = series["date"].max()
    forecast = []
    for i in range(1, horizon + 1):
        val = float(last_week[(i - 1) % 7])
        forecast.append({"date": (last_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                          "predicted_sales": val})
    return {"model_used": "naive_fallback (API unreachable)", "forecast": forecast}


st.title("📈 Retail Demand Forecasting Dashboard")
st.caption("XGBoost / Prophet / LSTM — best model selected automatically via MLflow experiment tracking")

history = load_history()
if history is None:
    st.error(
        "No dataset found at `data/retail_sales.csv`. "
        "Run `python data/generate_data.py` (or drop in your own retail sales CSV) first."
    )
    st.stop()

stores = sorted(history["store_id"].unique())
items = sorted(history["item_id"].unique())

with st.sidebar:
    st.header("Forecast Settings")
    store_id = st.selectbox("Store", stores, index=0)
    item_id = st.selectbox("Item / SKU", items, index=0)
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=14)
    show_history_days = st.slider("History window shown (days)", 30, 365, 120)
    st.divider()
    st.caption(f"API endpoint: `{API_URL}`")
    fetch = st.button("🔮 Generate forecast", type="primary", use_container_width=True)

series = history[(history.store_id == store_id) & (history.item_id == item_id)].sort_values("date")

col1, col2, col3 = st.columns(3)
col1.metric("Avg daily sales (last 90d)", f"{series['sales'].tail(90).mean():.1f}")
col2.metric("Peak day sales", f"{series['sales'].max():.0f}")
col3.metric("Total history", f"{len(series):,} days")

if fetch or "forecast_result" in st.session_state:
    if fetch:
        result, err = call_api_forecast(store_id, item_id, horizon)
        if err:
            st.warning(f"Could not reach FastAPI service at {API_URL} ({err}). Showing a local fallback forecast — start the API with `uvicorn api.main:app --reload` for real model predictions.")
            result = local_fallback_forecast(history, store_id, item_id, horizon)
        st.session_state["forecast_result"] = result

    result = st.session_state["forecast_result"]
    if result:
        fc_df = pd.DataFrame(result["forecast"])
        fc_df["date"] = pd.to_datetime(fc_df["date"])

        hist_plot = series.tail(show_history_days)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_plot["date"], y=hist_plot["sales"],
            mode="lines", name="Historical sales", line=dict(color="#4C78A8")
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["date"], y=fc_df["predicted_sales"],
            mode="lines+markers", name=f"Forecast ({result['model_used']})",
            line=dict(color="#F58518", dash="dash")
        ))
        fig.update_layout(
            title=f"Store {store_id} · Item {item_id} — Sales Forecast",
            xaxis_title="Date", yaxis_title="Units sold",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast table")
        st.dataframe(fc_df, use_container_width=True, hide_index=True)
        st.download_button("Download forecast CSV", fc_df.to_csv(index=False), "forecast.csv")
else:
    st.info("Set your store/item/horizon in the sidebar and click **Generate forecast**.")

st.divider()
st.subheader("Model comparison (from latest training run)")
results = load_results_summary()
if results:
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "val_rmse": round(r["val"]["rmse"], 2),
            "val_mae": round(r["val"]["mae"], 2),
            "test_rmse": round(r["test"]["rmse"], 2),
            "test_mae": round(r["test"]["mae"], 2),
            "test_mape_%": round(r["test"]["mape"], 1),
        })
    comp_df = pd.DataFrame(rows).sort_values("test_rmse")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    if "naive" in results:
        baseline_rmse = results["naive"]["test"]["rmse"]
        best_row = comp_df.iloc[0]
        if best_row["model"] != "naive":
            uplift = (baseline_rmse - best_row["test_rmse"]) / baseline_rmse * 100
            st.success(f"Best model **{best_row['model']}** improves test RMSE by **{uplift:.1f}%** over the naive seasonal baseline.")
else:
    st.caption("Run `python src/train.py` to populate model comparison metrics here.")
