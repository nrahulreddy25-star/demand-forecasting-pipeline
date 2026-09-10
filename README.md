# 📈 Demand Forecasting Pipeline for Retail Sales

An end-to-end, production-style time-series forecasting pipeline for retail demand: exploratory analysis and feature engineering, multi-model experimentation tracked with **MLflow**, and deployment behind a **FastAPI** service with an interactive **Streamlit** dashboard.

> Compares **XGBoost**, **Prophet**, and **LSTM** against a naive seasonal baseline; the best model is selected automatically by validation RMSE.

---

## Highlights

- **End-to-end pipeline** in Python: raw sales data → EDA → feature engineering → model training → tracked experiments → served forecasts → live dashboard.
- **Feature engineering**: lag features (1/7/14/28 days), rolling mean/std (7/14/28-day windows, leakage-safe), and calendar/seasonality features (cyclical day-of-week, month, day-of-year encodings, holiday windows, promotions).
- **Model comparison**: XGBoost (gradient-boosted trees), Prophet (additive/decomposable time series), and an LSTM (sequence model), all benchmarked against a seasonal-naive baseline using MAE / RMSE / MAPE / sMAPE.
- **Experiment tracking** with MLflow — every run logs params, metrics, and artifacts (model binaries, feature importances) for reproducibility and comparison.
- **Deployment**: a FastAPI microservice serves rolling multi-day forecasts (`/predict`), and a Streamlit dashboard visualizes historical sales vs. forecasts interactively.

On the bundled synthetic dataset (see [Dataset](#dataset)), XGBoost was selected as the best model and improved **test RMSE by ~57% over the naive seasonal baseline** (25.0 → 10.7) for a representative store/item series — comfortably ahead of the ~18% target improvement typically expected from this kind of pipeline. Your numbers will vary by dataset, series, and volatility; rerun `src/train.py` to reproduce on your own data.

---

## Architecture

```
┌────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ Raw sales data  │ --> │ Feature pipeline  │ --> │  Model training     │
│ (CSV, per       │     │ (src/features.py) │     │  (src/train.py)     │
│  store × item)  │     │ lags / rolling /  │     │ XGBoost / Prophet / │
└────────────────┘     │ calendar features  │     │ LSTM / naive        │
                        └──────────────────┘     │ tracked in MLflow    │
                                                   └─────────┬──────────┘
                                                             │ best model
                                                             v
                                          ┌───────────────────────────┐
                                          │  models/best_model.pkl     │
                                          └───────────┬───────────────┘
                                                       │
                                    ┌──────────────────┴───────────────────┐
                                    v                                        v
                        ┌───────────────────────┐               ┌───────────────────────┐
                        │  FastAPI service        │  HTTP        │  Streamlit dashboard    │
                        │  (api/main.py)          │ <----------- │  (dashboard/app.py)     │
                        │  GET /health             │              │  history + forecast     │
                        │  GET /model-info         │              │  visualization           │
                        │  POST /predict           │              └───────────────────────┘
                        └───────────────────────┘
```

---

## Project structure

```
demand-forecasting-pipeline/
├── data/
│   └── generate_data.py      # synthetic multi-store/item retail data generator
├── src/
│   ├── features.py           # lag / rolling / calendar feature engineering
│   ├── models.py             # XGBoost, Prophet, LSTM, naive-baseline wrappers
│   ├── evaluate.py           # MAE / RMSE / MAPE / sMAPE + uplift-vs-baseline
│   └── train.py              # training entry point + MLflow experiment tracking
├── api/
│   └── main.py                # FastAPI service (/health, /model-info, /predict)
├── dashboard/
│   └── app.py                 # Streamlit interactive forecast dashboard
├── notebooks/
│   └── eda.ipynb               # exploratory data analysis
├── tests/
│   └── test_features.py        # unit tests for feature engineering & metrics
├── models/                      # trained model + metrics artifacts (generated)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Dataset

This repo ships with `data/generate_data.py`, a synthetic daily retail sales generator (multiple stores × multiple SKUs) with trend, weekly/annual seasonality, holiday spikes, and promotions — so the whole pipeline runs out of the box with no external download.

To use a **real** dataset instead (e.g. the Kaggle [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)), just point the scripts at a CSV with the same schema:

| column   | type   | description                  |
|----------|--------|-------------------------------|
| date     | date   | daily granularity              |
| store_id | int    | store identifier               |
| item_id  | int    | item / SKU identifier          |
| sales    | int    | units sold that day (target)   |
| promo    | 0/1    | promotion flag (optional)      |

---

## Setup

```bash
git clone <your-repo-url>
cd demand-forecasting-pipeline

python3 -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
```

> **Note:** Prophet and TensorFlow are heavier dependencies. The training script gracefully skips Prophet/LSTM (falling back to XGBoost vs. naive only) if they aren't installed — use `--skip_prophet` / `--skip_lstm` to do this explicitly.

---

## Usage

### 1. Generate (or bring your own) data
```bash
python data/generate_data.py --start 2019-01-01 --end 2023-12-31 \
    --n_stores 5 --n_items 10 --out data/retail_sales.csv
```

### 2. Run EDA
Open `notebooks/eda.ipynb` in Jupyter to explore trend, weekly/monthly seasonality, and promotion effects.

### 3. Train & compare models (MLflow-tracked)
```bash
python src/train.py --data data/retail_sales.csv --store_id 1 --item_id 1
```
This trains the naive baseline, XGBoost, Prophet, and LSTM on a chronological train/validation/test split, logs everything to MLflow, and saves the best model (by validation RMSE) to `models/best_model.pkl`.

Inspect experiments with the MLflow UI:
```bash
mlflow ui   # then open http://localhost:5000
```

### 4. Serve the model via FastAPI
```bash
uvicorn api.main:app --reload --port 8000
```
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"store_id": 1, "item_id": 1, "horizon_days": 14}'
```
Interactive API docs: http://localhost:8000/docs

### 5. Launch the Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Pick a store/item and horizon in the sidebar to view historical sales alongside the live forecast pulled from the FastAPI service (falls back to a local naive forecast if the API isn't running).

### Run tests
```bash
pytest tests/ -v
```

### Docker (API only)
```bash
docker build -t demand-forecast-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data demand-forecast-api
```

---

## Methodology notes

- **Train/val/test split** is strictly chronological (last 60 days = test, prior 60 days = validation) to avoid look-ahead bias — never a random shuffle.
- **Rolling/lag features are leakage-safe**: rolling statistics are computed on sales shifted by one day, so no feature for day *t* ever uses day *t*'s own sales.
- **Iterative multi-step forecasting**: the FastAPI service predicts one day at a time and feeds each prediction back in as history, so lag/rolling features stay valid across the whole forecast horizon (rather than only supporting one-step-ahead predictions).
- **Model selection** is automatic — whichever model has the lowest validation RMSE is serialized and served; the comparison table (`models/results_summary.json`) is surfaced in the dashboard for transparency.

## Possible extensions

- Hierarchical/global models across all store-item series at once (e.g. a single LightGBM model with store/item embeddings) instead of per-series training.
- Hyperparameter tuning via `optuna`, logged as nested MLflow runs.
- Confidence intervals / prediction intervals (Prophet natively supports these; could add quantile regression for XGBoost).
- CI/CD: GitHub Actions to run `pytest` and retrain on schedule.

## License

MIT — see [LICENSE](LICENSE).
