"""
models.py
---------
Thin wrapper classes around XGBoost, Prophet and an LSTM (Keras/TensorFlow)
so all three can be trained/evaluated through a common interface from
train.py, plus a naive seasonal baseline used to compute the accuracy
uplift reported in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Naive baseline: predict sales = value from 7 days ago (seasonal naive)
# ---------------------------------------------------------------------------
class NaiveSeasonalBaseline:
    def __init__(self, season_length: int = 7):
        self.season_length = season_length

    def fit(self, *_args, **_kwargs):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Relies on lag_7 being present in the feature frame
        lag_col = f"lag_{self.season_length}"
        if lag_col in df.columns:
            return df[lag_col].values
        return df["lag_7"].values


# ---------------------------------------------------------------------------
# XGBoost regressor
# ---------------------------------------------------------------------------
class XGBoostForecaster:
    def __init__(self, **params):
        import xgboost as xgb
        default_params = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        default_params.update(params)
        self.model = xgb.XGBRegressor(**default_params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self, feature_names):
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Prophet (per-series). For a multi-series panel we aggregate to the
# store/item level requested at fit time.
# ---------------------------------------------------------------------------
class ProphetForecaster:
    def __init__(self, **params):
        from prophet import Prophet
        default_params = dict(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        default_params.update(params)
        self.model = Prophet(**default_params)
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        """df must have columns 'date' (or 'ds') and 'sales' (or 'y'),
        optionally 'promo' as a regressor."""
        prophet_df = df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"] + (["promo"] if "promo" in df.columns else [])]
        if "promo" in prophet_df.columns:
            self.model.add_regressor("promo")
        self.model.fit(prophet_df)
        self._fitted = True
        return self

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        future = future_df.rename(columns={"date": "ds"})[["ds"] + (["promo"] if "promo" in future_df.columns else [])]
        forecast = self.model.predict(future)
        return forecast["yhat"].clip(lower=0).values


# ---------------------------------------------------------------------------
# LSTM (Keras) sequence model
# ---------------------------------------------------------------------------
class LSTMForecaster:
    def __init__(self, lookback: int = 28, n_features: int = 1, units: int = 64, epochs: int = 20,
                 batch_size: int = 64):
        self.lookback = lookback
        self.n_features = n_features
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    def _build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(self.units, input_shape=(self.lookback, self.n_features), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    @staticmethod
    def make_sequences(series: np.ndarray, lookback: int):
        X, y = [], []
        for i in range(lookback, len(series)):
            X.append(series[i - lookback:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit(self, series: np.ndarray):
        self.scaler_mean, self.scaler_std = series.mean(), series.std() + 1e-8
        scaled = (series - self.scaler_mean) / self.scaler_std

        X, y = self.make_sequences(scaled, self.lookback)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        self.model = self._build_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, series: np.ndarray) -> np.ndarray:
        scaled = (series - self.scaler_mean) / self.scaler_std
        X, _ = self.make_sequences(scaled, self.lookback)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        preds = self.model.predict(X, verbose=0).flatten()
        return preds * self.scaler_std + self.scaler_mean
