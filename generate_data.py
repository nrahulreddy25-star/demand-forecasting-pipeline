"""
generate_data.py
-----------------
Generates a realistic synthetic daily retail sales dataset (multiple stores x
multiple SKUs) with trend, weekly/annual seasonality, holiday spikes,
promotions and noise. Used when no real dataset (e.g. Kaggle "Store Item
Demand Forecasting Challenge") is available, so the pipeline is fully
reproducible out of the box.

Usage:
    python data/generate_data.py --start 2019-01-01 --end 2023-12-31 \
        --n_stores 5 --n_items 10 --out data/retail_sales.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate_series(dates, base_level, trend_per_year, item_seasonality_amp,
                     weekday_amp, holiday_boost, promo_prob, noise_std, rng):
    n = len(dates)
    t = np.arange(n)

    # Long-run linear trend
    trend = base_level + trend_per_year * (t / 365.25)

    # Annual seasonality (peaks around winter holidays for a generic retailer)
    day_of_year = dates.dayofyear.values
    annual_seasonality = item_seasonality_amp * np.sin(
        2 * np.pi * (day_of_year - 80) / 365.25
    )

    # Weekly seasonality (weekend uplift)
    weekday = dates.dayofweek.values
    weekly_seasonality = weekday_amp * np.isin(weekday, [4, 5, 6]).astype(float)

    # Holiday spikes (simple fixed calendar: New Year, early July, Black
    # Friday / late Nov, and December run-up)
    is_holiday_window = (
        ((dates.month == 12) & (dates.day >= 15))
        | ((dates.month == 11) & (dates.day >= 24) & (dates.day <= 29))
        | ((dates.month == 1) & (dates.day <= 2))
        | ((dates.month == 7) & (dates.day <= 4))
    )
    holiday_effect = holiday_boost * is_holiday_window.astype(float)

    # Random promotions
    promo_flag = rng.random(n) < promo_prob
    promo_effect = np.where(promo_flag, base_level * 0.35, 0.0)

    noise = rng.normal(0, noise_std, size=n)

    sales = trend + annual_seasonality + weekly_seasonality + holiday_effect + promo_effect + noise
    sales = np.clip(sales, 0, None)
    sales = np.round(sales).astype(int)

    return sales, promo_flag.astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--n_stores", type=int, default=5)
    parser.add_argument("--n_items", type=int, default=10)
    parser.add_argument("--out", default="data/retail_sales.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dates = pd.date_range(args.start, args.end, freq="D")

    rows = []
    for store_id in range(1, args.n_stores + 1):
        store_scale = rng.uniform(0.7, 1.4)
        for item_id in range(1, args.n_items + 1):
            base_level = rng.uniform(15, 60) * store_scale
            trend_per_year = rng.uniform(-2, 8)
            seasonality_amp = base_level * rng.uniform(0.15, 0.4)
            weekday_amp = base_level * rng.uniform(0.05, 0.2)
            holiday_boost = base_level * rng.uniform(0.4, 1.0)
            promo_prob = rng.uniform(0.03, 0.08)
            noise_std = base_level * 0.12

            sales, promo_flag = generate_series(
                dates, base_level, trend_per_year, seasonality_amp,
                weekday_amp, holiday_boost, promo_prob, noise_std, rng
            )

            df = pd.DataFrame({
                "date": dates,
                "store_id": store_id,
                "item_id": item_id,
                "sales": sales,
                "promo": promo_flag,
            })
            rows.append(df)

    full = pd.concat(rows, ignore_index=True)
    full = full.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    full.to_csv(args.out, index=False)
    print(f"Wrote {len(full):,} rows to {args.out}")
    print(full.head())


if __name__ == "__main__":
    main()
