# -*- coding: utf-8 -*-
"""Прогноз температуры на горизонт H дней (таргет shift(-H)) по каждому городу."""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import prepare_forecasting_frame


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    nz = np.where(np.abs(y_true) < 0.1, 0.1, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / nz)) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def train_city_forecasters(
    city_df: pd.DataFrame,
    city_name: str,
    horizon: int = 30,
    train_year_max: int = 2023,
    val_year: int = 2024,
    test_year: int = 2025,
    use_prophet: bool = False,
) -> dict:
    df_feat, feature_cols = prepare_forecasting_frame(city_df, horizon=horizon)
    df_feat = df_feat.copy()
    df_feat["year"] = pd.to_datetime(df_feat["timestamp"]).dt.year

    train_mask = df_feat["year"] <= train_year_max
    val_mask = df_feat["year"] == val_year
    test_mask = df_feat["year"] == test_year

    X = df_feat[feature_cols].values
    y = df_feat["target"].values.astype(np.float64)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    out: dict = {"feature_cols": feature_cols, "city": city_name}

    print(f"\n=== {city_name}: train={len(X_train)} val={len(X_val)} test={len(X_test)} ===")

    models = {}
    metrics = {}

    if len(X_train) > 50:
        lgbm = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=10,
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_train, y_train)
        models["LightGBM"] = lgbm
        pred_test = lgbm.predict(X_test)
        metrics["LightGBM"] = evaluate_forecast(y_test, pred_test)

        rf = RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        models["RandomForest"] = rf
        metrics["RandomForest"] = evaluate_forecast(y_test, rf.predict(X_test))

    # Prophet: долго на больших рядах; включайте use_prophet=True при необходимости
    if not use_prophet:
        metrics["Prophet"] = {"skipped": True}
        out["models"] = models
        out["metrics"] = metrics
        out["inference_tail_df"] = df_feat
        return out

    try:
        from prophet import Prophet

        prophet_df = city_df[["timestamp", "temperature_2m"]].copy()
        prophet_df = prophet_df.rename(columns={"timestamp": "ds", "temperature_2m": "y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df = prophet_df.dropna()

        tr = prophet_df[prophet_df["ds"].dt.year <= train_year_max]
        te = prophet_df[prophet_df["ds"].dt.year == test_year]

        if len(tr) > 100 and len(te) > 5:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            m.fit(tr)
            future = pd.DataFrame({"ds": te["ds"].values})
            fc = m.predict(future)
            metrics["Prophet"] = evaluate_forecast(te["y"].values, fc["yhat"].values)
            models["Prophet"] = m
    except Exception as exc:
        metrics["Prophet"] = {"error": str(exc)}

    out["models"] = models
    out["metrics"] = metrics
    # Последняя строка признаков для инференса sklearn-моделей
    out["inference_tail_df"] = df_feat
    return out


def train_all_cities(
    data: dict[str, pd.DataFrame], horizon: int = 30, use_prophet: bool = False
) -> dict[str, dict]:
    return {
        city: train_city_forecasters(df, city, horizon=horizon, use_prophet=use_prophet)
        for city, df in data.items()
    }


def pick_best_sklearn_model(bundle: dict):
    """Выбираем LightGBM или RF по MAE на тесте."""
    metrics = bundle.get("metrics", {})
    best_name, best_mae = None, float("inf")
    for name in ("LightGBM", "RandomForest"):
        if name in metrics and isinstance(metrics[name], dict) and "mae" in metrics[name]:
            if metrics[name]["mae"] < best_mae:
                best_mae = metrics[name]["mae"]
                best_name = name
    if best_name is None:
        return None, None
    return bundle["models"][best_name], best_name
