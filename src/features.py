# -*- coding: utf-8 -*-
"""Признаки для классификации окон и для регрессии прогноза t+H."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" not in df.columns:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    def season(m: int) -> int:
        if m in (12, 1, 2):
            return 0
        if m in (3, 4, 5):
            return 1
        if m in (6, 7, 8):
            return 2
        return 3

    df["season"] = df["month"].map(season)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def create_lag_features(
    df: pd.DataFrame, target_col: str = "temperature_2m", lags: list[int] | None = None
) -> pd.DataFrame:
    df = df.copy()
    lags = lags or [1, 2, 3, 7, 14, 30]
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "temperature_2m",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    windows = windows or [3, 7, 14, 30]
    for w in windows:
        df[f"{target_col}_rolling_mean_{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_rolling_std_{w}"] = df[target_col].rolling(w).std()
        df[f"{target_col}_rolling_max_{w}"] = df[target_col].rolling(w).max()
        df[f"{target_col}_rolling_min_{w}"] = df[target_col].rolling(w).min()
    return df


def create_dynamics_features(df: pd.DataFrame, target_col: str = "temperature_2m") -> pd.DataFrame:
    df = df.copy()
    df["temp_velocity"] = df[target_col].diff()
    df["temp_acceleration"] = df["temp_velocity"].diff()
    if "surface_pressure" in df.columns:
        df["pressure_velocity"] = df["surface_pressure"].diff()
    return df


def create_aggregated_features_for_classification(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    df["temp_amplitude"] = (
        df["temperature_2m"].rolling(window).max()
        - df["temperature_2m"].rolling(window).min()
    )
    df["precip_days_count"] = (df["precipitation"] > 0.1).rolling(window).sum()
    df["max_wind_speed"] = df["wind_speed_10m"].rolling(window).max()
    df["mean_temp"] = df["temperature_2m"].rolling(window).mean()
    df["temp_std"] = df["temperature_2m"].rolling(window).std()

    for col in ["temperature_2m", "precipitation", "wind_speed_10m", "surface_pressure"]:
        if col in df.columns:
            df[f"{col}_mean_30d"] = df[col].rolling(30).mean()
            df[f"{col}_std_30d"] = df[col].rolling(30).std()
            df[f"{col}_min_30d"] = df[col].rolling(30).min()
            df[f"{col}_max_30d"] = df[col].rolling(30).max()
    return df


def prepare_features_for_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = create_time_features(df)
    df = create_aggregated_features_for_classification(df, window=30)
    return df


def aggregate_window_stats(window: np.ndarray, extended: bool = True) -> np.ndarray:
    """Окно (n_steps, n_features) → вектор статистик."""
    feats: list[float] = []
    for j in range(window.shape[1]):
        col = window[:, j]
        feats.extend([float(np.mean(col)), float(np.std(col))])
        feats.extend([float(np.max(col) - np.min(col)), float(np.percentile(col, 25)), float(np.percentile(col, 75))])
        if extended:
            feats.append(float(np.mean(np.diff(col))))
    return np.array(feats, dtype=np.float64)


def aggregate_windows_batch(X: np.ndarray) -> np.ndarray:
    """
    X: (n_samples, n_steps, n_features) → (n_samples, 6 * n_features).
    Векторная агрегация для RandomForest (быстро на больших выборках).
    """
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    amp = X.max(axis=1) - X.min(axis=1)
    q25 = np.percentile(X, 25, axis=1)
    q75 = np.percentile(X, 75, axis=1)
    vel = np.diff(X, axis=1).mean(axis=1)
    return np.hstack([mean, std, amp, q25, q75, vel]).astype(np.float64)


def prepare_forecasting_frame(
    df: pd.DataFrame,
    target_col: str = "temperature_2m",
    lookback_lags: list[int] | None = None,
    horizon: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Признаки по последним дням (лаги/скользящие), цель — температура через horizon дней.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    lookback_lags = lookback_lags or [1, 2, 3, 7, 14, 30]
    df = create_time_features(df)
    df = create_lag_features(df, target_col, lags=lookback_lags)
    df = create_rolling_features(df, target_col, windows=[3, 7, 14, 30])
    df = create_dynamics_features(df, target_col)

    others = ["relative_humidity_2m", "precipitation", "wind_speed_10m", "surface_pressure"]
    for feat in others:
        if feat in df.columns:
            df[f"{feat}_lag_1"] = df[feat].shift(1)
            df[f"{feat}_lag_7"] = df[feat].shift(7)

    df["target"] = df[target_col].shift(-horizon)
    df = df.dropna()

    exclude = {"timestamp", target_col, "target", "weathercode", "snowfall", "rain"}
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df, feature_cols
