# -*- coding: utf-8 -*-
"""
Объединённый пайплайн: классификация зоны → выбор городской модели прогноза.

Исправление относительно исходного HTML: для sklearn-моделей выполняется
реальный predict по последней доступной строке признаков, а не np.zeros.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from .features import aggregate_windows_batch
from .classify import zone_to_representative_city
from .config import CLIMATE_ZONES


class WeatherForecastingPipeline:
    def __init__(
        self,
        classifier,
        forecasting_bundles: dict[str, dict],
        zone_mapping_mode: str = "canonical_city",
    ):
        """
        classifier: KNeighborsTimeSeriesClassifier (DTW) или sklearn-классификатор на aggregate_windows_batch
        forecasting_bundles[city] — результат train_city_forecasters(...)
        zone_mapping_mode: после класса зоны подставляем «канонический» город зоны
        """
        self.classifier = classifier
        self.forecasting_bundles = forecasting_bundles
        self.zone_mapping_mode = zone_mapping_mode

    def classify_zone(self, window_3d: np.ndarray) -> tuple[str, dict[str, float]]:
        """window_3d: shape (n_steps, n_features) или (1, n_steps, n_features)."""
        w = np.asarray(window_3d, dtype=np.float64)
        if w.ndim == 3:
            w = w[0]

        if isinstance(self.classifier, KNeighborsTimeSeriesClassifier):
            X_ts = w.reshape(1, w.shape[0], w.shape[1])
            zone = str(self.classifier.predict(X_ts)[0])
            confidence = {zone: 1.0}
            return zone, confidence

        x_agg = aggregate_windows_batch(w.reshape(1, *w.shape))
        zone = self.classifier.predict(x_agg)[0]
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(x_agg)[0]
            classes = self.classifier.classes_
            confidence = {str(c): float(p) for c, p in zip(classes, proba)}
        else:
            confidence = {str(zone): 1.0}
        return str(zone), confidence

    def zone_to_city(self, zone: str) -> str:
        if self.zone_mapping_mode == "canonical_city":
            return zone_to_representative_city(zone)
        return CLIMATE_ZONES[zone][0]

    def forecast_sklearn_scalar(self, city: str, model, horizon: int) -> np.ndarray:
        """
        Модель предсказывает температуру ровно через H дней для текущего состояния.
        Возвращаем массив длины horizon константой (явное ограничение постановки).
        Для покадрового месячного ряда используйте Prophet отдельно.
        """
        bundle = self.forecasting_bundles[city]
        tail = bundle["inference_tail_df"]
        feature_cols = bundle["feature_cols"]
        last_x = tail[feature_cols].values[-1].reshape(1, -1)
        y_hat = float(model.predict(last_x)[0])
        return np.full(horizon, y_hat, dtype=np.float64)

    def forecast_temperature(self, city: str, horizon: int = 30) -> np.ndarray:
        bundle = self.forecasting_bundles[city]
        models = bundle["models"]

        if "Prophet" in models:
            m = models["Prophet"]
            fut = m.make_future_dataframe(periods=horizon, include_history=False)
            fc = m.predict(fut)
            return fc["yhat"].values[:horizon].astype(np.float64)

        for key in ("LightGBM", "RandomForest"):
            if key in models:
                return self.forecast_sklearn_scalar(city, models[key], horizon)

        raise RuntimeError(f"Нет подходящей модели для города {city}")

    def predict(self, window_3d: np.ndarray, horizon: int = 30) -> dict[str, Any]:
        zone, conf = self.classify_zone(window_3d)
        city = self.zone_to_city(zone)
        temps = self.forecast_temperature(city, horizon=horizon)
        return {
            "predicted_zone": zone,
            "forecast_city_model": city,
            "zone_confidence": conf,
            "forecast_temperature": temps,
            "horizon": horizon,
        }
