# -*- coding: utf-8 -*-
"""
Классификация типа климата по окну ряда (tslearn KNN-DTW + опционально RF на статистиках окна).
Разбиение по времени, без случайного смешивания соседних окон из одного города.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from .config import CITY_TO_ZONE
from .features import aggregate_windows_batch, prepare_features_for_classification


def prepare_classification_windows(
    data: dict[str, pd.DataFrame],
    window_size: int = 30,
    label_mode: str = "zone",
    window_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X: (n_samples, window, n_features), y: метки (город или климатическая зона).
    Также возвращаем массив дат конца окна (для временного сплита).
    """
    xs: list[np.ndarray] = []
    ys: list[str] = []
    end_dates: list[pd.Timestamp] = []

    for city, df in data.items():
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df = prepare_features_for_classification(df)

        label = CITY_TO_ZONE[city] if label_mode == "zone" else city

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_matrix = df[numeric_cols].values
        timestamps = df["timestamp"].values

        for i in range(0, len(feature_matrix) - window_size + 1, window_stride):
            window = feature_matrix[i : i + window_size]
            if np.isnan(window).any():
                continue
            xs.append(window)
            ys.append(label)
            end_dates.append(pd.Timestamp(timestamps[i + window_size - 1]))

    return np.array(xs, dtype=np.float64), np.array(ys), np.array(end_dates)


def time_based_split(
    X: np.ndarray,
    y: np.ndarray,
    end_dates: np.ndarray,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Последняя доля временной шкалы → тест (глобально по всем городам)."""
    order = np.argsort(pd.to_datetime(end_dates))
    X, y = X[order], y[order]
    cut = int(len(X) * (1 - test_ratio))
    return X[:cut], X[cut:], y[:cut], y[cut:]


def train_classifiers(
    data: dict[str, pd.DataFrame],
    window_size: int = 30,
    label_mode: str = "zone",
    test_ratio: float = 0.2,
    random_state: int = 42,
    skip_knn: bool = False,
    window_stride: int = 7,
) -> dict:
    """
    Обучает KNN-DTW и RandomForest на агрегированных статистиках окна.
    Для стратифицированного быстрого прототипа можно передать use_stratified_sklearn_split=True ниже.
    """
    X, y, ends = prepare_classification_windows(
        data, window_size, label_mode, window_stride=window_stride
    )

    # Временное разбиение (корректнее для рядов)
    X_train, X_test, y_train, y_test = time_based_split(X, y, ends, test_ratio)

    results: dict = {}

    print("--- KNeighborsTimeSeries (DTW) ---")
    if skip_knn:
        results["knn_dtw"] = {"skipped": True}
        print("пропуск DTW (skip_knn=True)")
    else:
        knn = KNeighborsTimeSeriesClassifier(n_neighbors=5, metric="dtw")
        try:
            max_fit = 800
            if len(X_train) > max_fit:
                idx = np.random.RandomState(42).choice(len(X_train), max_fit, replace=False)
                X_knn_fit, y_knn_fit = X_train[idx], y_train[idx]
                print(f"(ускорение демо: DTW fit на {max_fit} из {len(X_train)} окон)")
            else:
                X_knn_fit, y_knn_fit = X_train, y_train
            knn.fit(X_knn_fit, y_knn_fit)
            max_eval = 400
            if len(X_test) > max_eval:
                idx_te = np.random.RandomState(43).choice(len(X_test), max_eval, replace=False)
                X_eval, y_eval = X_test[idx_te], y_test[idx_te]
                print(f"(DTW predict на подвыборке теста: {max_eval})")
            else:
                X_eval, y_eval = X_test, y_test
            pred_k = knn.predict(X_eval)
            results["knn_dtw"] = {
                "accuracy": accuracy_score(y_eval, pred_k),
                "f1_macro": f1_score(y_eval, pred_k, average="macro", zero_division=0),
                "y_test": y_eval,
                "y_pred": pred_k,
                "model": knn,
            }
            print(f"Accuracy: {results['knn_dtw']['accuracy']:.4f}")
            print(classification_report(y_eval, pred_k, zero_division=0))
        except Exception as exc:
            print(f"Ошибка KNN-DTW: {exc}")
            results["knn_dtw"] = {"error": str(exc)}

    print("--- RandomForest на статистиках окна ---")
    print(f"Размер выборки классификации: train={len(X_train)}, test={len(X_test)}")
    X_train_agg = aggregate_windows_batch(X_train)
    X_test_agg = aggregate_windows_batch(X_test)
    rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=4)
    rf.fit(X_train_agg, y_train)
    pred_rf = rf.predict(X_test_agg)
    proba_rf = rf.predict_proba(X_test_agg)
    results["random_forest_agg"] = {
        "accuracy": accuracy_score(y_test, pred_rf),
        "f1_macro": f1_score(y_test, pred_rf, average="macro", zero_division=0),
        "model": rf,
        "y_test": y_test,
        "y_pred": pred_rf,
        "y_proba": proba_rf,
        "X_train_agg": X_train_agg,
        "y_train": y_train,
    }
    print(f"Accuracy: {results['random_forest_agg']['accuracy']:.4f}")
    print(classification_report(y_test, pred_rf, zero_division=0))

    return results


def zone_to_representative_city(zone: str) -> str:
    """Для второго этапа (модель по городу): выбираем канонический город зоны."""
    from .config import CLIMATE_ZONES

    return CLIMATE_ZONES[zone][0]
