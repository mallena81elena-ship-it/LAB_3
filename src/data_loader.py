# -*- coding: utf-8 -*-
"""Загрузка parquet по городам и синтетические данные для проверки пайплайна."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .config import CITIES, DATA_DIR


def _normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит входной фрейм к единому формату:
    обязательная колонка `timestamp` с datetime.
    Поддерживает данные, где время хранится в индексе (`time`) или в колонке `time`.
    """
    out = df.copy()

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        return out

    if "time" in out.columns:
        out = out.rename(columns={"time": "timestamp"})
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        return out

    # Частый случай parquet: DatetimeIndex с именем "time"
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        if "time" in out.columns:
            out = out.rename(columns={"time": "timestamp"})
        elif "index" in out.columns:
            out = out.rename(columns={"index": "timestamp"})
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        return out

    raise ValueError("Не найдена временная ось: ожидается timestamp/time или DatetimeIndex")


def load_all_data(data_path: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """
    Загрузка всех parquet для городов из CITIES.
    Шаблон имени: {Город}_YYYY-MM-DD_YYYY-MM-DD.parquet
    """
    data_path = Path(data_path or DATA_DIR)
    all_data: dict[str, pd.DataFrame] = {}

    for city in CITIES:
        city_files = sorted(data_path.glob(f"{city}_*.parquet"))
        if not city_files:
            continue
        chunks = [_normalize_time_column(pd.read_parquet(f)) for f in city_files]
        all_data[city] = pd.concat(chunks, ignore_index=True)
    return all_data


def create_demo_data(
    start: str = "2019-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Демо-ряды с разными сезонными параметрами (если реальных файлов нет)."""
    rng = np.random.default_rng(seed)
    date_range = pd.date_range(start, end, freq="D")
    data: dict[str, pd.DataFrame] = {}

    params = {
        "Москва": (5.5, 28.0, 2.0),
        "Сочи": (14.0, 20.0, 4.5),
        "Благовещенск": (1.5, 45.0, 1.5),
        "Находка": (5.0, 35.0, 2.5),
        "Геленджик": (12.0, 22.0, 3.0),
        "Санкт-Петербург": (5.0, 25.0, 2.5),
    }

    for city in CITIES:
        temp_mean, temp_amp, precip_mean = params[city]
        day_of_year = date_range.dayofyear.values
        temp = temp_mean + temp_amp * np.sin(
            2 * np.pi * day_of_year / 365.25 - np.pi / 2
        )
        temp += rng.normal(0, 3.0, len(date_range))

        data[city] = pd.DataFrame(
            {
                "timestamp": date_range,
                "temperature_2m": temp,
                "relative_humidity_2m": rng.uniform(50, 90, len(date_range)),
                "precipitation": rng.exponential(precip_mean, len(date_range)),
                "rain": rng.exponential(precip_mean * 0.7, len(date_range)),
                "snowfall": np.where(temp < 0, rng.exponential(0.5, len(date_range)), 0.0),
                "weathercode": rng.integers(0, 100, len(date_range)),
                "wind_speed_10m": rng.gamma(2.0, 2.0, len(date_range)),
                "surface_pressure": rng.normal(1013.0, 15.0, len(date_range)),
            }
        )
    return data


def load_or_demo(data_path: Path | str | None = None) -> dict[str, pd.DataFrame]:
    loaded = load_all_data(data_path)
    if len(loaded) >= len(CITIES):
        return loaded
    return create_demo_data()
