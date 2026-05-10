# -*- coding: utf-8 -*-
"""Константы: города, климатические типы (≥2 города на тип согласно ТЗ)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

CITIES = [
    "Москва",
    "Сочи",
    "Благовещенск",
    "Находка",
    "Геленджик",
    "Санкт-Петербург",
]

# Тип климата → города (задание: классификация по типу климата, не менее 2 городов в типе)
CLIMATE_ZONES = {
    "Умеренный": ["Москва", "Санкт-Петербург"],
    "Субтропический_приморский": ["Сочи", "Геленджик"],
    "Континентально_муссонный": ["Благовещенск", "Находка"],
}

CITY_TO_ZONE = {
    city: zone for zone, cities in CLIMATE_ZONES.items() for city in cities
}

# Латиница для имён файлов отчётов (EDA), чтобы не ломались пути в Windows/Git
CITY_FILE_SLUG = {
    "Москва": "moscow",
    "Сочи": "sochi",
    "Благовещенск": "blagoveshchensk",
    "Находка": "nakhodka",
    "Геленджик": "gelendzhik",
    "Санкт-Петербург": "saint_petersburg",
}
