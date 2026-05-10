# Лабораторная работа 3 — классификация и прогноз метеоданных

Холина Елена
## Структура

```
LAB_3/
├── README.md
├── requirements.txt
├── lab3_solution.ipynb   ← ноутбук итоговая работа (полный сценарий 2.1–2.8)
├── scripts/
│   └── build_notebook.py ← пересборка ноутбука при необходимости
├── data/              ← отчеты Город_YYYY-MM-DD_YYYY-MM-DD.parquet
├── results/figures      - ГРАФИКИ из lab3
└── src/
    ├── config.py       # города и группировка по климатическим зонам (≥2 города в зоне)
    ├── data_loader.py  # загрузка parquet или синтетика для проверки
    ├── features.py     # признаки классификации и регрессии
    ├── classify.py     # RF на статистиках окна + опционально KNN-DTW (tslearn)
    ├── forecast.py     # LightGBM / RandomForest / опционально Prophet
    ├── pipeline.py     # объединение этапов
    ├── eda_report.py   # EDA: графики → results/figures, ADF → txt
    └── main.py         # точка входа
```

