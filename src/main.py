# -*- coding: utf-8 -*-
"""
Точка входа: загрузка данных → классификация зоны климата → прогноз по городам.

Запуск из корня проекта:
    python -m src.main
    python -m src.main --figures          # EDA + модели
    python -m src.main --eda-only         # только графики и ADF
    python -m src.eda_report              # то же EDA отдельной командой
    python -m src.main --dtw --prophet    # полный медленный прогон
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

from .config import DATA_DIR, RESULTS_DIR
from .data_loader import load_or_demo
from .eda_report import generate_eda_report
from .classify import prepare_classification_windows, time_based_split, train_classifiers
from .forecast import pick_best_sklearn_model, train_all_cities
from .pipeline import WeatherForecastingPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtw", action="store_true", help="обучить KNN-DTW (долго)")
    parser.add_argument("--prophet", action="store_true", help="обучить Prophet")
    parser.add_argument(
        "--window-stride",
        type=int,
        default=7,
        help="шаг скольжения окна классификации (1=все окна, медленнее)",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="построить EDA и сохранить графики в results/figures перед обучением",
    )
    parser.add_argument(
        "--eda-only",
        action="store_true",
        help="только EDA (графики + ADF), без моделей",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка данных...")
    data = load_or_demo(DATA_DIR)
    if not data:
        raise SystemExit("Нет данных.")

    if args.figures or args.eda_only:
        print("\n--- EDA: сохранение графиков в results/figures ---")
        fig_dir = generate_eda_report(data=data, results_dir=RESULTS_DIR)
        print(f"Сохранено в: {fig_dir}")
        print(f"Краткий индекс: {RESULTS_DIR / 'EDA_REPORT.md'}")
        if args.eda_only:
            return

    print("\n--- Этап 1: классификация типа климата (метки зон, временной сплит) ---")
    clf_report = train_classifiers(
        data,
        window_size=30,
        label_mode="zone",
        test_ratio=0.2,
        skip_knn=not args.dtw,
        window_stride=args.window_stride,
    )
    rf_bundle = clf_report["random_forest_agg"]
    knn_model = clf_report.get("knn_dtw", {}).get("model")
    if args.dtw and knn_model is not None:
        clf_model = knn_model
        print(
            "\n[Требование ТЗ по классификации] Сквозной пайплайн использует "
            "KNeighborsTimeSeries с метрикой DTW (tslearn)."
        )
    else:
        clf_model = rf_bundle["model"]
        if args.dtw:
            print(
                "\n[Внимание] Флаг --dtw указан, но модель KNN-DTW недоступна — "
                "в пайплайне остаётся RandomForest (агрегированное окно)."
            )
        else:
            print(
                "\n[Внимание] Запуск без --dtw: DTW пропущен, RandomForest в пайплайне. "
                "Для формулировки преподавателя (специализированные методы TSC) выполните: "
                "python -m src.main --dtw"
            )
        print(
            "RandomForest ниже остаётся как дополнительный baseline для сравнения в отчёте, "
            "но основным для защиты по тексту ТЗ лучше считать KNN-DTW."
        )

    print("\n--- Этап 2: прогноз t+30 по каждому городу ---")
    forecast_bundles = train_all_cities(data, horizon=30, use_prophet=args.prophet)

    print("\nМетрики прогноза (тестовый год по умолчанию 2025):")
    for city, b in forecast_bundles.items():
        print(f"  {city}: {b['metrics']}")

    # Пайплайн: для каждой зоны используем модель «первого» города из конфига
    pipeline_bundles = {}
    for city, bundle in forecast_bundles.items():
        model, name = pick_best_sklearn_model(bundle)
        if model is not None:
            pipeline_bundles[city] = bundle
        else:
            print(f"Пропуск города без sklearn-модели: {city}")

    print("\n--- Этап 3: сквозной пайплайн (демо на последнем окне) ---")
    Xw, y_zone, ends = prepare_classification_windows(
        data, window_size=30, label_mode="zone", window_stride=args.window_stride
    )
    _, X_test, _, y_test = time_based_split(Xw, y_zone, ends, test_ratio=0.2)
    if len(X_test) > 0:
        pipe = WeatherForecastingPipeline(clf_model, pipeline_bundles)
        sample = X_test[-1]
        res = pipe.predict(sample, horizon=30)
        print("Истинная зона (последнее тестовое окно):", y_test[-1])
        print("Предсказанная зона:", res["predicted_zone"])
        print("Город модели прогноза:", res["forecast_city_model"])
        print("Пример прогноза (первые 5 значений):", res["forecast_temperature"][:5])


if __name__ == "__main__":
    main()
