# -*- coding: utf-8 -*-
"""
EDA и сохранение графиков в results/figures (без интерактивных окон).

Запуск:
    python -m src.eda_report
    python -m src.main --figures   # сначала графики, затем модели
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

# В Jupyter/Colab нужен интерактивный бэкенд; Agg только вне IPython (CLI).
try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from .config import CITY_FILE_SLUG, CITY_TO_ZONE, DATA_DIR, RESULTS_DIR
from .data_loader import load_or_demo
from .features import prepare_features_for_classification


def _finalize_figure(fig: plt.Figure, path: Path, *, dpi: int = 120) -> None:
    """Сохранить PNG; в ноутбуке — показать вывод ячейки (Colab/Jupyter)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        from IPython import get_ipython as _gi
        from IPython.display import display

        if _gi() is not None:
            display(fig)
    except Exception:
        pass
    plt.close(fig)


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    # Кириллица на Windows
    for fam in ("Segoe UI", "Arial", "DejaVu Sans"):
        try:
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            continue


def file_slug_city(name: str) -> str:
    return CITY_FILE_SLUG.get(name, "".join(c if c.isalnum() else "_" for c in name))


def plot_city_panel(df: pd.DataFrame, city: str, out_dir: Path) -> None:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()[:8]
    if not numeric:
        return

    n = len(numeric)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, col in enumerate(numeric):
        ax = axes[idx]
        ax.plot(df.index, df[col], linewidth=0.6, alpha=0.85)
        ax.set_title(f"{city}: {col}")
        ax.set_xlabel("Дата")
    for j in range(len(numeric), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"{city}: временные ряды (до 8 признаков)", y=1.01)
    plt.tight_layout()
    _finalize_figure(fig, out_dir / f"{file_slug_city(city)}_timeseries.png")

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, col in enumerate(numeric):
        ax = axes[idx]
        s = df[col].dropna()
        if len(s) > 1:
            ax.hist(s, bins=50, edgecolor="black", alpha=0.75)
        ax.set_title(f"{col}")
    for j in range(len(numeric), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"{city}: распределения", y=1.01)
    plt.tight_layout()
    _finalize_figure(fig, out_dir / f"{file_slug_city(city)}_histograms.png")

    if "temperature_2m" in df.columns:
        month_df = df.copy()
        month_df["month"] = month_df.index.month
        fig, ax = plt.subplots(figsize=(12, 5))
        month_df.boxplot(column="temperature_2m", by="month", ax=ax)
        ax.set_title(f"{city}: температура по месяцам")
        ax.set_xlabel("Месяц")
        plt.suptitle("")
        plt.tight_layout()
        _finalize_figure(fig, out_dir / f"{file_slug_city(city)}_temp_by_month.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title(f"{city}: корреляции")
    plt.tight_layout()
    _finalize_figure(fig, out_dir / f"{file_slug_city(city)}_correlation.png")


def plot_cross_city(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    monthly_rows = []
    for city, df in data.items():
        t = df.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"])
        t["month"] = t["timestamp"].dt.month
        g = t.groupby("month")["temperature_2m"].mean()
        for m, v in g.items():
            monthly_rows.append({"city": city, "month": m, "avg_temp": v})
    monthly_df = pd.DataFrame(monthly_rows)

    fig, ax = plt.subplots(figsize=(14, 7))
    for city in monthly_df["city"].unique():
        sub = monthly_df[monthly_df["city"] == city]
        ax.plot(sub["month"], sub["avg_temp"], marker="o", label=city, linewidth=2)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Средняя температура, °C")
    ax.set_title("Годовой ход температуры по городам")
    ax.legend()
    plt.tight_layout()
    _finalize_figure(fig, out_dir / "cross_city_monthly_profile.png")

    rng = np.random.default_rng(0)
    rows = []
    max_per_city = 800
    for city, df in data.items():
        vals = df["temperature_2m"].values
        if len(vals) > max_per_city:
            vals = rng.choice(vals, size=max_per_city, replace=False)
        for v in vals:
            rows.append({"city": city, "temperature": v})
    temp_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=temp_df, x="city", y="temperature", ax=ax, inner="box")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_title("Распределение температуры по городам")
    plt.tight_layout()
    _finalize_figure(fig, out_dir / "cross_city_temp_violin.png")


def plot_pca_climate_zones(data: dict[str, pd.DataFrame], out_dir: Path, sample_per_city: int = 120) -> None:
    rng = np.random.default_rng(42)
    parts_x: list[np.ndarray] = []
    parts_y: list[str] = []
    parts_zone: list[str] = []

    for city, df in data.items():
        df_p = prepare_features_for_classification(df.copy())
        mask = df_p["month"].isin([6, 7, 8]) if "month" in df_p.columns else slice(None)
        num = df_p.loc[mask].select_dtypes(include=[np.number]).dropna()
        if len(num) == 0:
            continue
        if len(num) > sample_per_city:
            idx = rng.choice(len(num), sample_per_city, replace=False)
            num = num.iloc[idx]
        parts_x.append(num.values)
        parts_y.extend([city] * len(num))
        parts_zone.extend([CITY_TO_ZONE.get(city, "?")] * len(num))

    if not parts_x:
        return

    X = np.vstack(parts_x)
    zones = np.array(parts_zone)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(11, 7))
    for z in np.unique(zones):
        m = zones == z
        ax.scatter(xy[m, 0], xy[m, 1], alpha=0.55, s=28, label=z)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA (летние точки): тип климата")
    ax.legend()
    plt.tight_layout()
    _finalize_figure(fig, out_dir / "pca_zones_summer.png")


def write_adf_summary(data: dict[str, pd.DataFrame], path: Path) -> None:
    lines = ["Тест Дики–Фуллера по temperature_2m (p < 0.05 → стационарность)", ""]
    for city, df in sorted(data.items()):
        s = pd.to_datetime(df["timestamp"])
        series = df.set_index(s)["temperature_2m"].dropna()
        if len(series) < 30:
            continue
        r = adfuller(series.values)
        lines.append(f"{city}: ADF={r[0]:.4f}, p-value={r[1]:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_eda_report(
    data: dict[str, pd.DataFrame] | None = None,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
) -> Path:
    """
    Строит все графики и краткий текстовый отчёт.
    Возвращает путь к каталогу с фигурами.
    """
    _setup_style()
    results_dir = Path(results_dir or RESULTS_DIR)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = data if data is not None else load_or_demo(data_dir or DATA_DIR)
    if not data:
        raise RuntimeError("Нет данных для EDA")

    for city, df in data.items():
        plot_city_panel(df, city, fig_dir)

    plot_cross_city(data, fig_dir)
    plot_pca_climate_zones(data, fig_dir)
    write_adf_summary(data, results_dir / "eda_adf_summary.txt")

    index_md = results_dir / "EDA_REPORT.md"
    lines = [
        "# EDA: сохранённые графики",
        "",
        f"Каталог PNG: `{fig_dir}`",
        "",
        "## Файлы",
        "",
        "- По каждому городу (латиница в имени файла, см. `CITY_FILE_SLUG` в `config.py`):",
        "  `*_timeseries.png`, `*_histograms.png`, `*_temp_by_month.png`, `*_correlation.png`",
        "- `cross_city_monthly_profile.png`, `cross_city_temp_violin.png` (violin — случайная подвыборка до 800 точек на город)",
        "- `pca_zones_summer.png` — разделимость **типов климата** (летняя выборка)",
        "- `eda_adf_summary.txt` — стационарность температуры (ADF)",
        "",
    ]
    index_md.write_text("\n".join(lines), encoding="utf-8")

    return fig_dir


def main_cli() -> None:
    generate_eda_report()
    print(f"Готово. Графики: {RESULTS_DIR / 'figures'}")
    print(f"Индекс: {RESULTS_DIR / 'EDA_REPORT.md'}")
    print(f"ADF: {RESULTS_DIR / 'eda_adf_summary.txt'}")


if __name__ == "__main__":
    main_cli()
