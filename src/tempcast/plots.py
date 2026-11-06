"""图表。文字使用英文，避免在没有中文字体的机器上出现乱码。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .data import MONTHS  # noqa: E402

# ---- 配色：经色盲安全校验的分类色、单色顺序色阶、蓝-灰-红发散色阶 ----
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9890"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_LIGHT, BLUE_MID, BLUE_DARK = "#b7d3f6", "#86b6ef", "#184f95"
MODEL_COLORS = {
    "AR + trend": BLUE,
    "Linear trend": ORANGE,
    "Ensemble": AQUA,
    "Damped Holt": YELLOW,
    "12-month mean": NEUTRAL,
    "Naive": NEUTRAL,
}
MODEL_STYLES = {"12-month mean": (0, (4, 2)), "Naive": (0, (1, 1.5))}
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#0d366b", "#3987e5", "#f0efec", "#e66767", "#8a1f1f"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "text.color": TEXT,
    }
)

YLABEL = "Anomaly vs 1951–1980 (°C)"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_history(monthly: pd.Series, annual: pd.Series, path: Path) -> Path:
    rolling = monthly.rolling(12, center=True).mean()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(monthly.index, monthly, color=BLUE_LIGHT, linewidth=0.7, label="monthly", zorder=2)
    ax.plot(rolling.index, rolling, color=BLUE_DARK, linewidth=2, label="12-month mean", zorder=3)
    ax.axhline(0, color=TEXT_2, linewidth=0.8, zorder=1)
    yr = int(annual.idxmax())
    peak = pd.Timestamp(f"{yr}-07-01")
    ax.annotate(
        f"{yr}: warmest year\n+{annual.max():.2f} °C",
        xy=(peak, annual.max()),
        xytext=(pd.Timestamp("2000-01-01"), annual.max() + 0.15),
        fontsize=9,
        color=TEXT,
        arrowprops={"arrowstyle": "-", "color": TEXT_2, "linewidth": 0.8},
    )
    ax.set_ylabel(YLABEL)
    ax.set_xlim(monthly.index.min(), monthly.index.max() + pd.DateOffset(months=6))
    ax.xaxis.set_major_locator(mdates.YearLocator(20))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    ax.set_title(
        f"Global surface temperature anomaly, {monthly.index.min():%Y}–{monthly.index.max():%Y}"
    )
    return _save(fig, path)


def plot_heatmap(monthly: pd.Series, path: Path) -> Path:
    grid = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "v": monthly.to_numpy()}
    ).pivot(index="month", columns="year", values="v")
    fig, ax = plt.subplots(figsize=(12, 3.6))
    lim = float(np.nanmax(np.abs(grid.to_numpy())))
    im = ax.imshow(
        grid,
        aspect="auto",
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
        interpolation="nearest",
        extent=[grid.columns.min() - 0.5, grid.columns.max() + 0.5, 12.5, 0.5],
    )
    ax.set_yticks(range(1, 13), MONTHS)
    ax.tick_params(axis="y", labelsize=7, length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.outline.set_visible(False)
    cbar.set_label("°C vs 1951–1980", color=TEXT_2)
    ax.set_title("Every month since 1880 (blank cells: not yet published)")
    return _save(fig, path)


def plot_rates(rates: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(rates))
    vals = rates["rate_per_decade"].to_numpy()
    err = np.vstack([vals - rates["ci_low"], rates["ci_high"] - vals])
    ax.bar(x, vals, color=ORANGE, width=0.6, zorder=2)
    ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=4, zorder=3)
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    for xi, v, hi, lo in zip(x, vals, rates["ci_high"], rates["ci_low"]):
        y = hi + 0.012 if v >= 0 else lo - 0.012
        ax.text(xi, y, f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_xticks(x, rates["period"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("°C per decade")
    ax.set_ylim(min(-0.1, rates["ci_low"].min() - 0.04), rates["ci_high"].max() + 0.05)
    ax.set_title("Warming rate by period (linear trend, 95% CI)")
    return _save(fig, path)


def plot_backtest(mae_h: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = mae_h.mean().sort_values().index
    for name in order:
        ax.plot(
            mae_h.index,
            mae_h[name],
            color=MODEL_COLORS.get(name, NEUTRAL),
            linewidth=1.6 if name in MODEL_STYLES else 2,
            linestyle=MODEL_STYLES.get(name, "-"),
            label=name,
        )
    ax.set_xlabel("Forecast horizon (months ahead)")
    ax.set_ylabel("Mean absolute error (°C)")
    ax.set_xlim(1, mae_h.index.max())
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="sorted by\naverage error", title_fontsize=8)
    ax.set_title("Backtest: how far off is each model, by horizon?")
    return _save(fig, path)


def plot_forecast(
    monthly: pd.Series, fc: pd.DataFrame, model_name: str, path: Path, years_back: int = 10
) -> Path:
    hist = monthly.loc[monthly.index.max() - pd.DateOffset(years=years_back) :]
    rolling = monthly.rolling(12).mean().loc[hist.index]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(fc.index, fc["lo95"], fc["hi95"], color=BLUE_LIGHT, linewidth=0, label="95% interval")
    ax.fill_between(fc.index, fc["lo80"], fc["hi80"], color=BLUE_MID, linewidth=0, label="80% interval")
    ax.plot(hist.index, hist, color=TEXT_2, linewidth=1, label="observed (monthly)")
    ax.plot(rolling.index, rolling, color=TEXT, linewidth=2, label="observed 12-month mean")
    joined = pd.concat([hist.iloc[-1:], fc["forecast"]])
    ax.plot(joined.index, joined, color=ORANGE, linewidth=2, label=f"forecast ({model_name})")
    ax.axvline(hist.index[-1], color=TEXT_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(
        hist.index[-1], ax.get_ylim()[1], "  forecast →", va="top", ha="left", fontsize=9, color=TEXT_2
    )
    ax.set_ylabel(YLABEL)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", ncol=2)
    ax.set_title(f"Next {len(fc)} months: point forecast with empirical prediction intervals")
    return _save(fig, path)
