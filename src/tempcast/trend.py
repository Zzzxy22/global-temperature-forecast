"""长期趋势分析：各时期升温速率、最热年份、相对 19 世纪末的升温幅度。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# 气候学上常用的分段：早期升温、二战后停滞、现代快速升温
PERIODS = [(1880, 1939), (1940, 1969), (1970, 1999), (2000, None)]
EARLY_BASELINE = (1880, 1900)


def warming_rate(annual: pd.Series, start: int, end: int | None = None) -> dict:
    """OLS 线性趋势，单位 °C/十年，附 95% 置信区间。

    注意：年际自相关会让标准误偏小，置信区间应视为偏乐观的下限。
    """
    seg = annual.loc[start:end]
    res = stats.linregress(seg.index.to_numpy(float), seg.to_numpy(float))
    t = stats.t.ppf(0.975, len(seg) - 2)
    return {
        "period": f"{seg.index.min()}–{seg.index.max()}",
        "start": int(seg.index.min()),
        "end": int(seg.index.max()),
        "rate_per_decade": res.slope * 10,
        "ci_low": (res.slope - t * res.stderr) * 10,
        "ci_high": (res.slope + t * res.stderr) * 10,
        "r2": res.rvalue**2,
        "n_years": len(seg),
    }


def period_rates(annual: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([warming_rate(annual, s, e) for s, e in PERIODS])


def warmest_years(annual: pd.Series, n: int = 10) -> pd.DataFrame:
    top = annual.sort_values(ascending=False).head(n)
    return pd.DataFrame(
        {"rank": range(1, len(top) + 1), "year": top.index, "anomaly_c": top.to_numpy()}
    )


def decade_means(annual: pd.Series) -> pd.Series:
    d = (annual.index // 10) * 10
    return annual.groupby(d).mean().rename("anomaly_c").rename_axis("decade")


def relative_to_early(annual: pd.Series) -> pd.Series:
    """相对 1880–1900 平均的距平（常被用作“工业化前”的近似）。"""
    base = annual.loc[EARLY_BASELINE[0] : EARLY_BASELINE[1]].mean()
    return (annual - base).rename("vs_1880_1900")


def summarize(annual: pd.Series, monthly: pd.Series) -> dict:
    rel = relative_to_early(annual)
    rolling = monthly.rolling(12).mean()
    return {
        "first_month": monthly.index.min(),
        "last_month": monthly.index.max(),
        "n_months": len(monthly),
        "last_full_year": int(annual.index.max()),
        "last_year_anomaly": float(annual.iloc[-1]),
        "last_year_vs_early": float(rel.iloc[-1]),
        "warmest_year": int(annual.idxmax()),
        "warmest_year_anomaly": float(annual.max()),
        "warmest_year_vs_early": float(rel.max()),
        "warmest_month": monthly.idxmax(),
        "warmest_month_anomaly": float(monthly.max()),
        "latest_12m_mean": float(rolling.iloc[-1]),
        "top10_since_2010": int((warmest_years(annual).year >= 2010).sum()),
        "early_baseline": float(annual.loc[EARLY_BASELINE[0] : EARLY_BASELINE[1]].mean()),
    }
