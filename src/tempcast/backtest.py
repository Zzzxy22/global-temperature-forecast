"""滚动起点回测（rolling-origin evaluation）与经验预测区间。

做法：从 `start` 开始，每隔 `step` 个月（默认每个月）设一个预测起点；在起点之前的数据上
训练模型，预测之后 1..H 个月，并与真实值比较。这样每个模型都只用“当时”
能看到的数据，避免信息泄露。
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .models import Forecaster


def rolling_origin(
    series: pd.Series,
    models: list[Forecaster],
    horizon: int = 24,
    start: str = "2000-01-01",
    step: int = 1,
) -> pd.DataFrame:
    y = series.to_numpy(dtype=float)
    first = series.index.get_loc(pd.Timestamp(start))
    origins = range(first, len(y) - horizon + 1, step)
    rows = []
    for o in origins:
        train, actual = y[:o], y[o : o + horizon]
        for proto in models:
            m = copy.deepcopy(proto).fit(train)
            pred = m.predict(horizon)
            for h in range(horizon):
                rows.append(
                    {
                        "model": proto.name,
                        "origin": series.index[o],
                        "h": h + 1,
                        "actual": actual[h],
                        "pred": pred[h],
                    }
                )
    df = pd.DataFrame(rows)
    df["error"] = df["actual"] - df["pred"]
    return df


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """每个模型的 MAE / RMSE（全部预测步长平均）以及 1、12、24 个月的 MAE。"""
    g = bt.assign(abs_err=bt["error"].abs(), sq_err=bt["error"] ** 2)
    overall = g.groupby("model").agg(mae=("abs_err", "mean"), rmse=("sq_err", "mean"))
    overall["rmse"] = np.sqrt(overall["rmse"])
    by_h = g.pivot_table(index="model", columns="h", values="abs_err", aggfunc="mean")
    max_h = int(by_h.columns.max())
    for h in sorted({1, min(12, max_h), max_h}):
        overall[f"mae_h{h}"] = by_h[h]
    overall["bias"] = g.groupby("model")["error"].mean()
    overall["n_origins"] = g.groupby("model")["origin"].nunique()
    return overall.sort_values("mae").reset_index()


def mae_by_horizon(bt: pd.DataFrame) -> pd.DataFrame:
    return bt.assign(abs_err=bt["error"].abs()).pivot_table(
        index="h", columns="model", values="abs_err", aggfunc="mean"
    )


def error_quantiles(
    bt: pd.DataFrame, model: str, levels=(0.8, 0.95), pool: int = 2
) -> pd.DataFrame:
    """按预测步长计算回测误差的经验分位数，用于构造预测区间。

    预测值 + 误差分位数 = 区间。这种方法不假设误差服从正态分布，
    且自动反映“越往后越不确定”。每个步长只有几十个样本，因此合并
    相邻 ±`pool` 个步长的误差一起估计，使区间随步长平滑变化。
    """
    e = bt[bt["model"] == model]
    hs = sorted(e["h"].unique())
    rows = {}
    for h in hs:
        errs = e.loc[(e["h"] - h).abs() <= pool, "error"]
        row = {}
        for lv in levels:
            k = int(lv * 100)
            row[f"lo{k}"] = errs.quantile((1 - lv) / 2)
            row[f"hi{k}"] = errs.quantile(1 - (1 - lv) / 2)
        rows[h] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "h"
    return out


def coverage(bt: pd.DataFrame, model: str, level: float = 0.8) -> float:
    """区间在回测中的实际覆盖率（样本内，偏乐观，仅供参考）。"""
    q = error_quantiles(bt, model, (level,))
    e = bt[bt["model"] == model].join(q, on="h")
    k = int(level * 100)
    inside = (e["error"] >= e[f"lo{k}"]) & (e["error"] <= e[f"hi{k}"])
    return float(inside.mean())


def forecast(
    series: pd.Series, model: Forecaster, quantiles: pd.DataFrame, horizon: int = 24
) -> pd.DataFrame:
    """在全部数据上重新训练，输出未来 `horizon` 个月的点预测与区间。"""
    m = copy.deepcopy(model).fit(series.to_numpy(dtype=float))
    pred = m.predict(horizon)
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    out = pd.DataFrame({"forecast": pred}, index=idx)
    q = quantiles.reindex(range(1, horizon + 1)).to_numpy()
    for j, col in enumerate(quantiles.columns):
        out[col] = pred + q[:, j]
    out.index.name = "month"
    return out
