"""预测模型。全部只依赖 numpy / scipy / scikit-learn，每个模型都实现

    fit(y: np.ndarray) -> self
    predict(h: int) -> np.ndarray   # 未来 1..h 个月的点预测
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge


class Forecaster:
    name = "base"

    def fit(self, y: np.ndarray) -> "Forecaster":
        raise NotImplementedError

    def predict(self, h: int) -> np.ndarray:
        raise NotImplementedError


class Naive(Forecaster):
    """最后一个观测值持续不变。"""

    name = "Naive"

    def fit(self, y):
        self.last_ = float(y[-1])
        return self

    def predict(self, h):
        return np.full(h, self.last_)


class RecentMean(Forecaster):
    """最近 12 个月的平均值（平滑掉单月噪声的基线）。"""

    name = "12-month mean"

    def __init__(self, window: int = 12):
        self.window = window

    def fit(self, y):
        self.level_ = float(np.mean(y[-self.window :]))
        return self

    def predict(self, h):
        return np.full(h, self.level_)


class LinearTrend(Forecaster):
    """对最近 `window` 个月做直线回归并外推。"""

    name = "Linear trend"

    def __init__(self, window: int = 360):
        self.window = window

    def fit(self, y):
        seg = y[-self.window :]
        t = np.arange(len(seg), dtype=float)
        self.slope_, self.intercept_ = np.polyfit(t, seg, 1)
        self.t_end_ = len(seg) - 1
        return self

    def predict(self, h):
        t = self.t_end_ + np.arange(1, h + 1)
        return self.intercept_ + self.slope_ * t


class DampedHolt(Forecaster):
    """阻尼趋势指数平滑（Holt 线性趋势 + 阻尼因子 φ）。

    参数 α（水平）、β（趋势）、φ（阻尼）通过最小化一步预测误差平方和估计。
    """

    name = "Damped Holt"

    def __init__(self, window: int = 600):
        self.window = window

    @staticmethod
    def _filter(y, alpha, beta, phi):
        level, trend = y[0], np.mean(np.diff(y[:13]))
        sse = 0.0
        for obs in y[1:]:
            pred = level + phi * trend
            err = obs - pred
            sse += err * err
            new_level = pred + alpha * err
            trend = phi * trend + beta * (new_level - level - phi * trend)
            level = new_level
        return level, trend, sse

    def fit(self, y):
        y = np.asarray(y[-self.window :], dtype=float)

        def loss(p):
            return self._filter(y, *p)[2]

        res = minimize(
            loss,
            x0=[0.5, 0.01, 0.95],
            bounds=[(0.01, 0.99), (0.0001, 0.3), (0.8, 0.999)],
            method="L-BFGS-B",
        )
        self.alpha_, self.beta_, self.phi_ = res.x
        self.level_, self.trend_, _ = self._filter(y, *res.x)
        return self

    def predict(self, h):
        damp = np.cumsum(self.phi_ ** np.arange(1, h + 1))
        return self.level_ + damp * self.trend_


class ARTrend(Forecaster):
    """带线性趋势的自回归模型：

        y_t = c + b·t + Σ φ_i · y_{t-i} + ε_t

    用 Ridge 回归估计（轻微正则化保证数值稳定），多步预测采用递推方式。
    长期预测会收敛到趋势线，短期则利用最近几个月的偏离（如厄尔尼诺）。
    """

    name = "AR + trend"

    def __init__(self, lags: int = 12, window: int = 480, alpha: float = 1.0):
        self.lags = lags
        self.window = window
        self.alpha = alpha

    def _design(self, y, t0):
        rows, target = [], []
        for i in range(self.lags, len(y)):
            rows.append(np.r_[t0 + i, y[i - self.lags : i][::-1]])
            target.append(y[i])
        return np.array(rows), np.array(target)

    def fit(self, y):
        y = np.asarray(y[-(self.window + self.lags) :], dtype=float)
        X, target = self._design(y, 0)
        # 时间以“年”为单位缩放，避免趋势系数被正则化压没
        self.t_scale_ = 12.0
        X[:, 0] /= self.t_scale_
        self.model_ = Ridge(alpha=self.alpha).fit(X, target)
        self.history_ = list(y[-self.lags :])
        self.t_next_ = len(y)
        return self

    def predict(self, h):
        hist = list(self.history_)
        out = []
        for k in range(h):
            x = np.r_[(self.t_next_ + k) / self.t_scale_, hist[::-1][: self.lags]]
            yhat = float(self.model_.predict(x[None, :])[0])
            out.append(yhat)
            hist.append(yhat)
        return np.array(out)


class Ensemble(Forecaster):
    """若干模型点预测的简单平均。"""

    name = "Ensemble"

    def __init__(self, members: list[Forecaster] | None = None):
        self.members = members

    def fit(self, y):
        self.members_ = [m.fit(y) for m in (self.members or [LinearTrend(), DampedHolt(), ARTrend()])]
        return self

    def predict(self, h):
        return np.mean([m.predict(h) for m in self.members_], axis=0)


def all_models() -> list[Forecaster]:
    return [Naive(), RecentMean(), LinearTrend(), DampedHolt(), ARTrend(), Ensemble()]
