import numpy as np
import pandas as pd
import pytest

from tempcast.backtest import coverage, error_quantiles, forecast, rolling_origin, score
from tempcast.models import ARTrend, DampedHolt, LinearTrend, Naive, RecentMean, all_models
from tempcast.trend import warming_rate


@pytest.fixture
def linear_series():
    t = np.arange(400, dtype=float)
    return 0.002 * t + 0.1


def test_baselines(linear_series):
    assert np.all(Naive().fit(linear_series).predict(3) == linear_series[-1])
    assert RecentMean().fit(linear_series).predict(2)[0] == pytest.approx(linear_series[-12:].mean())


def test_linear_trend_recovers_exact_line(linear_series):
    pred = LinearTrend(window=100).fit(linear_series).predict(5)
    expected = 0.002 * np.arange(400, 405) + 0.1
    assert np.allclose(pred, expected)


def test_ar_trend_extrapolates_trend(linear_series):
    rng = np.random.default_rng(0)
    y = linear_series + rng.normal(0, 0.01, len(linear_series))
    pred = ARTrend(lags=6, window=300).fit(y).predict(24)
    expected = 0.002 * np.arange(400, 424) + 0.1
    assert np.abs(pred - expected).max() < 0.05


def test_damped_holt_parameters_in_bounds(linear_series):
    m = DampedHolt().fit(linear_series)
    assert 0 < m.alpha_ < 1 and 0 < m.beta_ < 1 and 0.8 <= m.phi_ < 1
    assert len(m.predict(12)) == 12


def test_all_models_return_finite_forecasts():
    y = np.sin(np.arange(300) / 5) * 0.1 + np.arange(300) * 0.001
    for m in all_models():
        p = m.fit(y).predict(24)
        assert p.shape == (24,) and np.isfinite(p).all(), m.name


def test_backtest_has_no_leakage_and_scores():
    idx = pd.date_range("1990-01-01", periods=240, freq="MS")
    s = pd.Series(np.arange(240) * 0.01, index=idx)
    bt = rolling_origin(s, [Naive()], horizon=6, start="2005-01-01", step=12)
    # Naive 的预测 = 起点前最后一个值，所以误差 = h × 0.01
    assert np.allclose(bt["error"], bt["h"] * 0.01)
    sc = score(bt)
    assert sc.loc[0, "model"] == "Naive"
    q = error_quantiles(bt, "Naive", levels=(0.8,))
    assert list(q.columns) == ["lo80", "hi80"]
    assert 0 <= coverage(bt, "Naive", 0.8) <= 1
    fc = forecast(s, Naive(), q, horizon=6)
    assert fc.index[0] == pd.Timestamp("2010-01-01")
    assert (fc["lo80"] <= fc["hi80"]).all()


def test_warming_rate_units():
    a = pd.Series(np.arange(50) * 0.02, index=range(1970, 2020))
    r = warming_rate(a, 1970)
    assert r["rate_per_decade"] == pytest.approx(0.2)
