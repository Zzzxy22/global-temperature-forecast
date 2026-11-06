import numpy as np
import pandas as pd
import pytest

from tempcast.data import annual_series, load_monthly, load_table, monthly_series


def test_bundled_data_loads():
    m = load_monthly()
    assert m.index[0] == pd.Timestamp("1880-01-01")
    assert m.index.freqstr == "MS"
    assert not m.isna().any()
    assert m.index.is_monotonic_increasing
    # 1951–1980 是基准期，平均距平应接近 0
    assert abs(m.loc["1951":"1980"].mean()) < 0.02


def test_trailing_missing_months_are_dropped(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text(
        "Land-Ocean: Global Means\n"
        "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D\n"
        "2000,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2,.65\n"
        "2001,.5,.6,***,***,***,***,***,***,***,***,***,***,***\n",
        encoding="utf-8",
    )
    m = monthly_series(load_table(csv))
    assert len(m) == 14
    assert m.index[-1] == pd.Timestamp("2001-02-01")
    a = annual_series(m)
    assert list(a.index) == [2000]
    assert a.iloc[0] == pytest.approx(0.65)


def test_invalid_file_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_table(bad)


def test_annual_only_complete_years():
    a = annual_series(load_monthly())
    assert a.index.min() == 1880
    assert np.all(np.diff(a.index) == 1)
