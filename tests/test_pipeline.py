"""端到端测试：跑一遍完整流程（预测 12 个月以缩短耗时）。"""

import pandas as pd

from tempcast.__main__ import run


def test_full_pipeline(tmp_path):
    report = run(csv=None, out=tmp_path, horizon=12)
    text = report.read_text(encoding="utf-8")
    assert "核心结论" in text and "未来 12 个月预测" in text

    figs = sorted(p.name for p in (tmp_path / "figures").glob("*.png"))
    assert figs == [
        "backtest_mae.png",
        "forecast.png",
        "history.png",
        "monthly_heatmap.png",
        "warming_rates.png",
    ]
    fc = pd.read_csv(tmp_path / "forecast.csv")
    assert len(fc) == 12
    assert (fc["lo95"] <= fc["lo80"]).all() and (fc["hi80"] <= fc["hi95"]).all()
    scores = pd.read_csv(tmp_path / "backtest_scores.csv")
    # 最佳模型应明显优于朴素预测
    naive = scores.loc[scores["model"] == "Naive", "mae"].item()
    assert scores["mae"].min() < naive * 0.9
