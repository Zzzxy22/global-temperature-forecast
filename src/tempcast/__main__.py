"""命令行入口：python -m tempcast"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from . import plots
from .backtest import coverage, error_quantiles, forecast, mae_by_horizon, rolling_origin, score
from .data import DEFAULT_CSV, SOURCE_URL, annual_series, load_monthly
from .models import all_models
from .report import annual_outlook, build_report
from .trend import decade_means, period_rates, summarize, warmest_years


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="全球气温时间序列分析与预测（NASA GISTEMP v4）")
    p.add_argument("--csv", type=Path, default=None, help="GISTEMP 格式的 CSV（默认使用仓库自带数据）")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="输出目录（默认 outputs/）")
    p.add_argument("--horizon", type=int, default=24, help="预测未来多少个月（默认 24）")
    p.add_argument(
        "--update",
        action="store_true",
        help=f"先从 NASA 下载最新数据覆盖 data/ 中的文件（{SOURCE_URL}）",
    )
    return p.parse_args(argv)


def download_latest(dest: Path = DEFAULT_CSV) -> Path:
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "tempcast"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read()
    if b"Year," not in content:
        raise RuntimeError("下载内容不是预期的 GISTEMP 表格，已放弃更新")
    dest.write_bytes(content)
    return dest


def run(csv: Path | None, out: Path, horizon: int = 24) -> Path:
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] 加载数据")
    monthly = load_monthly(csv)
    annual = annual_series(monthly)

    print("[2/5] 趋势分析")
    summary = summarize(annual, monthly)
    rates = period_rates(annual)
    warmest = warmest_years(annual)
    decades = decade_means(annual)

    print("[3/5] 滚动回测（约 300 个预测起点 × 6 个模型，约需半分钟）")
    models = all_models()
    bt = rolling_origin(monthly, models, horizon=horizon)
    scores = score(bt)
    scores.to_csv(out / "backtest_scores.csv", index=False)
    best = scores.loc[0, "model"]
    best_model = next(m for m in models if m.name == best)
    q = error_quantiles(bt, best)
    cov80 = coverage(bt, best, 0.8)

    print(f"[4/5] 用 {best} 预测未来 {horizon} 个月")
    fc = forecast(monthly, best_model, q, horizon)
    fc.round(4).to_csv(out / "forecast.csv")
    outlook = annual_outlook(monthly, fc, annual)

    figures = {
        "history": plots.plot_history(monthly, annual, fig_dir / "history.png"),
        "heatmap": plots.plot_heatmap(monthly, fig_dir / "monthly_heatmap.png"),
        "rates": plots.plot_rates(rates, fig_dir / "warming_rates.png"),
        "backtest": plots.plot_backtest(mae_by_horizon(bt), fig_dir / "backtest_mae.png"),
        "forecast": plots.plot_forecast(monthly, fc, best, fig_dir / "forecast.png"),
    }

    print("[5/5] 写入报告")
    report = build_report(
        summary, rates, warmest, decades, scores, best, cov80, fc, outlook, figures, out / "report.md"
    )
    print(f"\n完成！最佳模型：{best}（回测 MAE {scores.loc[0, 'mae']:.3f} °C）")
    if len(outlook):
        o = outlook.iloc[0]
        print(f"{int(o['year'])} 年预计年均距平 {o['forecast']:+.2f} °C（历史第 {int(o['rank'])} 热）")
    print(f"报告：{report}")
    return report


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.update:
        path = download_latest(args.csv or DEFAULT_CSV)
        print(f"已下载最新数据到 {path}")
    run(args.csv, args.out, args.horizon)


if __name__ == "__main__":
    main()
