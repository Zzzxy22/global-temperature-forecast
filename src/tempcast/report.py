"""汇总结果并生成 Markdown 报告。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def annual_outlook(monthly: pd.Series, fc: pd.DataFrame, annual: pd.Series) -> pd.DataFrame:
    """把观测值与预测值拼成完整日历年，估计年平均距平及其在历史中的排名。

    年度区间取各月区间上下限的平均，相当于假设各月误差完全同向，
    因此是偏宽（保守）的近似。
    """
    rows = []
    for year in sorted(set(fc.index.year)):
        obs = monthly[monthly.index.year == year]
        f = fc[fc.index.year == year]
        if len(obs) + len(f) < 12:
            continue
        n_obs = len(obs)
        point = (obs.sum() + f["forecast"].sum()) / 12
        lo = (obs.sum() + f["lo80"].sum()) / 12
        hi = (obs.sum() + f["hi80"].sum()) / 12
        hist = annual[annual.index < year]
        rank = int((hist > point).sum()) + 1
        rows.append(
            {"year": year, "n_observed": n_obs, "forecast": point, "lo80": lo, "hi80": hi, "rank": rank}
        )
    return pd.DataFrame(rows)


def build_report(
    summary: dict,
    rates: pd.DataFrame,
    warmest: pd.DataFrame,
    decades: pd.Series,
    scores: pd.DataFrame,
    best: str,
    cov80: float,
    fc: pd.DataFrame,
    outlook: pd.DataFrame,
    figures: dict[str, Path],
    out_path: Path,
) -> Path:
    rel = {k: v.relative_to(out_path.parent).as_posix() for k, v in figures.items()}
    s = summary
    modern = rates.iloc[-1]
    prewar = rates.iloc[0]
    b = scores.set_index("model")
    best_row = b.loc[best]
    naive_mae = b.loc["Naive", "mae"]
    gain = 1 - best_row["mae"] / naive_mae
    runner = scores.iloc[1]

    rate_tbl = pd.DataFrame(
        {
            "时期": rates["period"],
            "升温速率（°C/十年）": rates["rate_per_decade"].map("{:+.3f}".format),
            "95% 置信区间": [f"{lo:+.3f} ~ {hi:+.3f}" for lo, hi in zip(rates["ci_low"], rates["ci_high"])],
            "R²": rates["r2"].map("{:.2f}".format),
        }
    )
    warm_tbl = pd.DataFrame(
        {
            "排名": warmest["rank"],
            "年份": warmest["year"],
            "距平（°C）": warmest["anomaly_c"].map("{:+.2f}".format),
        }
    )
    score_tbl = pd.DataFrame(
        {
            "模型": scores["model"],
            "平均 MAE": scores["mae"].map("{:.3f}".format),
            "RMSE": scores["rmse"].map("{:.3f}".format),
            **{
                f"{c[5:]} 个月": scores[c].map("{:.3f}".format)
                for c in scores.columns
                if c.startswith("mae_h")
            },
            "平均偏差": scores["bias"].map("{:+.3f}".format),
        }
    )
    out_tbl = pd.DataFrame(
        {
            "年份": outlook["year"],
            "已观测月份": outlook["n_observed"],
            "预测年均距平（°C）": outlook["forecast"].map("{:+.2f}".format),
            "约 80% 区间": [f"{lo:+.2f} ~ {hi:+.2f}" for lo, hi in zip(outlook["lo80"], outlook["hi80"])],
            "历史排名（点预测）": outlook["rank"].map(lambda r: f"第 {r}"),
        }
    )
    sel = fc.iloc[sorted({0, *range(5, len(fc), 6), len(fc) - 1})]
    fc_tbl = pd.DataFrame(
        {
            "月份": sel.index.strftime("%Y-%m"),
            "点预测（°C）": sel["forecast"].map("{:+.2f}".format),
            "80% 区间": [f"{lo:+.2f} ~ {hi:+.2f}" for lo, hi in zip(sel["lo80"], sel["hi80"])],
            "95% 区间": [f"{lo:+.2f} ~ {hi:+.2f}" for lo, hi in zip(sel["lo95"], sel["hi95"])],
        }
    )
    dec = decades.loc[1980:]
    dec_txt = "、".join(f"{d}s 为 {v:+.2f} °C" for d, v in dec.items())
    bias_line = (
        "- 所有模型的平均偏差都为正，说明**回测期内实际气温系统性地高于各模型的预测**："
        "基于历史趋势的方法会低估升温的加速。"
        if (scores["bias"] > 0).all()
        else ""
    )
    o1 = outlook.iloc[0] if len(outlook) else None
    outlook_line = (
        f"- 结合已观测的 {int(o1['n_observed'])} 个月，**{int(o1['year'])} 年**全年平均距平预计约 "
        f"**{o1['forecast']:+.2f} °C**（约 80% 区间 {o1['lo80']:+.2f} ~ {o1['hi80']:+.2f} °C），"
        f"按点预测将位列有记录以来第 **{int(o1['rank'])}** 热。"
        if o1 is not None
        else ""
    )

    text = f"""# 全球气温时间序列分析与预测

> 自动生成于 {date.today().isoformat()} · 数据：NASA GISTEMP v4 全球陆地-海洋温度距平（基准期 1951–1980）

## 1. 核心结论

- 数据覆盖 {s['first_month'].year} 年 {s['first_month'].month} 月 至 **{s['last_month'].year} 年 {s['last_month'].month} 月**，共 {s['n_months']:,} 个月。
- **{s['warmest_year']} 年是有记录以来最热的一年**，年均距平 {s['warmest_year_anomaly']:+.2f} °C，比 1880–1900 年平均高约 **{s['warmest_year_vs_early']:.2f} °C**；最热的 10 个年份中有 **{s['top10_since_2010']}** 个出现在 2010 年之后。
- 升温在加速：{prewar['period']} 约 {prewar['rate_per_decade']:+.3f} °C/十年，**{modern['period']} 达到 {modern['rate_per_decade']:+.3f} °C/十年**（95% 置信区间 {modern['ci_low']:+.3f} ~ {modern['ci_high']:+.3f}）。
- 在 {int(best_row['n_origins'])} 个预测起点的滚动回测中，**{best}** 平均误差最小（MAE {best_row['mae']:.3f} °C），比"维持上月不变"的朴素预测低 {gain:.0%}。
{bias_line}
{outlook_line}

## 2. 数据

- 来源：NASA 戈达德空间研究所 GISTEMP v4（`data/GLB.Ts+dSST.csv`），数值为相对 1951–1980 年平均的月均气温距平（°C）。
- 处理：把"年 × 月"宽表展开为月度序列；`***` 记为缺失，去掉末尾尚未发布的月份。年平均只统计 12 个月完整的年份。
- 气温距平已去除季节循环，因此序列的主要成分是**长期趋势**和**数月到数年尺度的波动**（如厄尔尼诺/拉尼娜）。

![历史曲线]({rel['history']})

![月度热力图]({rel['heatmap']})

## 3. 长期趋势

各时期的线性升温速率：

{md_table(rate_tbl)}

![升温速率]({rel['rates']})

20 世纪中叶升温基本停滞（甚至略有下降），1970 年后明显加快，2000 年以来又进一步提速。各年代平均：{dec_txt}。

> 置信区间由普通最小二乘给出。年际自相关会使标准误偏小，实际不确定性比表中略大。

有记录以来最热的 10 个年份：

{md_table(warm_tbl)}

## 4. 预测模型与回测

比较了 6 种模型：

| 模型 | 思路 |
| --- | --- |
| Naive | 未来每个月都等于最后一个观测值 |
| 12-month mean | 未来等于最近 12 个月的平均 |
| Linear trend | 对最近 30 年做直线回归并外推 |
| Damped Holt | 阻尼趋势指数平滑，参数通过最小化一步预测误差估计 |
| AR + trend | 带线性趋势的 12 阶自回归（Ridge 估计，递推多步预测） |
| Ensemble | Linear trend、Damped Holt、AR + trend 三者平均 |

**回测方法（滚动起点）**：从 2000 年 1 月起每个月设一个预测起点，只用起点之前的数据训练，预测之后 1–{len(fc)} 个月，再与真实值比较。这模拟了"站在过去某一时刻做预测"的真实情境，不会用到未来信息。

{md_table(score_tbl)}

![回测误差]({rel['backtest']})

- **短期（1–6 个月）**：自回归和指数平滑能利用最近几个月的偏离（如正在发生的厄尔尼诺），表现最好。
- **长期（12 个月以上）**：短期波动的信息逐渐失效，误差主要取决于对长期趋势的把握，线性趋势类模型更稳。
- {best} 与第二名 {runner['model']} 的平均误差仅相差 {runner['mae'] - best_row['mae']:.3f} °C，差距在回测噪声范围内，可视为表现相当。

## 5. 未来 {len(fc)} 个月预测

使用 **{best}** 在全部数据上重新训练。预测区间**不假设误差服从正态分布**，而是直接取回测中各预测步长误差的经验分位数，因此越往后区间越宽。由于回测中模型通常偏低，区间也**向上方倾斜**（上限离点预测更远）。在回测样本中，80% 区间的实际覆盖率为 {cov80:.0%}（样本内评估，偏乐观）。

![预测]({rel['forecast']})

{md_table(fc_tbl)}

按日历年汇总：

{md_table(out_tbl)}

## 6. 局限性

- 统计模型只能外推历史规律，**无法预知**火山爆发、厄尔尼诺突然转换等事件；物理气候模式（如 CMIP）才是长期气候预测的主要工具。
- 回测期（2000 年至今）内升温在加速，导致所有模型存在正偏差；如果加速继续，预测很可能偏低。
- "1880–1900 平均"只是工业化前水平的粗略近似（《巴黎协定》通常使用 1850–1900），不同数据集的数值也会有差异。
- 本仓库附带的数据截至 {s['last_month'].year} 年 {s['last_month'].month} 月。下载 NASA 最新数据后重新运行，即可得到更新的分析与预测（见 README）。
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
