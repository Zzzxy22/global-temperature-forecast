"""读取 NASA GISTEMP v4 全球地表温度距平表（GLB.Ts+dSST.csv）。

文件格式：第一行是标题，第二行是表头（Year, Jan … Dec, J-D, …），
缺失值记为 `***`。距平单位为 °C，基准期为 1951–1980 年平均。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "GLB.Ts+dSST.csv"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SOURCE_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"


def _find_header(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.startswith("Year,"):
                return i
    raise ValueError(f"{path} 不是有效的 GISTEMP 表格：找不到以 'Year,' 开头的表头")


def load_table(path: str | Path | None = None) -> pd.DataFrame:
    """读取原始年 × 月表格，`***` 转为 NaN。"""
    path = Path(path) if path else DEFAULT_CSV
    df = pd.read_csv(path, skiprows=_find_header(path), na_values=["***", "****"])
    # 部分版本的文件末尾会重复表头或附加说明行
    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()].copy()
    df["Year"] = df["Year"].astype(int)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("Year")


def monthly_series(table: pd.DataFrame) -> pd.Series:
    """展开为按月排列的时间序列，并去掉末尾尚未发布的月份。"""
    long = table[MONTHS].stack(future_stack=True)
    idx = pd.to_datetime(
        [f"{y}-{MONTHS.index(m) + 1:02d}-01" for y, m in long.index]
    )
    s = pd.Series(long.to_numpy(dtype=float), index=idx, name="anomaly_c").sort_index()
    last_valid = s.last_valid_index()
    s = s.loc[:last_valid]
    if s.isna().any():
        # 序列中间的缺失（极少见）用线性插值填补
        s = s.interpolate(limit_direction="both")
    s.index.freq = "MS"
    return s


def annual_series(monthly: pd.Series) -> pd.Series:
    """完整年份（12 个月齐全）的年平均距平。"""
    g = monthly.groupby(monthly.index.year)
    annual = g.mean()[g.count() == 12]
    annual.index.name = "year"
    return annual.rename("anomaly_c")


def load_monthly(path: str | Path | None = None) -> pd.Series:
    return monthly_series(load_table(path))


def to_numpy(s: pd.Series) -> np.ndarray:
    return s.to_numpy(dtype=float)
