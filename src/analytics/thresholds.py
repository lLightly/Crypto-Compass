from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import pandas as pd


def get_deviation_levels(
    df_or_series: Union[pd.DataFrame, pd.Series],
    *,
    col: str = "deviation_pct",
    sigma_levels: Optional[Iterable[int]] = None,
    lookback_points: int = 252 * 3,
) -> Optional[Dict[str, float]]:
    if df_or_series is None:
        return None

    if isinstance(df_or_series, pd.Series):
        s = pd.to_numeric(df_or_series, errors="coerce").dropna()
    else:
        if df_or_series.empty or col not in df_or_series.columns:
            return None
        s = pd.to_numeric(df_or_series[col], errors="coerce").dropna()

    if s.empty:
        return None

    tail = s.iloc[-lookback_points:] if len(s) > int(lookback_points) else s
    mean = float(tail.mean())
    std = float(tail.std(ddof=0))

    lv = list(sigma_levels or [1, 2])
    out: Dict[str, float] = {"mean": mean}
    for k in lv:
        out[f"+{k}σ"] = mean + k * std
        out[f"-{k}σ"] = mean - k * std
    return out


def get_quantile_thresholds(
    series: pd.Series,
    *,
    lookback_points: int = 104,
    quantiles: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    if series is None:
        return None

    q = quantiles or {"p5": 0.05, "p10": 0.10, "p90": 0.90, "p95": 0.95}
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None

    tail = clean.iloc[-lookback_points:] if len(clean) > int(lookback_points) else clean
    qs = tail.quantile([q["p5"], q["p10"], q["p90"], q["p95"]])
    return {
        "p5": round(float(qs.loc[q["p5"]]), 2),
        "p10": round(float(qs.loc[q["p10"]]), 2),
        "p90": round(float(qs.loc[q["p90"]]), 2),
        "p95": round(float(qs.loc[q["p95"]]), 2),
    }