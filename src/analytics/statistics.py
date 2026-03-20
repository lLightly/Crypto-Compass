from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def add_vix_deviation_indicators(
    df: pd.DataFrame,
    *,
    window: int | None = None,
    price_col: str = "close",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    w = int(window if window is not None else get_settings().assets.btc.engine.vix.rolling_window_days)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")

    out["rolling_mean"] = out[price_col].rolling(w, min_periods=1).mean()
    out["deviation_pct"] = (out[price_col] / out["rolling_mean"] - 1.0) * 100.0
    return out.reset_index(drop=True)


def calculate_cot_z_score(
    df: pd.DataFrame,
    *,
    column: str = "Comm_Net",
    window: int | None = None,
    enabled: bool | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    s = get_settings()
    if enabled is None:
        enabled = bool(s.assets.btc.engine.cot.z_score.enabled)
    if not enabled:
        return df.copy()

    w = int(window if window is not None else s.assets.btc.engine.cot.z_score.window_weeks)

    out = df.copy()
    x = pd.to_numeric(out[column], errors="coerce")
    roll = x.rolling(w, min_periods=1)
    roll_mean = roll.mean()
    roll_std = roll.std(ddof=0)
    count = roll.count()

    z = (x - roll_mean) / roll_std.replace(0, np.nan)
    out["Z_Score_Comm"] = z

    status = pd.Series("OK", index=out.index, dtype="object")
    status.loc[x.isna()] = "INVALID_INPUT"
    status.loc[(count < 2) & status.eq("OK")] = "INSUFFICIENT_HISTORY"
    status.loc[(count >= 2) & (roll_std == 0) & status.eq("OK")] = "ZERO_STD"
    status.loc[out["Z_Score_Comm"].isna() & status.eq("OK")] = "INVALID"

    out["Z_Score_Comm_Status"] = status.astype(str)
    return out.reset_index(drop=True)


def compute_max_drawdown(equity: pd.Series) -> Optional[float]:
    if equity is None:
        return None
    eq = pd.to_numeric(equity, errors="coerce").astype(float).dropna()
    if len(eq) < 2:
        return None
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def compute_sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 365) -> Optional[float]:
    r = pd.to_numeric(returns, errors="coerce").astype(float).dropna()
    if len(r) < 2:
        return None
    annual_rf = rf / periods_per_year
    excess = r - annual_rf
    mean_ex = excess.mean()
    std = excess.std(ddof=0)
    if std == 0.0:
        return None
    return float((mean_ex / std) * np.sqrt(periods_per_year))


def compute_cagr(total_return: float, num_periods: int, periods_per_year: int = 365) -> Optional[float]:
    if num_periods <= 0 or total_return is None or total_return < -1:
        return None
    return (1 + total_return) ** (periods_per_year / num_periods) - 1


def _price_series(df_price: pd.DataFrame) -> pd.Series:
    if df_price is None or df_price.empty:
        return pd.Series(dtype=float)
    tmp = df_price[["date", "close"]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    tmp = tmp.dropna(subset=["date"]).sort_values("date")
    return tmp.set_index("date")["close"].astype(float)


def forward_return(price: pd.Series, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Optional[float]:
    if price is None or price.empty:
        return None
    start_px = price.asof(start_ts)
    end_px = price.asof(end_ts)
    if pd.isna(start_px) or pd.isna(end_px) or start_px == 0:
        return None
    return float(end_px / start_px - 1.0)


def trend_accuracy(
    signals: pd.DataFrame,
    df_price: pd.DataFrame,
    horizon_months: int,
    bullish_label: str = "Бычий тренд",
    bearish_label: str = "Медвежий тренд",
) -> Tuple[Optional[float], Optional[float], Dict[str, int]]:
    confusion = {"bull_correct": 0, "bull_wrong": 0, "bear_correct": 0, "bear_wrong": 0, "evaluated": 0}

    if signals is None or signals.empty or df_price is None or df_price.empty:
        return None, None, confusion

    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    sig = sig.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    price = _price_series(df_price)
    if price.empty:
        return None, None, confusion

    total = len(sig)
    evaluated = 0
    correct = 0

    for _, row in sig.iterrows():
        verdict = str(row.get("verdict", ""))
        if verdict not in (bullish_label, bearish_label):
            continue

        start_ts = pd.Timestamp(row["date"]).normalize()
        end_ts = (start_ts + pd.DateOffset(months=int(horizon_months))).normalize()
        fr = forward_return(price, start_ts, end_ts)
        if fr is None:
            continue

        evaluated += 1
        if verdict == bullish_label:
            if fr > 0:
                correct += 1
                confusion["bull_correct"] += 1
            elif fr < 0:
                confusion["bull_wrong"] += 1
        else:
            if fr < 0:
                correct += 1
                confusion["bear_correct"] += 1
            elif fr > 0:
                confusion["bear_wrong"] += 1

    confusion["evaluated"] = evaluated
    if evaluated == 0:
        return None, None, confusion

    return float(correct / evaluated), float(evaluated / total) if total > 0 else None, confusion