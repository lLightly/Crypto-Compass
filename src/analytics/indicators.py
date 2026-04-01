from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import get_settings


def _cot_index_with_status(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    s = pd.to_numeric(series, errors="coerce")
    roll = s.rolling(window=window, min_periods=1)
    roll_min = roll.min()
    roll_max = roll.max()
    denom = roll_max - roll_min
    count = roll.count()

    idx = (s - roll_min) / denom.replace(0, np.nan) * 100

    status = pd.Series("OK", index=s.index, dtype="object")
    status.loc[s.isna()] = "INVALID_INPUT"
    status.loc[(count < 2) & status.eq("OK")] = "INSUFFICIENT_HISTORY"
    status.loc[(count >= 2) & (denom == 0) & status.eq("OK")] = "ZERO_RANGE"
    status.loc[idx.isna() & status.eq("OK")] = "INVALID"

    return idx, status


def build_cot_indicators(
    df: pd.DataFrame,
    *,
    window_weeks: int | None = None,
    enabled: bool | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    s = get_settings()
    if enabled is None:
        enabled = bool(s.assets.btc.engine.cot.index.enabled)
    if not enabled:
        return df.copy()

    w = int(window_weeks if window_weeks is not None else s.assets.btc.engine.cot.index.window_weeks)

    out = df.copy()

    comm_idx, comm_status = _cot_index_with_status(out["Comm_Net"], window=w)
    large_idx, large_status = _cot_index_with_status(out["Large_Specs_Net"], window=w)

    out["COT_Index_Comm"] = comm_idx.round(2)
    out["COT_Index_Comm_Status"] = comm_status.astype(str)

    out["COT_Index_Large"] = large_idx.round(2)
    out["COT_Index_Large_Status"] = large_status.astype(str)

    out["COT_Index_Large_Inverted"] = (100 - out["COT_Index_Large"]).round(2)
    out["COT_Index_Large_Inverted_Status"] = out["COT_Index_Large_Status"]

    return out