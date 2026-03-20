from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.utils.dates import to_timestamp

STATUS_OK = "OK"
STATUS_MISSING = "MISSING"
STATUS_STALE = "STALE"
STATUS_INVALID = "INVALID"


@dataclass(frozen=True)
class AsofResult:
    value: Any
    source_date: pd.Timestamp | None
    status: str
    message: str


def ensure_datetime_sorted(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return out


def latest_dataset_date(df: pd.DataFrame | None, *, date_col: str = "date") -> pd.Timestamp | None:
    if df is None or df.empty or date_col not in df.columns:
        return None
    clean = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if clean.empty:
        return None
    return pd.Timestamp(clean.max()).tz_localize(None).normalize()


def asof_value(
    df: pd.DataFrame,
    *,
    as_of: Any,
    value_col: str,
    date_col: str = "date",
    tolerance: pd.Timedelta | None = None,
) -> AsofResult:
    if df is None or df.empty:
        return AsofResult(None, None, STATUS_MISSING, "Dataset is empty")
    if date_col not in df.columns:
        return AsofResult(None, None, STATUS_MISSING, f"Missing column: {date_col}")
    if value_col not in df.columns:
        return AsofResult(None, None, STATUS_MISSING, f"Missing column: {value_col}")

    as_of_ts = to_timestamp(as_of)
    clean = ensure_datetime_sorted(df, date_col=date_col)
    if clean.empty:
        return AsofResult(None, None, STATUS_MISSING, "No valid dated rows")

    mask = clean[date_col] <= as_of_ts
    if not mask.any():
        return AsofResult(None, None, STATUS_MISSING, f"No row <= {as_of_ts.date().isoformat()}")

    row = clean.loc[mask].iloc[-1]
    src_date = pd.Timestamp(row[date_col]).normalize()
    val = row.get(value_col)

    if pd.isna(val):
        return AsofResult(None, src_date, STATUS_INVALID, f"{value_col} is NaN on {src_date.date().isoformat()}")

    age_msg = ""
    if tolerance is not None:
        age = as_of_ts - src_date
        if age > tolerance:
            return AsofResult(
                None,
                src_date,
                STATUS_STALE,
                f"Data too old: {src_date.date().isoformat()} (age {age.days}d > {tolerance.days}d)",
            )
        age_msg = f"Used {src_date.date().isoformat()} for {as_of_ts.date().isoformat()}" if src_date != as_of_ts else "Текущие данные"

    return AsofResult(val, src_date, STATUS_OK, age_msg)


def slice_until(df: pd.DataFrame, end_ts: Any, *, date_col: str = "date") -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return pd.DataFrame()
    end_ts = to_timestamp(end_ts)
    clean = ensure_datetime_sorted(df, date_col=date_col)
    if clean.empty:
        return clean
    dates = clean[date_col].to_numpy()
    pos = dates.searchsorted(end_ts.to_datetime64(), side="right")
    return clean.iloc[:pos].reset_index(drop=True)