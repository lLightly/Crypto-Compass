from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd


def to_timestamp(value: Any) -> pd.Timestamp:
    """
    Convert a date-like input into a normalized pandas Timestamp (00:00:00).
    Raises ValueError for invalid inputs (caller may fallback).
    """
    if value is None:
        raise ValueError("Date is None")

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    if isinstance(value, dt.datetime):
        return pd.Timestamp(value).normalize()

    if isinstance(value, dt.date):
        return pd.Timestamp(dt.datetime.combine(value, dt.time.min)).normalize()

    # strings etc
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value!r}")
    return pd.Timestamp(ts).normalize()