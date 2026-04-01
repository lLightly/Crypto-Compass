from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.settings import get_settings
from src.utils.pandas_utils import ensure_datetime_sorted, latest_dataset_date

_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "vix": ("date", "close"),
    "btc": ("date", "open", "high", "low", "close"),
    "eth": ("date", "open", "high", "low", "close"),
    "spx": ("date", "close"),
    "dxy": ("date", "close"),
    "us10y": ("date", "close"),
    "btc_cot": ("date", "Comm_Net", "Large_Specs_Net", "Small_Traders_Net", "open_interest_all"),
    "eth_cot": ("date", "Comm_Net", "Large_Specs_Net", "Small_Traders_Net", "open_interest_all"),
}


@dataclass(frozen=True)
class DatasetCheck:
    name: str
    path: str
    status: str
    rows: int
    min_date: pd.Timestamp | None
    max_date: pd.Timestamp | None
    issues: tuple[str, ...]

    def as_row(self) -> dict[str, object]:
        return {
            "Dataset": self.name,
            "Status": self.status,
            "Rows": self.rows,
            "Min Date": self.min_date,
            "Max Date": self.max_date,
            "Path": self.path,
            "Issues": " | ".join(self.issues),
        }


def load_dataset(name: str) -> pd.DataFrame | None:
    s = get_settings()
    rel = s.files.get(name)
    if not rel:
        return None
    path = Path(s.data_dir) / rel
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return ensure_datetime_sorted(df) if "date" in df.columns else df


def validate_dataset(name: str, df: pd.DataFrame | None) -> DatasetCheck:
    s = get_settings()
    rel = s.files.get(name)
    path = Path(s.data_dir) / rel if rel else None
    issues: list[str] = []
    min_date = max_date = None
    rows = 0

    if not rel:
        issues.append("dataset not configured")
    elif not path.exists():
        issues.append("file not found")

    if df is None:
        issues.append("dataset not loaded")
    elif df.empty:
        issues.append("dataset is empty")
    else:
        rows = int(len(df))
        required = _REQUIRED_COLUMNS.get(name, ("date",))
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"missing columns: {', '.join(missing)}")
        if "date" in df.columns:
            parsed = pd.to_datetime(df["date"], errors="coerce")
            bad = int(parsed.isna().sum())
            if bad:
                issues.append(f"invalid date rows: {bad}")
            good = parsed.dropna()
            if good.empty:
                issues.append("no valid dates")
            else:
                min_date = pd.Timestamp(good.min()).tz_localize(None).normalize()
                max_date = pd.Timestamp(good.max()).tz_localize(None).normalize()

    if not issues:
        status = "OK"
    elif df is None or (path is not None and not path.exists()):
        status = "MISSING"
    else:
        status = "INVALID"

    return DatasetCheck(
        name=name,
        path=path.as_posix() if path is not None else "",
        status=status,
        rows=rows,
        min_date=min_date,
        max_date=max_date,
        issues=tuple(issues),
    )


def validate_datasets(dfs: dict[str, pd.DataFrame | None]) -> list[DatasetCheck]:
    return [validate_dataset(name, dfs.get(name)) for name in get_settings().files]


def dataset_checks_frame(checks: list[DatasetCheck]) -> pd.DataFrame:
    return pd.DataFrame([c.as_row() for c in checks])


def datasets_max_date(dfs: dict[str, pd.DataFrame | None], names: list[str] | tuple[str, ...] | None = None):
    keys = list(names) if names is not None else list(dfs)
    dates = [latest_dataset_date(dfs.get(name)) for name in keys]
    dates = [x for x in dates if x is not None]
    return None if not dates else max(dates)


def filter_df(df: pd.DataFrame | None, start, end) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame()
    out = ensure_datetime_sorted(df)
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return out[out["date"].between(start, end, inclusive="both")].reset_index(drop=True)