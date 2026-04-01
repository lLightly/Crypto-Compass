from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_csv(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if "date" in out.columns:
        d = pd.to_datetime(out["date"], errors="coerce")
        if pd.api.types.is_datetime64_any_dtype(d):
            out["date"] = d.dt.tz_localize(None).dt.normalize().dt.strftime("%Y-%m-%d")

    out.to_csv(p, index=False)
    logger.info("Saved: %s", p.as_posix())