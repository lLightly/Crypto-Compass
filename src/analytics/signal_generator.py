from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analytics.engines import get_asset_engine
from src.constants import ASSETS, VERDICT_BEARISH, VERDICT_BULLISH, VERDICT_NO_DATA

logger = logging.getLogger(__name__)


def score_asset(asset: str, dfs: dict[str, pd.DataFrame], as_of: pd.Timestamp | None = None):
    return get_asset_engine(asset).score_asset(dfs, as_of=as_of)


def generate_conclusion(dfs: dict[str, pd.DataFrame], as_of: pd.Timestamp | None = None):
    per_asset: dict[str, tuple[pd.DataFrame, float | None, str, float, str]] = {}
    totals, buy_thrs, sell_thrs = [], [], []

    for asset in ASSETS:
        eng = get_asset_engine(asset)
        try:
            table, total, verdict, conf, narrative = eng.score_asset(dfs, as_of=as_of, use_publication_lag=False)
        except Exception as e:
            logger.exception("Conclusion failed for %s", asset)
            table = pd.DataFrame(
                [
                    {
                        "Factor": "Internal Error",
                        "Status": "ERROR",
                        "Value": None,
                        "Lookup Date": pd.Timestamp(as_of).normalize() if as_of is not None else pd.NaT,
                        "Source Date": pd.NaT,
                        "Thresholds": "",
                        "Score": np.nan,
                        "Rationale": str(e),
                    }
                ]
            )
            per_asset[asset] = (table, None, VERDICT_NO_DATA, 0.0, f"- **Internal Error** [ERROR]: {e}")
            continue

        per_asset[asset] = (table, total, verdict, conf, narrative)
        buy_thrs.append(float(eng.cfg.verdict_thresholds.verdict_buy))
        sell_thrs.append(float(eng.cfg.verdict_thresholds.verdict_sell))
        if total is not None and verdict != VERDICT_NO_DATA:
            totals.append(float(total))

    score = float(np.mean(totals)) if totals else None
    buy_thr = float(np.mean(buy_thrs)) if buy_thrs else 1.5
    sell_thr = float(np.mean(sell_thrs)) if sell_thrs else buy_thr
    verdict = (
        VERDICT_NO_DATA
        if score is None
        else VERDICT_BULLISH if score >= buy_thr else VERDICT_BEARISH if score <= -sell_thr else "нейтральный"
    )
    return per_asset, (None if score is None else round(score, 2)), verdict


def generate_signals(dfs_full: dict[str, pd.DataFrame], asset: str = "BTC") -> pd.DataFrame:
    return get_asset_engine(asset).generate_signals(dfs_full)