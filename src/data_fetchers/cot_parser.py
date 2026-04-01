from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
LIMIT = 50000

MARKETS: Dict[str, str] = {
    "BTC": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
    "ETH": "ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE",
}


def fetch_cot_raw(asset: str = "BTC") -> pd.DataFrame:
    market = MARKETS.get(asset.upper())
    if not market:
        raise ValueError(f"Unknown asset: {asset}")

    offset = 0
    data: List[dict] = []
    while True:
        params = {"$limit": LIMIT, "$offset": offset, "$where": f"market_and_exchange_names='{market}'"}
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        data.extend(batch)
        offset += LIMIT

    return pd.DataFrame(data)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out.rename(columns={"report_date_as_yyyy_mm_dd": "date"})

    out["date"] = (
        pd.to_datetime(out["date"], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    num_cols = [
        "open_interest_all",
        "comm_positions_long_all",
        "comm_positions_short_all",
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "nonrept_positions_long_all",
        "nonrept_positions_short_all",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["Comm_Net"] = out["comm_positions_long_all"] - out["comm_positions_short_all"]
    out["Large_Specs_Net"] = out["noncomm_positions_long_all"] - out["noncomm_positions_short_all"]
    out["Small_Traders_Net"] = out["nonrept_positions_long_all"] - out["nonrept_positions_short_all"]

    return out