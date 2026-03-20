from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

from src.config.settings import get_settings
from src.data_fetchers import finance_api
from src.data_fetchers.cot_parser import fetch_cot_raw, preprocess
from src.utils.helpers import save_csv

logger = logging.getLogger(__name__)


def _update_price(name: str, fetch_fn: Callable[[], pd.DataFrame], proc_dir: Path) -> None:
    try:
        df = fetch_fn()
        save_csv(df, str(proc_dir / f"{name}_price.csv"))
    except Exception as e:
        logger.exception("Failed to update price dataset %s: %s", name, e)


def _update_cot(asset: str, raw_dir: Path, proc_dir: Path) -> None:
    try:
        cot_raw = fetch_cot_raw(asset)
        if cot_raw is None or cot_raw.empty:
            logger.warning("COT raw is empty for %s", asset)
            return

        save_csv(cot_raw, str(raw_dir / f"{asset.lower()}_cot_raw.csv"))
        cot = preprocess(cot_raw)
        if cot.empty:
            logger.warning("COT preprocess produced empty df for %s", asset)
            return

        save_csv(cot.sort_values("date"), str(proc_dir / f"{asset.lower()}_cot_processed.csv"))
    except Exception as e:
        logger.exception("Failed to update COT dataset %s: %s", asset, e)


def update_all_data() -> None:
    s = get_settings()

    raw_dir = Path("data/raw")
    proc_dir = Path(s.data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    try:
        vix_raw = finance_api.fetch_vix()
        save_csv(vix_raw, str(raw_dir / "vix.csv"))
        save_csv(vix_raw, str(proc_dir / "vix_processed.csv"))
    except Exception as e:
        logger.exception("Failed to update VIX: %s", e)

    price_fetchers: Dict[str, Callable[[], pd.DataFrame]] = {
        "btc": finance_api.fetch_btc,
        "eth": finance_api.fetch_eth,
        "spx": finance_api.fetch_spx,
        "dxy": finance_api.fetch_dxy,
        "us10y": finance_api.fetch_us10y,
    }
    for name, fn in price_fetchers.items():
        _update_price(name, fn, proc_dir)

    _update_cot("BTC", raw_dir, proc_dir)
    _update_cot("ETH", raw_dir, proc_dir)