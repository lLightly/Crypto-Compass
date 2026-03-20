from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
import yfinance as yf

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _fetch_yahoo(ticker: str, start: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, interval=interval)

    if df is None or df.empty:
        raise RuntimeError(
            f"Failed to load {ticker} from Yahoo Finance (empty dataframe)"
        )

    out = df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )[["date", "open", "high", "low", "close", "volume"]]

    out["date"] = (
        pd.to_datetime(out["date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    return out


def fetch_vix(*, start: str = "2019-05-12", interval: str = "1d") -> pd.DataFrame:
    return _fetch_yahoo("^VIX", start, interval)


def fetch_btc(*, start: str = "2020-05-12", interval: str = "1d") -> pd.DataFrame:
    s = get_settings()
    return _fetch_yahoo("BTC-USD", start or s.assets.btc_price_start, interval)


def fetch_eth(*, start: str = "2023-03-28", interval: str = "1d") -> pd.DataFrame:
    s = get_settings()
    return _fetch_yahoo("ETH-USD", start or s.assets.eth_price_start, interval)


def fetch_spx(*, start: str = "2020-05-12", interval: str = "1d") -> pd.DataFrame:
    return _fetch_yahoo("^GSPC", start, interval)


def fetch_dxy(*, start: str = "2020-05-12", interval: str = "1d") -> pd.DataFrame:
    return _fetch_yahoo("DX=F", start, interval)


def fetch_us10y(*, start: str = "2020-05-12", interval: str = "1d") -> pd.DataFrame:
    return _fetch_yahoo("^TNX", start, interval)