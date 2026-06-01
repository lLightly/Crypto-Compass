from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Mapping

import pandas as pd
import yfinance as yf

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YahooDownloadSpec:
    ticker: str
    start: str


def _is_retryable_yahoo_error(exc: Exception) -> bool:
    text = str(exc).lower()
    name = exc.__class__.__name__.lower()
    return (
        "yfratelimiterror" in name
        or "too many requests" in text
        or "rate limited" in text
        or "429" in text
        or "timeout" in text
        or "connection" in text
    )


def _download_with_retries(
    *,
    tickers: list[str],
    start: str,
    interval: str,
    max_attempts: int = 3,
    base_delay_sec: float = 2.0,
) -> pd.DataFrame:
    ticker_arg = " ".join(tickers)
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            data = yf.download(
                tickers=ticker_arg,
                start=start,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            if data is None or data.empty:
                raise RuntimeError(f"Yahoo Finance returned an empty dataframe for: {ticker_arg}")
            return data
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts - 1 or not _is_retryable_yahoo_error(exc):
                break
            delay = base_delay_sec * (2 ** attempt) + random.uniform(0.0, 1.0)
            logger.warning(
                "Yahoo download failed for %s, attempt %s/%s: %s. Retrying in %.1fs.",
                ticker_arg,
                attempt + 1,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"Yahoo Finance download failed for {ticker_arg}: {last_error}") from last_error


def _extract_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)

        if ticker in level0:
            return data[ticker].copy()
        if ticker in level1:
            return data.xs(ticker, level=1, axis=1).copy()
        return pd.DataFrame()

    return data.copy()


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def _normalize_yahoo_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    src = df.copy()
    src = src.reset_index()

    date_col = _find_col(src, ("date", "datetime"))
    open_col = _find_col(src, ("open",))
    high_col = _find_col(src, ("high",))
    low_col = _find_col(src, ("low",))
    close_col = _find_col(src, ("close", "adj_close"))
    volume_col = _find_col(src, ("volume",))

    if date_col is None or close_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = (
        pd.to_datetime(src[date_col], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    out["open"] = pd.to_numeric(src[open_col], errors="coerce") if open_col else pd.NA
    out["high"] = pd.to_numeric(src[high_col], errors="coerce") if high_col else pd.NA
    out["low"] = pd.to_numeric(src[low_col], errors="coerce") if low_col else pd.NA
    out["close"] = pd.to_numeric(src[close_col], errors="coerce")
    out["volume"] = pd.to_numeric(src[volume_col], errors="coerce") if volume_col else pd.NA

    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return out


def _earliest_start(specs: Mapping[str, YahooDownloadSpec]) -> str:
    starts = pd.to_datetime([spec.start for spec in specs.values()], errors="coerce")
    starts = starts.dropna()
    if starts.empty:
        raise ValueError("At least one valid Yahoo start date is required")
    return pd.Timestamp(starts.min()).date().isoformat()


def fetch_yahoo_prices(
    specs: Mapping[str, YahooDownloadSpec],
    *,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    if not specs:
        return {}

    start = _earliest_start(specs)
    tickers = [spec.ticker for spec in specs.values()]
    batch = _download_with_retries(tickers=tickers, start=start, interval=interval)

    result: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for name, spec in specs.items():
        raw = _extract_ticker_frame(batch, spec.ticker)
        normalized = _normalize_yahoo_frame(raw)
        if not normalized.empty:
            min_date = pd.Timestamp(spec.start).normalize()
            normalized = normalized[normalized["date"] >= min_date].reset_index(drop=True)

        if normalized.empty:
            missing.append(name)
        else:
            result[name] = normalized

    for name in missing:
        spec = specs[name]
        time.sleep(1.0 + random.uniform(0.0, 0.5))
        single = _download_with_retries(tickers=[spec.ticker], start=spec.start, interval=interval, max_attempts=2)
        normalized = _normalize_yahoo_frame(_extract_ticker_frame(single, spec.ticker))
        if normalized.empty:
            raise RuntimeError(f"Yahoo Finance returned no usable rows for {name} ({spec.ticker})")
        result[name] = normalized

    return result


def price_specs() -> dict[str, YahooDownloadSpec]:
    s = get_settings()
    return {
        "vix": YahooDownloadSpec("^VIX", "2019-05-12"),
        "btc": YahooDownloadSpec("BTC-USD", s.assets.btc_price_start),
        "eth": YahooDownloadSpec("ETH-USD", s.assets.eth_price_start),
        "spx": YahooDownloadSpec("^GSPC", "2020-05-12"),
        "dxy": YahooDownloadSpec("DX-Y.NYB", "2020-05-12"),
        "us10y": YahooDownloadSpec("^TNX", "2020-05-12"),
    }


def fetch_all_prices(*, interval: str = "1d") -> dict[str, pd.DataFrame]:
    return fetch_yahoo_prices(price_specs(), interval=interval)


def fetch_vix(*, start: str = "2019-05-12", interval: str = "1d") -> pd.DataFrame:
    return fetch_yahoo_prices({"vix": YahooDownloadSpec("^VIX", start)}, interval=interval)["vix"]


def fetch_btc(*, start: str = "2020-05-12", interval: str = "1d") -> pd.DataFrame:
    s = get_settings()
    return fetch_yahoo_prices({"btc": YahooDownloadSpec("BTC-USD", start or s.assets.btc_price_start)}, interval=interval)["btc"]


def fetch_eth(*, start: str = "2023-03-28", interval: str = "1d") -> pd.DataFrame:
    s = get_settings()
    return fetch_yahoo_prices({"eth": YahooDownloadSpec("ETH-USD", start or s.assets.eth_price_start)}, interval=interval)["eth"]


def fetch_spx(*, start: str = "2020-05-12", interval: str = "1d") -> pd.DataFrame:
    return fetch_yahoo_prices({"spx": YahooDownloadSpec("^GSPC", start)}, interval=interval)["spx"]


def fetch_dxy(*, start: str = "2020-05-12", interval: str = "1d") -> pd.DataFrame:
    return fetch_yahoo_prices({"dxy": YahooDownloadSpec("DX-Y.NYB", start)}, interval=interval)["dxy"]


def fetch_us10y(*, start: str = "2020-05-12", interval: str = "1d") -> pd.DataFrame:
    return fetch_yahoo_prices({"us10y": YahooDownloadSpec("^TNX", start)}, interval=interval)["us10y"]