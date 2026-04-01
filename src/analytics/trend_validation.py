from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.analytics.engines import get_asset_engine
from src.analytics.statistics import compute_cagr, compute_max_drawdown, compute_sharpe, trend_accuracy
from src.config.settings import get_settings
from src.constants import VERDICT_BULLISH, VERDICT_NO_DATA
from src.utils.pandas_utils import ensure_datetime_sorted


@dataclass
class TrendValidationResult:
    equity_curve: pd.DataFrame
    metrics: dict[str, Optional[float]]
    confusion: dict[str, int]
    signals: pd.DataFrame
    warnings: list[str]
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)


def _slice_price(df: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    out = ensure_datetime_sorted(df)
    if start_date is not None:
        out = out[out["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        out = out[out["date"] <= pd.Timestamp(end_date).normalize()]
    return out.reset_index(drop=True)


def _apply_trailing_stop(daily: pd.DataFrame, stop_pct: float) -> pd.DataFrame:
    daily = daily.copy()
    if stop_pct <= 0:
        daily["pos"] = daily["tgt_pos"]
        daily["trailing_stop_hit"] = False
        return daily

    pos, hits = [], []
    peak = np.nan
    prev = 0
    for close, tgt in daily[["close", "tgt_pos"]].itertuples(index=False):
        cur = int(tgt)
        hit = False
        if prev:
            peak = close if np.isnan(peak) else max(float(peak), float(close))
            if close <= peak * (1 - stop_pct):
                cur, hit, peak = 0, True, np.nan
            elif not cur:
                peak = np.nan
        elif cur:
            peak = float(close)
        pos.append(cur)
        hits.append(hit)
        prev = cur

    daily["pos"] = pos
    daily["trailing_stop_hit"] = hits
    return daily


def run_trend_validation(
    dfs: dict[str, pd.DataFrame],
    asset: str,
    initial_capital: float,
    start_date=None,
    end_date=None,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    rf_rate: float = 0.05,
    periods_per_year: int = 365,
    trailing_stop_pct: float = 0.0,
) -> TrendValidationResult:
    key = str(asset).lower()
    px_src = dfs.get(key)
    if px_src is None or px_src.empty:
        return TrendValidationResult(pd.DataFrame(), {}, {}, pd.DataFrame(), ["No price dataset found."])

    px = _slice_price(px_src, start_date, end_date)
    if len(px) < 2:
        return TrendValidationResult(pd.DataFrame(), {}, {}, pd.DataFrame(), ["Empty price slice."])

    warnings: list[str] = []
    eng = get_asset_engine(asset)
    sig_all = eng.generate_signals(dfs)
    if sig_all.empty:
        warnings.append(str(sig_all.attrs.get("error") or f"{asset}: signal generation failed."))
        return TrendValidationResult(pd.DataFrame(), {}, {}, sig_all, warnings)

    sig_all = sig_all.copy()
    sig_all["signal_date"] = pd.to_datetime(sig_all["signal_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    sig_all["exec_date"] = pd.to_datetime(sig_all["exec_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    sig_all["report_date"] = pd.to_datetime(sig_all["report_date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    blocked = sig_all[sig_all["data_status"] != "OK"] if "data_status" in sig_all.columns else pd.DataFrame()
    if not blocked.empty:
        warnings.append(f"{asset}: {len(blocked)} signal(s) blocked due to missing/stale/invalid factors.")

    exec_sig = sig_all[["exec_date", "verdict"]].dropna().sort_values("exec_date").rename(columns={"exec_date": "date"})
    daily = px[["date", "close"]].copy()
    daily = pd.merge_asof(daily, exec_sig, on="date", direction="backward")
    daily["verdict"] = daily["verdict"].fillna(VERDICT_NO_DATA)
    daily["tgt_pos"] = (daily["verdict"] == VERDICT_BULLISH).astype(int)
    daily = _apply_trailing_stop(daily, trailing_stop_pct)

    daily["asset_ret"] = daily["close"].pct_change().fillna(0.0)
    daily["prev_pos"] = daily["pos"].shift().fillna(0).astype(int)
    daily["trade"] = daily["pos"].sub(daily["prev_pos"]).abs()
    daily["strategy_ret"] = daily["asset_ret"] * daily["prev_pos"] - daily["trade"] * (fee_pct + slippage_pct)
    daily["Equity"] = float(initial_capital) * (1.0 + daily["strategy_ret"]).cumprod()

    curve = daily[["date", "close", "Equity"]].copy()
    total_ret = float(curve["Equity"].iloc[-1] / float(initial_capital) - 1.0)
    bh_eq = (float(initial_capital) / daily["close"].iloc[0]) * daily["close"]
    bh_ret = float(bh_eq.iloc[-1] / float(initial_capital) - 1.0)

    n = len(daily) - 1
    cagr = compute_cagr(total_ret, n, periods_per_year)
    bh_cagr = compute_cagr(bh_ret, n, periods_per_year)
    dd = compute_max_drawdown(daily["Equity"])
    bh_dd = compute_max_drawdown(bh_eq)

    sig_eval = sig_all.copy()
    sig_eval["date"] = sig_eval["exec_date"].where(sig_eval["exec_date"].notna(), sig_eval["signal_date"])
    sig_show = sig_all.copy()

    if start_date is not None:
        start_ts = pd.Timestamp(start_date).normalize()
        sig_eval = sig_eval[sig_eval["date"] >= start_ts]
        sig_show = sig_show[(sig_show["signal_date"] >= start_ts) | (sig_show["exec_date"] >= start_ts)]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date).normalize()
        sig_eval = sig_eval[sig_eval["date"] <= end_ts]
        sig_show = sig_show[(sig_show["signal_date"] <= end_ts) | (sig_show["exec_date"] <= end_ts)]

    acc, cov, confusion = trend_accuracy(
        signals=sig_eval.reset_index(drop=True),
        df_price=px,
        horizon_months=int(get_settings().compass.trend_horizon_months),
    )

    metrics = {
        "total_return": total_ret,
        "bh_total_return": bh_ret,
        "cagr": cagr,
        "bh_cagr": bh_cagr,
        "max_dd": dd,
        "bh_max_dd": bh_dd,
        "calmar": (cagr / -dd if cagr is not None and dd is not None and dd < 0 else None),
        "bh_calmar": (bh_cagr / -bh_dd if bh_cagr is not None and bh_dd is not None and bh_dd < 0 else None),
        "sharpe": compute_sharpe(daily["strategy_ret"], rf=rf_rate, periods_per_year=periods_per_year),
        "sharpe_bh": compute_sharpe(daily["asset_ret"], rf=rf_rate, periods_per_year=periods_per_year),
        "trend_accuracy": acc,
        "trend_coverage": cov,
        "horizon_months": float(get_settings().compass.trend_horizon_months),
    }
    return TrendValidationResult(curve, metrics, confusion, sig_show.reset_index(drop=True), warnings, daily)