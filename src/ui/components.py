from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.analytics.indicators import build_cot_indicators
from src.analytics.scoring import get_vix_scoring_levels
from src.analytics.statistics import add_vix_deviation_indicators, calculate_cot_z_score
from src.config.settings import AssetEngineSettings, COTIndexSettings, get_settings

DEFAULT_TEMPLATE = "plotly_dark"


def _engine_cfg(asset: str) -> AssetEngineSettings:
    return get_settings().engine_for(asset)


def _pad(fig: go.Figure, x: pd.Series, padding_days: int) -> go.Figure:
    pad = pd.Timedelta(days=padding_days)
    fig.update_xaxes(range=[x.min() - pad, x.max() + pad], tickformat="%d.%m.%y")
    return fig


def _set_x_range(fig: go.Figure, x_range_min, x_range_max, padding_days: int) -> None:
    if x_range_min is None or x_range_max is None:
        return
    pad = pd.Timedelta(days=padding_days)
    fig.update_xaxes(range=[x_range_min - pad, x_range_max + pad], tickformat="%d.%m.%y")


def _clip_display(df: pd.DataFrame, start=None, end=None, *, date_col: str = "date") -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    if start is not None:
        out = out[out[date_col] >= pd.Timestamp(start).normalize()]
    if end is not None:
        out = out[out[date_col] <= pd.Timestamp(end).normalize()]
    return out.reset_index(drop=True)


def _rolling_bands(series: pd.Series, lookback: int, sigma_levels: list[int]) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.rolling(lookback, min_periods=1).mean()
    std = s.rolling(lookback, min_periods=1).std(ddof=0)
    out = pd.DataFrame({"mean": mean})
    for sigma in sigma_levels:
        out[f"+{sigma}σ"] = mean + sigma * std
        out[f"-{sigma}σ"] = mean - sigma * std
    return out


def _rolling_quantiles(series: pd.Series, lookback: int, quantiles: dict[str, float]) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame({k: s.rolling(lookback, min_periods=1).quantile(v) for k, v in quantiles.items()})


def _append_rows(rows: list[dict[str, object]], factor: str, items: dict[str, object]) -> None:
    for key, value in items.items():
        rows.append({"Factor": factor, "Parameter": key, "Value": value})


def _index_rows(rows: list[dict[str, object]], factor: str, cfg: COTIndexSettings) -> None:
    q = cfg.quantiles
    _append_rows(
        rows,
        factor,
        {
            "enabled": bool(cfg.enabled),
            "weight": float(cfg.weight),
            "window_weeks": int(cfg.window_weeks),
            "quantile_lookback_points": int(cfg.quantile_lookback_points),
            "quantile_p5": float(q.p5),
            "quantile_p10": float(q.p10),
            "quantile_p90": float(q.p90),
            "quantile_p95": float(q.p95),
            "bear_logic": "bull=max(abs,quant); bear=abs+quant",
        },
    )
    for label, comp in (("comm", cfg.scoring.comm), ("large_inv", cfg.scoring.large_inv)):
        _append_rows(
            rows,
            factor,
            {
                f"{label}_abs_strong_bear": float(comp.abs_thresholds.strong_bear),
                f"{label}_abs_bear": float(comp.abs_thresholds.bear),
                f"{label}_abs_bull": float(comp.abs_thresholds.bull),
                f"{label}_abs_strong_bull": float(comp.abs_thresholds.strong_bull),
                f"{label}_score_strong_bear": float(comp.scores.strong_bear),
                f"{label}_score_bear": float(comp.scores.bear),
                f"{label}_score_bull": float(comp.scores.bull),
                f"{label}_score_strong_bull": float(comp.scores.strong_bull),
            },
        )


def _display_value(value: object) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    return str(value)


def signal_parameter_table(asset: str) -> pd.DataFrame:
    cfg = _engine_cfg(asset)
    rows: list[dict[str, object]] = []

    _append_rows(
        rows,
        "Verdict",
        {
            "buy_threshold": float(cfg.verdict_thresholds.verdict_buy),
            "sell_threshold": float(cfg.verdict_thresholds.verdict_sell),
            "min_confidence_for_verdict": float(cfg.min_confidence_for_verdict),
        },
    )
    _append_rows(
        rows,
        "Availability",
        {
            "trend_validation_market_lookback_days": 1,
            "trend_validation_cot_publication_lag_days": int(cfg.cot.publication_lag_days),
            "conclusion_uses_publication_lag": False,
            "trend_validation_uses_publication_lag": True,
        },
    )

    if cfg.vix.enabled:
        sc = cfg.vix.scoring
        _append_rows(
            rows,
            "VIX Risk Regime",
            {
                "enabled": bool(cfg.vix.enabled),
                "weight": float(cfg.vix.weight),
                "scale": float(cfg.vix.scale),
                "asof_tolerance_days": int(cfg.vix.asof_tolerance_days),
                "rolling_window_days": int(cfg.vix.rolling_window_days),
                "levels_lookback_points": int(cfg.vix.levels_lookback_points),
                "sigma_levels": ", ".join(str(x) for x in cfg.vix.sigma_levels),
                "score_risk_off": float(sc.risk_off_score),
                "score_strong_risk_off": float(sc.strong_risk_off_score),
                "score_very_strong_risk_off": float(sc.very_strong_risk_off_score),
                "score_risk_on": float(sc.risk_on_score),
                "score_strong_risk_on": float(sc.strong_risk_on_score),
                "score_very_strong_risk_on": float(sc.very_strong_risk_on_score),
            },
        )

    if cfg.cot.index.enabled:
        _index_rows(rows, "COT Index Composite", cfg.cot.index)

    if cfg.cot.z_score.enabled:
        thr = cfg.cot.z_score.thresholds
        sc = cfg.cot.z_score.scores
        _append_rows(
            rows,
            "COT Z-Score",
            {
                "enabled": bool(cfg.cot.z_score.enabled),
                "weight": float(cfg.cot.z_score.weight),
                "window_weeks": int(cfg.cot.z_score.window_weeks),
                "threshold_strong_bear": float(thr.strong_bear),
                "threshold_strong_bull": float(thr.strong_bull),
                "score_strong_bear": float(sc.strong_bear),
                "score_strong_bull": float(sc.strong_bull),
                "asof_tolerance_days": int(cfg.cot.asof_tolerance_days),
            },
        )

    if cfg.interaction.enabled:
        _append_rows(
            rows,
            "Interaction",
            {
                "enabled": bool(cfg.interaction.enabled),
                "mode": cfg.interaction.mode,
                "vix_strong": float(cfg.interaction.vix_strong),
                "cot_strong": float(cfg.interaction.cot_strong),
                "boost": float(cfg.interaction.boost),
            },
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Factor"] = out["Factor"].astype(str)
    out["Parameter"] = out["Parameter"].astype(str)
    out["Value"] = out["Value"].map(_display_value)
    return out


def candlestick(
    df: pd.DataFrame,
    title: str,
    padding_days: int | None = None,
    x_range_min=None,
    x_range_max=None,
) -> go.Figure:
    padding_days = padding_days if padding_days is not None else get_settings().ui.plot_padding_days
    fig = go.Figure(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=title.split()[0],
        )
    )
    _set_x_range(fig, x_range_min, x_range_max, padding_days)
    if x_range_min is None or x_range_max is None:
        _pad(fig, df["date"], padding_days)

    fig.update_layout(
        title=title,
        template=DEFAULT_TEMPLATE,
        height=600,
        xaxis_rangeslider_visible=False,
        yaxis_title="Цена в USD",
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def vix_deviation(
    df: pd.DataFrame,
    *,
    asset: str,
    padding_days: int | None = None,
    x_range_min=None,
    x_range_max=None,
) -> go.Figure:
    padding_days = padding_days if padding_days is not None else get_settings().ui.plot_padding_days
    cfg = _engine_cfg(asset).vix

    df = add_vix_deviation_indicators(df, window=int(cfg.rolling_window_days))
    bands = _rolling_bands(df["deviation_pct"], int(cfg.levels_lookback_points), list(cfg.sigma_levels))
    latest_levels = {c: float(bands[c].dropna().iloc[-1]) for c in bands.columns if not bands[c].dropna().empty}
    active_keys = {x["key"] for x in get_vix_scoring_levels(latest_levels, cfg.sigma_levels)}

    plot_df = pd.concat([df[["date", "deviation_pct"]], bands], axis=1)
    plot_df = _clip_display(plot_df, x_range_min, x_range_max)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["deviation_pct"], mode="lines", name="Deviation %", line=dict(color="deepskyblue", width=2)))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["mean"], mode="lines", name="Mean", line=dict(color="yellow", width=2)))

    for key in sorted(active_keys, key=lambda x: (x[0], int(x[1:-1]))):
        color = "red" if key.startswith("+") else "limegreen"
        dash = "dot" if "1σ" in key else "dash" if "2σ" in key else "solid"
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df[key], mode="lines", name=key, line=dict(color=color, dash=dash, width=2)))

    _set_x_range(fig, x_range_min, x_range_max, padding_days)
    if x_range_min is None or x_range_max is None:
        _pad(fig, plot_df["date"], padding_days)

    fig.update_layout(
        title=f"VIX отклонение с порогами очков ({int(cfg.rolling_window_days)}д средняя, {int(cfg.levels_lookback_points)}д уровни)",
        yaxis_title="Отклонение (%)",
        template=DEFAULT_TEMPLATE,
        height=420,
        hovermode="x unified",
        showlegend=True,
    )
    return fig


def cot_index(
    df: pd.DataFrame,
    asset: str,
    padding_days: int | None = None,
    x_range_min=None,
    x_range_max=None,
) -> go.Figure:
    padding_days = padding_days if padding_days is not None else get_settings().ui.plot_padding_days
    cfg = _engine_cfg(asset).cot.index

    df = build_cot_indicators(df, enabled=True, window_weeks=int(cfg.window_weeks))
    q = cfg.quantiles
    qmap = {"p5": float(q.p5), "p10": float(q.p10), "p90": float(q.p90), "p95": float(q.p95)}
    comm_q = _rolling_quantiles(df["COT_Index_Comm"], int(cfg.quantile_lookback_points), qmap).add_prefix("comm_")
    large_q = _rolling_quantiles(df["COT_Index_Large_Inverted"], int(cfg.quantile_lookback_points), qmap).add_prefix("large_")

    plot_df = pd.concat(
        [
            df[["date", "COT_Index_Large_Inverted", "COT_Index_Comm"]],
            comm_q,
            large_q,
        ],
        axis=1,
    )
    plot_df = _clip_display(plot_df, x_range_min, x_range_max)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["COT_Index_Large_Inverted"], name="Large Inverted", line=dict(color="deepskyblue", width=2)))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["COT_Index_Comm"], name="Commercial", line=dict(color="orange", width=2)))

    for key, color, dash in [("p95", "limegreen", "dot"), ("p90", "limegreen", "dash"), ("p10", "red", "dash"), ("p5", "red", "dot")]:
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df[f"comm_{key}"], name=f"Comm {key}", line=dict(color=color, dash=dash, width=1)))
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df[f"large_{key}"], name=f"Large {key}", line=dict(color=color, dash=dash, width=1)))

    min_date = plot_df["date"].min()
    max_date = plot_df["date"].max()
    for label, thr, color in [
        ("Comm bull", cfg.scoring.comm.abs_thresholds.bull, "green"),
        ("Comm strong bull", cfg.scoring.comm.abs_thresholds.strong_bull, "green"),
        ("Comm bear", cfg.scoring.comm.abs_thresholds.bear, "red"),
        ("Comm strong bear", cfg.scoring.comm.abs_thresholds.strong_bear, "red"),
        ("Large bull", cfg.scoring.large_inv.abs_thresholds.bull, "green"),
        ("Large strong bull", cfg.scoring.large_inv.abs_thresholds.strong_bull, "green"),
        ("Large bear", cfg.scoring.large_inv.abs_thresholds.bear, "red"),
        ("Large strong bear", cfg.scoring.large_inv.abs_thresholds.strong_bear, "red"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[min_date, max_date],
                y=[float(thr), float(thr)],
                mode="lines",
                name=label,
                line=dict(color=color, dash="dash", width=1),
            )
        )

    _set_x_range(fig, x_range_min, x_range_max, padding_days)
    if x_range_min is None or x_range_max is None:
        _pad(fig, plot_df["date"], padding_days)

    fig.update_layout(
        title=f"COT индекс {asset} ({int(cfg.window_weeks)}н) — с порогами очков ({int(cfg.quantile_lookback_points)}д)",
        yaxis_title="Проценты",
        template=DEFAULT_TEMPLATE,
        height=460,
        hovermode="x unified",
        showlegend=True,
    )
    return fig


def net_positions(df: pd.DataFrame, padding_days: int | None = None, x_range_min=None, x_range_max=None) -> go.Figure:
    padding_days = padding_days if padding_days is not None else get_settings().ui.plot_padding_days
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["Comm_Net"], name="Commercial", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["Large_Specs_Net"], name="Large", line=dict(color="limegreen")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["Small_Traders_Net"], name="Small", line=dict(color="deepskyblue")))
    _set_x_range(fig, x_range_min, x_range_max, padding_days)
    if x_range_min is None or x_range_max is None:
        _pad(fig, df["date"], padding_days)
    fig.update_layout(title="Чистые позиции", yaxis_title="Чистые контрактры", template=DEFAULT_TEMPLATE, height=400, hovermode="x unified", showlegend=False)
    return fig


def z_score(
    df: pd.DataFrame,
    *,
    asset: str,
    padding_days: int | None = None,
    x_range_min=None,
    x_range_max=None,
) -> go.Figure:
    padding_days = padding_days if padding_days is not None else get_settings().ui.plot_padding_days
    cfg = _engine_cfg(asset).cot.z_score

    df = calculate_cot_z_score(df, enabled=True, window=int(cfg.window_weeks))
    plot_df = _clip_display(df[["date", "Z_Score_Comm"]], x_range_min, x_range_max)
    thr = cfg.thresholds

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["Z_Score_Comm"], name="Z-Score", line=dict(color="yellow", width=2)))
    fig.add_hline(y=float(thr.strong_bull), line_color="green", line_dash="dash", annotation_text=f"strong_bull {float(thr.strong_bull):.2f}")
    fig.add_hline(y=float(thr.strong_bear), line_color="red", line_dash="dash", annotation_text=f"strong_bear {float(thr.strong_bear):.2f}")

    _set_x_range(fig, x_range_min, x_range_max, padding_days)
    if x_range_min is None or x_range_max is None:
        _pad(fig, plot_df["date"], padding_days)

    fig.update_layout(
        title=f"COT Z-Score (Commercial, {int(cfg.window_weeks)}н)",
        yaxis_title="Z-Score",
        template=DEFAULT_TEMPLATE,
        height=400,
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def open_interest(df: pd.DataFrame, asset: str, padding_days: int | None = None, x_range_min=None, x_range_max=None) -> go.Figure:
    padding_days = padding_days if padding_days is not None else get_settings().ui.plot_padding_days
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["open_interest_all"], name="Open Interest", line=dict(color="purple", width=2)))
    _set_x_range(fig, x_range_min, x_range_max, padding_days)
    if x_range_min is None or x_range_max is None:
        _pad(fig, df["date"], padding_days)
    fig.update_layout(title=f"Открытый интерес {asset}", yaxis_title="Контракты", template=DEFAULT_TEMPLATE, height=400, hovermode="x unified", showlegend=False)
    return fig


def liquidity_vacuum(
    df_btc: pd.DataFrame,
    df_dxy: pd.DataFrame,
    df_us10y: pd.DataFrame,
    padding_days: int | None = None,
    x_range_min=None,
    x_range_max=None,
) -> go.Figure:
    padding_days = padding_days if padding_days is not None else get_settings().ui.plot_padding_days
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_btc["date"], y=df_btc["close"], name="BTC", yaxis="y1", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df_dxy["date"], y=df_dxy["close"], name="DXY", yaxis="y2", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df_us10y["date"], y=df_us10y["close"], name="US10Y", yaxis="y3", line=dict(color="purple")))
    min_date = min(df_btc["date"].min(), df_dxy["date"].min(), df_us10y["date"].min())
    max_date = max(df_btc["date"].max(), df_dxy["date"].max(), df_us10y["date"].max())
    _set_x_range(fig, x_range_min, x_range_max, padding_days)
    if x_range_min is None or x_range_max is None:
        _pad(fig, pd.Series([min_date, max_date]), padding_days)
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title="Вакуум ликвидности: BTC / DXY / US10Y",
        yaxis=dict(title="Цены", side="left"),
        yaxis2=dict(overlaying="y", visible=False),
        yaxis3=dict(overlaying="y", visible=False),
        height=400,
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def rolling_correlation(
    df_btc: pd.DataFrame,
    df_spx: pd.DataFrame,
    window: int = 60,
    min_periods: int = 20,
    padding_days: int | None = None,
    x_range_min=None,
    x_range_max=None,
) -> go.Figure:
    padding_days = padding_days if padding_days is not None else get_settings().ui.plot_padding_days
    btc_min, btc_max = df_btc["date"].min().normalize(), df_btc["date"].max().normalize()
    spx_min, spx_max = df_spx["date"].min().normalize(), df_spx["date"].max().normalize()
    start, end = min(btc_min, spx_min), max(btc_max, spx_max)
    date_range = pd.date_range(start=start, end=end, freq="D")
    btc_series = df_btc.assign(date=df_btc["date"].dt.normalize()).set_index("date")["close"].reindex(date_range)
    spx_series = df_spx.assign(date=df_spx["date"].dt.normalize()).set_index("date")["close"].reindex(date_range).ffill()
    prices = pd.DataFrame({"btc": btc_series, "spx": spx_series}).dropna(subset=["btc"])
    corr_series = prices["btc"].rolling(window=window, min_periods=min_periods).corr(prices["spx"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=corr_series.index, y=corr_series.values, name=f"{window}d corr", line=dict(color="cyan", width=2)))
    fig.add_hline(y=0.8, line_dash="dash", line_color="red")
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray")
    fig.add_hline(y=-0.8, line_dash="dash", line_color="green")
    _set_x_range(fig, x_range_min, x_range_max, padding_days)
    if x_range_min is None or x_range_max is None:
        _pad(fig, pd.Series(corr_series.index), padding_days)
    fig.update_layout(template=DEFAULT_TEMPLATE, title=f"BTC / S&P 500 Скользящая корреляция ({window}д)", yaxis=dict(range=[-1, 1], title="Корреляция"), height=400, hovermode="x unified", showlegend=False)
    return fig


def equity_curve_chart(
    df: pd.DataFrame,
    initial_capital: float,
    signals: pd.DataFrame | None = None,
    padding_days: int | None = None,
    x_range_min=None,
    x_range_max=None,
) -> go.Figure:
    padding_days = padding_days if padding_days is not None else get_settings().ui.plot_padding_days
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="No equity data", template=DEFAULT_TEMPLATE)
        return fig

    fig.add_trace(go.Scatter(x=df["date"], y=df["Equity"], mode="lines", name="Доходность «Компас»", line=dict(color="#00ff9d", width=3)))
    if "close" in df.columns and df["close"].iloc[0] > 0:
        bh_coins = initial_capital / df["close"].iloc[0]
        bh_equity = bh_coins * df["close"]
        fig.add_trace(go.Scatter(x=df["date"], y=bh_equity, mode="lines", name="«Купи & Держи»", line=dict(color="deepskyblue", width=2, dash="dash")))
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", annotation_text="Стартовый капитал")
    _set_x_range(fig, x_range_min, x_range_max, padding_days)
    if x_range_min is None or x_range_max is None:
        _pad(fig, df["date"], padding_days)
    fig.update_layout(title="Кривая капитала: «Компас» vs «Купи & Держи»", yaxis_title="Доходность (USD)", template=DEFAULT_TEMPLATE, height=520, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(tickformat="%d.%m.%y")
    return fig