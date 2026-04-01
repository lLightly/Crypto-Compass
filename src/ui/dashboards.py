from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analytics.engines import get_asset_engine
from src.analytics.trend_validation import run_trend_validation
from src.config.settings import get_settings
from src.constants import VERDICT_BEARISH, VERDICT_BULLISH, VERDICT_NEUTRAL, VERDICT_NO_DATA
from src.ui import components


def _show_signal_parameters(asset: str, *, key: str) -> None:
    with st.expander(f"{asset} — параметры генерации сигналов", expanded=False):
        st.dataframe(components.signal_parameter_table(asset), width="stretch")


def _asset_dashboard(
    asset: str,
    px: pd.DataFrame,
    vix_hist: pd.DataFrame,
    cot_view: pd.DataFrame,
    cot_hist: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
):
    a = asset.lower()
    x0 = pd.Timestamp(start_date).normalize()
    x1 = pd.Timestamp(end_date).normalize()

    _show_signal_parameters(asset, key=f"{a}_params")

    if not px.empty:
        st.plotly_chart(components.candlestick(px, f"{asset} цена"), width="stretch", key=f"{a}_price")

    if not vix_hist.empty:
        st.plotly_chart(
            components.vix_deviation(vix_hist, asset=asset, x_range_min=x0, x_range_max=x1),
            width="stretch",
            key=f"{a}_vix",
        )

    if cot_hist.empty:
        return

    eng = get_asset_engine(asset)
    if eng.cfg.cot.index.enabled:
        st.plotly_chart(
            components.cot_index(cot_hist, asset=asset, x_range_min=x0, x_range_max=x1),
            width="stretch",
            key=f"{a}_cot_idx",
        )

    if not cot_view.empty:
        st.plotly_chart(components.net_positions(cot_view), width="stretch", key=f"{a}_net")

    if eng.cfg.cot.z_score.enabled:
        st.plotly_chart(
            components.z_score(cot_hist, asset=asset, x_range_min=x0, x_range_max=x1),
            width="stretch",
            key=f"{a}_z",
        )

    if not cot_view.empty:
        st.plotly_chart(components.open_interest(cot_view, asset=asset), width="stretch", key=f"{a}_oi")


def btc_dashboard(
    px: pd.DataFrame,
    vix_hist: pd.DataFrame,
    cot_view: pd.DataFrame,
    cot_hist: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
):
    _asset_dashboard("BTC", px, vix_hist, cot_view, cot_hist, start_date, end_date)


def eth_dashboard(
    px: pd.DataFrame,
    vix_hist: pd.DataFrame,
    cot_view: pd.DataFrame,
    cot_hist: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
):
    _asset_dashboard("ETH", px, vix_hist, cot_view, cot_hist, start_date, end_date)


def macro_dashboard(btc: pd.DataFrame, spx: pd.DataFrame, dxy: pd.DataFrame, us10y: pd.DataFrame):
    xs = [df for df in (btc, spx, dxy, us10y) if not df.empty and "date" in df.columns]
    x0 = min((df["date"].min() for df in xs), default=None)
    x1 = max((df["date"].max() for df in xs), default=None)

    if not btc.empty and not dxy.empty and not us10y.empty:
        st.plotly_chart(
            components.liquidity_vacuum(btc, dxy, us10y, x_range_min=x0, x_range_max=x1),
            width="stretch",
            key="macro_liq",
        )
    if not btc.empty and not spx.empty:
        st.plotly_chart(
            components.rolling_correlation(btc, spx, window=60, min_periods=1, x_range_min=x0, x_range_max=x1),
            width="stretch",
            key="macro_corr",
        )


def _fmt_pct(x: float | None) -> str:
    return "No data" if x is None or pd.isna(x) else f"{x * 100:+.2f}%"


def _fmt_num(x: float | None) -> str:
    return "No data" if x is None or pd.isna(x) else f"{x:.2f}"


def _fmt_dd(x: float | None) -> str:
    return "No data" if x is None or pd.isna(x) else f"{x * 100:.2f}%"


def _aligned_points(sig: pd.DataFrame, eq: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if sig.empty or eq.empty or date_col not in sig.columns:
        return pd.DataFrame()

    cols_needed = [date_col, "verdict", "report_date", "signal_date", "exec_date", "data_status", "reason", "block_reason"]
    cols = list(dict.fromkeys(c for c in cols_needed if c in sig.columns))

    pts = sig.loc[:, cols].copy()
    pts[date_col] = pd.to_datetime(pts[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    pts = pts.dropna(subset=[date_col]).copy()

    pts["date"] = pts[date_col]
    if date_col != "date":
        pts = pts.drop(columns=[date_col])

    pts = pts.sort_values("date").reset_index(drop=True)

    eq_sorted = eq[["date", "Equity"]].copy()
    eq_sorted["date"] = pd.to_datetime(eq_sorted["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    eq_sorted = eq_sorted.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return pd.merge_asof(pts, eq_sorted, on="date", direction="forward")


def _signal_table_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "report_date",
        "signal_date",
        "exec_date",
        "verdict",
        "total_score",
        "position",
        "confidence",
        "data_status",
        "reason",
        "block_reason",
    ]
    suffix_order = {
        "__status": 0,
        "__score": 1,
        "__value": 2,
        "__lookup_date": 3,
        "__source_date": 4,
        "__thresholds": 5,
        "__rationale": 6,
    }
    extra = [c for c in df.columns if "__" in c]
    extra.sort(key=lambda c: (c.rsplit("__", 1)[0], suffix_order.get("__" + c.rsplit("__", 1)[1], 99), c))
    return [c for c in base + extra if c in df.columns]


def _pretty_factor_name(key: str) -> str:
    mapping = {
        "vix_risk_regime": "VIX Risk Regime",
        "cot_index_composite": "COT Index Composite",
        "cot_z_score": "COT Z-Score",
    }
    return mapping.get(key, key.replace("_", " ").title())


def _factor_breakdown_from_row(row: pd.Series) -> pd.DataFrame:
    grouped: dict[str, dict[str, object]] = {}
    for col, value in row.items():
        if "__" not in col:
            continue
        factor_key, suffix = col.rsplit("__", 1)
        grouped.setdefault(factor_key, {})[suffix] = value

    rows = []
    for factor_key, payload in grouped.items():
        rows.append(
            {
                "Factor": _pretty_factor_name(factor_key),
                "Status": payload.get("status"),
                "Score": payload.get("score"),
                "Value": payload.get("value"),
                "Lookup Date": payload.get("lookup_date"),
                "Source Date": payload.get("source_date"),
                "Thresholds": payload.get("thresholds"),
                "Rationale": payload.get("rationale"),
            }
        )
    return pd.DataFrame(rows).sort_values("Factor").reset_index(drop=True)


def _signal_label(row: pd.Series) -> str:
    ts = pd.Timestamp(row["signal_date"]).strftime("%Y-%m-%d")
    return f"{ts} | {row.get('verdict', '—')} | {row.get('data_status', '—')}"


def trend_validation_dashboard(dfs, btc_min: dt.date, eth_min: dt.date, btc_max: dt.date, eth_max: dt.date):
    s = get_settings()
    st.header("Историческое тестирование стратегии")
    st.caption(
        "report_date = дата COT, signal_date = дата публикации, exec_date = первый следующий торговый день после signal_date. "
        "В Trend Validation рыночные факторы читаются не позже предыдущего дневного бара, чтобы исключить ошибку просмотра будущего."
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        asset = st.selectbox("Актив", ["BTC", "ETH"], key="trend_asset")
    ak = asset.lower()

    with c2:
        initial_capital = st.number_input("Стартовый капитал ($)", min_value=1.0, value=100.0, step=10.0, key=f"trend_capital_{ak}")

    _show_signal_parameters(asset, key=f"trend_{ak}_params")

    min_d = {"BTC": btc_min, "ETH": eth_min}[asset]
    max_d = {"BTC": btc_max, "ETH": eth_max}[asset]
    default_s = max(min_d, dt.date(max_d.year - s.ui.default_years, 1, 1))
    start_date, end_date = st.slider(
        f"Диапазон дат для {asset}",
        min_value=min_d,
        max_value=max_d,
        value=(default_s, max_d),
        format="DD.MM.YYYY",
        key=f"trend_slider_{ak}",
    )

    result = run_trend_validation(
        dfs,
        asset,
        initial_capital=float(initial_capital),
        start_date=start_date,
        end_date=end_date,
        fee_pct=s.backtest.fee_pct,
        slippage_pct=s.backtest.slippage_pct,
        rf_rate=s.backtest.rf_rate,
        periods_per_year=s.backtest.periods_per_year,
        trailing_stop_pct=s.backtest.trailing_stop_pct,
    )

    for w in result.warnings:
        st.warning(w)
    if result.equity_curve.empty:
        st.error("Trend Validation заблокирован: см. предупреждения выше для точной причины.")
        return

    fig = components.equity_curve_chart(result.equity_curve, initial_capital=float(initial_capital))
    if not result.daily.empty:
        cmap = {
            VERDICT_BULLISH: "rgba(0,255,0,0.2)",
            VERDICT_BEARISH: "rgba(255,0,0,0.2)",
            VERDICT_NEUTRAL: "rgba(128,128,128,0.2)",
            VERDICT_NO_DATA: "rgba(0,0,0,0.2)",
        }
        d = result.daily.reset_index(drop=True)
        start_i, prev = 0, d.iloc[0]["verdict"]
        for i in range(1, len(d) + 1):
            cur = None if i == len(d) else d.iloc[i]["verdict"]
            if cur != prev:
                fig.add_shape(
                    type="rect",
                    x0=d.iloc[start_i]["date"],
                    x1=d.iloc[i - 1]["date"],
                    y0=0,
                    y1=1,
                    yref="paper",
                    fillcolor=cmap.get(prev, "rgba(255,255,255,0)"),
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
                start_i, prev = i, cur

    if not result.signals.empty:
        eq = result.equity_curve[["date", "Equity"]].copy()
        sig_pts = _aligned_points(result.signals, eq, "signal_date")
        exe_pts = _aligned_points(result.signals, eq, "exec_date")
        colors = {VERDICT_BULLISH: "green", VERDICT_BEARISH: "red", VERDICT_NEUTRAL: "yellow", VERDICT_NO_DATA: "black"}

        for verdict, color in colors.items():
            spts = sig_pts[sig_pts["verdict"] == verdict]
            if not spts.empty:
                fig.add_scatter(
                    x=spts["date"],
                    y=spts["Equity"],
                    mode="markers",
                    marker=dict(color=color, size=10, symbol="diamond", line=dict(width=1, color="black")),
                    name=f"{verdict} signal",
                    hovertemplate=(
                        "Signal %{x|%d.%m.%Y}<br>Verdict: "
                        + verdict
                        + "<br>Status: %{customdata[0]}<br>Reason: %{customdata[1]}<extra></extra>"
                    ),
                    customdata=spts[["data_status", "reason"]].fillna("—"),
                )
            epts = exe_pts[exe_pts["verdict"] == verdict]
            if not epts.empty:
                fig.add_scatter(
                    x=epts["date"],
                    y=epts["Equity"],
                    mode="markers",
                    marker=dict(color=color, size=9, symbol="triangle-right", line=dict(width=1, color="black")),
                    name=f"{verdict} exec",
                    hovertemplate=(
                        "Exec %{x|%d.%m.%Y}<br>Verdict: "
                        + verdict
                        + "<br>Status: %{customdata[0]}<br>Reason: %{customdata[1]}<extra></extra>"
                    ),
                    customdata=epts[["data_status", "reason"]].fillna("—"),
                )

    st.plotly_chart(fig, width="stretch", key=f"equity_curve_compass_{ak}")

    m = result.metrics
    st.subheader("Метрики стратегии «Компас»")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Cовокупный среднегодовой темп роста (CAGR) ", _fmt_pct(m.get("cagr")))
    a2.metric("Максимальная просадка (Max DD)", _fmt_dd(m.get("max_dd")))
    a3.metric("Коэффицент Калмар (Calmar ratio)", _fmt_num(m.get("calmar")))
    a4.metric("Коэффицент Шарп (Sharpe ratio)", _fmt_num(m.get("sharpe")))
    a5.metric("«Компас» доходность", _fmt_pct(m.get("total_return")))

    b1, b2, _, _, _ = st.columns(5)
    b1.metric("Точность тренда", "No data" if m.get("trend_accuracy") is None else f"{m['trend_accuracy'] * 100:.1f}%")
    b2.metric("Покрытие", "No data" if m.get("trend_coverage") is None else f"{m['trend_coverage'] * 100:.1f}%")

    st.subheader("Метрики стратегии «Купи & Держи»")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cовокупный среднегодовой темп роста (CAGR)", _fmt_pct(m.get("bh_cagr")))
    c2.metric("Максимальная просадка (Max DD)", _fmt_dd(m.get("bh_max_dd")))
    c3.metric("Коэффицент Калмар (Calmar ratio)", _fmt_num(m.get("bh_calmar")))
    c4.metric("Коэффицент Шарп (Sharpe ratio)", _fmt_num(m.get("sharpe_bh")))
    c5.metric("B&H доходность", _fmt_pct(m.get("bh_total_return")))

    conf = result.confusion
    pie = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]], subplot_titles=("Бычий", "Медвежий"))
    pie.add_trace(go.Pie(labels=["Верно", "Неверно"], values=[conf.get("bull_correct", 0), conf.get("bull_wrong", 0)], hole=0.3, marker_colors=["green", "red"]), 1, 1)
    pie.add_trace(go.Pie(labels=["Верно", "Неверно"], values=[conf.get("bear_correct", 0), conf.get("bear_wrong", 0)], hole=0.3, marker_colors=["green", "red"]), 1, 2)
    pie.update_layout(title_text="Пончики точности", height=400)
    st.plotly_chart(pie, width="stretch", key=f"trend_confusion_{ak}")

    if st.checkbox("Показать таблицу режимов (сигналов + диагностика)", key=f"trend_show_signals_{ak}"):
        st.caption("reason всегда заполнен. block_reason заполняется только если data_status != OK.")
        cols = _signal_table_columns(result.signals)
        table_view = result.signals.sort_values("signal_date", ascending=False).head(120).copy()
        for col in ("reason", "block_reason"):
            if col in table_view.columns:
                table_view[col] = table_view[col].fillna("—")
        st.dataframe(
            table_view[cols].style.format({"total_score": "{:+.2f}", "confidence": "{:.2f}"}),
            width="stretch",
        )

    if not result.signals.empty and st.checkbox("Показать срабатывание выбранного сигнала", key=f"trend_show_breakdown_{ak}"):
        sig_sorted = result.signals.sort_values("signal_date", ascending=False).reset_index(drop=True)
        labels = [_signal_label(row) for _, row in sig_sorted.iterrows()]
        selected_label = st.selectbox("Сигнал", labels, index=0, key=f"trend_breakdown_select_{ak}")
        selected_row = sig_sorted.iloc[labels.index(selected_label)]
        st.markdown(
            f"**report_date:** {pd.Timestamp(selected_row['report_date']).date() if pd.notna(selected_row['report_date']) else '—'}  "
            f"**signal_date:** {pd.Timestamp(selected_row['signal_date']).date() if pd.notna(selected_row['signal_date']) else '—'}  "
            f"**exec_date:** {pd.Timestamp(selected_row['exec_date']).date() if pd.notna(selected_row['exec_date']) else '—'}"
        )
        if selected_row.get("reason"):
            st.info(str(selected_row["reason"]))
        if selected_row.get("block_reason"):
            st.warning(str(selected_row["block_reason"]))
        st.dataframe(_factor_breakdown_from_row(selected_row), width="stretch")

    if st.checkbox("Показать логи дневной доходности", key=f"trend_show_daily_{ak}"):
        st.caption("tgt_pos/pos — long-only позиции 0/1. На Neutral/Bearish они обязаны быть 0. trade = |pos - prev_pos|.")
        st.dataframe(result.daily.sort_values("date", ascending=False).head(200).style.format({"Equity": "{:,.2f}"}), width="stretch")