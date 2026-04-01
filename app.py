from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import streamlit as st

from main import main as update_data
from src.analytics.signal_generator import generate_conclusion
from src.config.settings import get_settings
from src.constants import VERDICT_NO_DATA
from src.services.data_loader import dataset_checks_frame, datasets_max_date, filter_df, load_dataset, validate_datasets
from src.ui.dashboards import btc_dashboard, eth_dashboard, macro_dashboard, trend_validation_dashboard
from src.utils.logging_config import configure_logging
from src.utils.pandas_utils import ensure_datetime_sorted

configure_logging(level=logging.INFO)
st.set_page_config(page_title="СИК — «Компас»", layout="wide")
st.title("Система интерпретации крипторынка — «Компас»")

settings = get_settings()
DATASETS = list(settings.files)
st.session_state.setdefault("_data_v", 0)

if st.button("Обновить все данные"):
    with st.spinner("Скачиваю и обрабатываю данные…"):
        update_data()
        st.session_state["_data_v"] += 1
        st.cache_data.clear()
    st.success("Данные обновлены!")


@st.cache_data(show_spinner=False)
def _cached_ds(name: str, ver: int) -> pd.DataFrame | None:
    return load_dataset(name)


def _cot_default_start(cot_df: pd.DataFrame | None, asset_min: dt.date, weeks_back: int) -> dt.date:
    if cot_df is None or cot_df.empty or "date" not in cot_df.columns:
        return asset_min
    dates = sorted(pd.to_datetime(cot_df["date"], errors="coerce").dropna().dt.date.unique())
    if not dates:
        return asset_min
    return max(asset_min, dates[0] if len(dates) <= weeks_back + 5 else dates[-weeks_back - 1])


def _max_date_or_fallback(dfs_map: dict[str, pd.DataFrame | None], names: list[str] | tuple[str, ...], fallback: dt.date) -> dt.date:
    ts = datasets_max_date(dfs_map, names)
    return fallback if ts is None else ts.date()


def _filter_many(dfs_map: dict[str, pd.DataFrame | None], names: list[str] | tuple[str, ...], start: dt.date, end: dt.date):
    return tuple(filter_df(dfs_map.get(name), start, end) for name in names)


def _asset_filtered(dfs_map: dict[str, pd.DataFrame | None], asset: str, start: dt.date, end: dt.date):
    key = asset.lower()
    return _filter_many(dfs_map, [key, "vix", f"{key}_cot"], start, end)


def _macro_filtered(dfs_map: dict[str, pd.DataFrame | None], start: dt.date, end: dt.date):
    return _filter_many(dfs_map, ["btc", "spx", "dxy", "us10y"], start, end)


def _history_until(df: pd.DataFrame | None, end: dt.date) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame()
    out = ensure_datetime_sorted(df)
    end_ts = pd.Timestamp(end).normalize()
    return out[out["date"] <= end_ts].reset_index(drop=True)


def _asset_dashboard_data(dfs_map: dict[str, pd.DataFrame | None], asset: str, start: dt.date, end: dt.date):
    key = asset.lower()
    return (
        filter_df(dfs_map.get(key), start, end),
        _history_until(dfs_map.get("vix"), end),
        filter_df(dfs_map.get(f"{key}_cot"), start, end),
        _history_until(dfs_map.get(f"{key}_cot"), end),
    )


def _fmt_signed_or_dash(value: object) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    return f"{float(value):+.2f}"


dfs = {name: _cached_ds(name, st.session_state["_data_v"]) for name in DATASETS}
checks = validate_datasets(dfs)
invalid_checks = [c for c in checks if c.status != "OK"]
if invalid_checks:
    st.error("Обнаружены проблемы с датасетами. Trend Validation и Conclusion заблокированы, пока ошибки не исправлены.")
    st.dataframe(dataset_checks_frame(checks), width="stretch")
    st.stop()

btc_cot_df, eth_cot_df = dfs.get("btc_cot"), dfs.get("eth_cot")
btc_min_date = max(
    settings.assets.btc_cot_min_date,
    pd.to_datetime(btc_cot_df["date"]).min().date() if btc_cot_df is not None and not btc_cot_df.empty else settings.assets.btc_cot_min_date,
)
eth_min_date = max(
    settings.assets.eth_cot_min_date,
    pd.to_datetime(eth_cot_df["date"]).min().date() if eth_cot_df is not None and not eth_cot_df.empty else settings.assets.eth_cot_min_date,
)

weeks_back = settings.cot.default_weeks
all_fallback = dt.date.today()
default_btc_start = _cot_default_start(btc_cot_df, btc_min_date, weeks_back)
default_eth_start = _cot_default_start(eth_cot_df, eth_min_date, weeks_back)
macro_min_date = settings.assets.macro_min_date
conclusion_min_date = settings.assets.conclusion_min_date

btc_max_date = _max_date_or_fallback(dfs, ["btc", "btc_cot", "vix"], all_fallback)
eth_max_date = _max_date_or_fallback(dfs, ["eth", "eth_cot", "vix"], all_fallback)
macro_max_date = _max_date_or_fallback(dfs, ["btc", "spx", "dxy", "us10y"], all_fallback)
conclusion_max_date = _max_date_or_fallback(dfs, ["btc", "eth", "btc_cot", "eth_cot", "vix"], all_fallback)
default_macro_start = max(macro_min_date, dt.date(macro_max_date.year - settings.ui.default_years, 1, 1))

tab_btc, tab_eth, tab_macro, tab_conclusion, tab_validation = st.tabs(
    ["BITCOIN Dashboard", "ETH Dashboard", "Macro Context", "Conclusion", "Trend Validation"]
)
slider_step = dt.timedelta(days=int(settings.ui.slider_step_days))

with tab_btc:
    start_date, end_date = st.slider(
        "Выберите диапазон дат для BTC",
        min_value=btc_min_date,
        max_value=btc_max_date,
        value=(default_btc_start, btc_max_date),
        step=slider_step,
        format="DD.MM.YYYY",
        key="btc_slider",
    )
    btc_dashboard(*_asset_dashboard_data(dfs, "BTC", start_date, end_date), start_date, end_date)

with tab_eth:
    start_date, end_date = st.slider(
        "Выберите диапазон дат для ETH",
        min_value=eth_min_date,
        max_value=eth_max_date,
        value=(default_eth_start, eth_max_date),
        step=slider_step,
        format="DD.MM.YYYY",
        key="eth_slider",
    )
    eth_dashboard(*_asset_dashboard_data(dfs, "ETH", start_date, end_date), start_date, end_date)

with tab_macro:
    start_date, end_date = st.slider(
        "Выберите диапазон дат для Macro Context",
        min_value=macro_min_date,
        max_value=macro_max_date,
        value=(default_macro_start, macro_max_date),
        step=slider_step,
        format="DD.MM.YYYY",
        key="macro_slider",
    )
    macro_dashboard(*_macro_filtered(dfs, start_date, end_date))

with tab_conclusion:
    end_date = st.slider(
        "Выберите дату обзора (по состоянию на)",
        min_value=conclusion_min_date,
        max_value=conclusion_max_date,
        value=conclusion_max_date,
        step=slider_step,
        format="DD.MM.YYYY",
        key="concl_slider",
    )
    as_of_ts = pd.Timestamp(end_date).normalize()
    live_dfs = {k: v for k, v in dfs.items() if v is not None}
    per_asset, combined_score, combined_verdict = generate_conclusion(live_dfs, as_of=as_of_ts)

    st.subheader("«Компас» — факторы и режим")
    has_rows = False
    for asset, item in per_asset.items():
        df_table, total, asset_verdict, confidence, narrative = item
        st.markdown(f"### {asset}")
        if not df_table.empty:
            has_rows = True
            display_df = df_table.rename(columns={
                "Factor": "Фактор",
                "Status": "Статус",
                "Value": "Значение",
                "Lookup Date": "Дата запроса",
                "Source Date": "Дата источника",
                "Thresholds": "Пороги",
                "Score": "Балл",
                "Rationale": "Обоснование"
            })
            st.dataframe(
                display_df.style.format({"Балл": _fmt_signed_or_dash}),
                width="stretch"
            )
        total_txt = "No data" if total is None else f"{total:+.2f}"
        st.markdown(f"**Итог ({asset}): {total_txt} → {asset_verdict}**  \n**Уверенность:** {confidence:.2f}")
        if asset_verdict == VERDICT_NO_DATA:
            st.warning(f"{asset}: сигнал недоступен — точная причина указана в diagnostics table выше.")
        if narrative:
            st.markdown("**Объективная оценка:**")
            st.markdown(narrative)

    if not has_rows:
        st.warning("Нет данных в выбранном диапазоне.")
    else:
        combined_txt = "No data" if combined_score is None else f"{combined_score:+.2f}"
        st.markdown(
            f"""---
### <span style="font-size:24px">Комбинированный обзор рынка</span>
**Суммарный балл**: {combined_txt}
**Вердикт**: **{combined_verdict}**
""",
            unsafe_allow_html=True,
        )

with tab_validation:
    trend_validation_dashboard(dfs, btc_min_date, eth_min_date, btc_max_date, eth_max_date)

st.caption("Система интерпретации крипторынка на основе макроиндикаторов, отчётов COT и рыночного режима.")