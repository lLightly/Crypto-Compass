from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.cot_scoring import calculate_cot_index_composite, calculate_cot_zscore_score
from src.analytics.indicators import build_cot_indicators
from src.analytics.scoring import get_vix_scoring_levels, vix_score
from src.analytics.statistics import add_vix_deviation_indicators, calculate_cot_z_score
from src.analytics.thresholds import get_deviation_levels, get_quantile_thresholds
from src.config.settings import AssetEngineSettings
from src.constants import VERDICT_BEARISH, VERDICT_BULLISH, VERDICT_NEUTRAL, VERDICT_NO_DATA
from src.utils.pandas_utils import STATUS_OK, STATUS_STALE, asof_value, ensure_datetime_sorted, slice_until

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactorDiagnostic:
    name: str
    status: str
    value: object
    lookup_date: pd.Timestamp | None
    source_date: pd.Timestamp | None
    thresholds: str
    score: float | None
    rationale: str


class BaseAssetEngine:
    def __init__(self, asset: str, cfg: AssetEngineSettings):
        self.asset = asset.upper()
        self.cfg = cfg

    @property
    def asset_key(self) -> str:
        return self.asset.lower()

    def _prepare(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out = {
            k: (ensure_datetime_sorted(v) if v is not None and not v.empty and "date" in v.columns else pd.DataFrame())
            for k, v in dfs.items()
        }

        vix = out.get("vix")
        if vix is not None and not vix.empty:
            if "close" not in vix.columns:
                raise ValueError("VIX dataset missing required column: close")
            out["vix"] = add_vix_deviation_indicators(vix, window=int(self.cfg.vix.rolling_window_days))

        cot_key = f"{self.asset_key}_cot"
        cot = out.get(cot_key)
        if cot is not None and not cot.empty:
            missing = [c for c in ("Comm_Net", "Large_Specs_Net") if c not in cot.columns]
            if missing:
                raise ValueError(f"{cot_key} missing required columns: {', '.join(missing)}")
            cot = build_cot_indicators(cot, enabled=self.cfg.cot.index.enabled, window_weeks=int(self.cfg.cot.index.window_weeks))
            cot = calculate_cot_z_score(cot, enabled=self.cfg.cot.z_score.enabled, window=int(self.cfg.cot.z_score.window_weeks))
            out[cot_key] = cot

        return out

    @staticmethod
    def _slice(dfs: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
        return {k: (slice_until(v, as_of) if not v.empty and "date" in v.columns else pd.DataFrame()) for k, v in dfs.items()}

    @staticmethod
    def _factor_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    @staticmethod
    def _diag_table(diags: list[FactorDiagnostic]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "Factor": d.name,
                "Status": d.status,
                "Value": d.value,
                "Lookup Date": d.lookup_date,
                "Source Date": d.source_date,
                "Thresholds": d.thresholds,
                "Score": d.score,
                "Rationale": d.rationale,
            }
            for d in diags
        )
        if "Value" in df.columns:
            df["Value"] = df["Value"].astype(str).replace("nan", "—")
        if "Lookup Date" in df.columns:
            df["Lookup Date"] = pd.to_datetime(df["Lookup Date"], errors="coerce")
        if "Source Date" in df.columns:
            df["Source Date"] = pd.to_datetime(df["Source Date"], errors="coerce")
        return df

    def _infer_asof(self, px: pd.DataFrame) -> pd.Timestamp:
        px = ensure_datetime_sorted(px)
        return pd.Timestamp("1970-01-01") if px.empty else pd.Timestamp(px["date"].iloc[-1]).normalize()

    @staticmethod
    def _error_signal_frame(message: str) -> pd.DataFrame:
        out = pd.DataFrame()
        out.attrs["error"] = message
        return out

    @staticmethod
    def _blocked_reason(table: pd.DataFrame) -> str:
        if table.empty or "Status" not in table.columns:
            return "No diagnostics available"
        bad = table[table["Status"] != STATUS_OK]
        if bad.empty:
            return ""
        return " | ".join(f"{r['Factor']}[{r['Status']}]: {r['Rationale']}" for _, r in bad.iterrows())

    def _signal_reason(self, table: pd.DataFrame) -> str:
        if table.empty:
            return "No diagnostics available"

        parts: list[str] = []
        for _, r in table.iterrows():
            factor = str(r.get("Factor", "")).strip() or "Unknown factor"
            status = str(r.get("Status", "")).strip() or "UNKNOWN"
            rationale = str(r.get("Rationale", "")).strip()
            if not rationale or rationale.lower() == "nan":
                rationale = "No rationale"
            parts.append(f"{factor}[{status}] {self._fmt_score(r.get('Score'))}: {rationale}")

        return " | ".join(parts)

    def _serialize_diag_columns(self, row: dict[str, object], table: pd.DataFrame) -> None:
        for _, r in table.iterrows():
            fk = self._factor_key(str(r["Factor"]))
            row[f"{fk}__status"] = r["Status"]
            row[f"{fk}__score"] = np.nan if r["Score"] is None else float(r["Score"])
            row[f"{fk}__value"] = r["Value"]
            row[f"{fk}__lookup_date"] = r["Lookup Date"]
            row[f"{fk}__source_date"] = r["Source Date"]
            row[f"{fk}__thresholds"] = r["Thresholds"]
            row[f"{fk}__rationale"] = r["Rationale"]

    def _signal_generation_error(self, dfs_full: dict[str, pd.DataFrame]) -> str | None:
        px = dfs_full.get(self.asset_key)
        if px is None or px.empty or "date" not in px.columns:
            return f"{self.asset}: price dataset missing or invalid; signal generation blocked"
        cot_key = f"{self.asset_key}_cot"
        cot = dfs_full.get(cot_key)
        if cot is None or cot.empty:
            return f"{self.asset}: COT dataset missing or empty; signal generation blocked"
        if "date" not in cot.columns:
            return f"{self.asset}: {cot_key} missing date column; signal generation blocked"
        missing = [c for c in ("Comm_Net", "Large_Specs_Net") if c not in cot.columns]
        if missing:
            return f"{self.asset}: {cot_key} missing columns: {', '.join(missing)}; signal generation blocked"
        return None

    @staticmethod
    def _latest_row_with_tolerance(
        hist: pd.DataFrame,
        *,
        lookup_date: pd.Timestamp,
        tolerance: pd.Timedelta | None,
    ) -> tuple[pd.Series | None, pd.Timestamp | None, str, str]:
        if hist is None or hist.empty:
            return None, None, "MISSING", f"No row <= {lookup_date.date().isoformat()}"

        row = hist.iloc[-1]
        source_date = pd.Timestamp(row["date"]).normalize()
        if tolerance is not None:
            age = lookup_date - source_date
            if age > tolerance:
                return (
                    row,
                    source_date,
                    STATUS_STALE,
                    f"Data too old: {source_date.date().isoformat()} (age {age.days}d > {tolerance.days}d)",
                )

        message = "Текущие данные" if source_date == lookup_date else f"Данные за {source_date.date().isoformat()} для {lookup_date.date().isoformat()}"
        return row, source_date, STATUS_OK, message

    @staticmethod
    def _status_reason(status: str, *, field: str) -> str:
        mapping = {
            "INSUFFICIENT_HISTORY": f"{field}: insufficient history to compute indicator",
            "ZERO_RANGE": f"{field}: rolling max == rolling min, index undefined",
            "ZERO_STD": f"{field}: rolling std == 0, z-score undefined",
            "INVALID_INPUT": f"{field}: source input is NaN or invalid",
            "INVALID": f"{field}: indicator is NaN/invalid",
        }
        return mapping.get(status, f"{field}: status={status}")

    @staticmethod
    def _fmt_value(value: object) -> str:
        if value is None:
            return "—"
        try:
            if pd.isna(value):
                return "—"
        except Exception:
            pass
        return str(value)

    def score_asset(
        self,
        dfs: dict[str, pd.DataFrame],
        as_of: pd.Timestamp | None = None,
        *,
        use_publication_lag: bool = True,
    ):
        px = dfs.get(self.asset_key)
        if px is None or px.empty or "date" not in px.columns:
            diag = FactorDiagnostic("Price Data", "MISSING", None, None, None, "", None, "Insufficient price data")
            table = self._diag_table([diag])
            return table, None, VERDICT_NO_DATA, 0.0, "- **Price Data** [MISSING]: Insufficient price data"

        as_of = self._infer_asof(px) if as_of is None else pd.Timestamp(as_of).normalize()
        pre = self._prepare(dfs)
        return self.score_asset_asof(self._slice(pre, as_of), as_of, use_publication_lag=use_publication_lag)

    def generate_signals(self, dfs_full: dict[str, pd.DataFrame]) -> pd.DataFrame:
        err = self._signal_generation_error(dfs_full)
        if err:
            return self._error_signal_frame(err)

        px = ensure_datetime_sorted(dfs_full[self.asset_key])
        if len(px) < 2:
            return self._error_signal_frame(f"{self.asset}: not enough price history for signal generation")

        pre = self._prepare(dfs_full)
        cot = pre.get(f"{self.asset_key}_cot")
        if cot is None or cot.empty or "date" not in cot.columns:
            return self._error_signal_frame(f"{self.asset}: prepared COT dataset is unavailable; signal generation blocked")

        report_dates = cot["date"].drop_duplicates().sort_values().reset_index(drop=True)
        if report_dates.empty:
            return self._error_signal_frame(f"{self.asset}: no COT report dates available; signal generation blocked")

        lag = pd.Timedelta(days=int(self.cfg.cot.publication_lag_days))
        signal_dates = (report_dates + lag).rename("signal_date")
        price_dates = px["date"].drop_duplicates().sort_values().reset_index(drop=True)
        rows: list[dict[str, object]] = []

        for idx, signal_date in enumerate(signal_dates):
            signal_date = pd.Timestamp(signal_date).normalize()
            if signal_date < price_dates.iloc[0] or signal_date > price_dates.iloc[-1]:
                continue

            sliced = self._slice(pre, signal_date)
            table, total, verdict, conf, _ = self.score_asset_asof(sliced, signal_date, use_publication_lag=True)
            pos = price_dates.searchsorted(signal_date.to_datetime64(), side="right")
            exec_date = pd.NaT if pos >= len(price_dates) else pd.Timestamp(price_dates.iloc[pos]).normalize()
            report_date = pd.Timestamp(report_dates.iloc[idx]).normalize()
            block_reason = self._blocked_reason(table)
            reason = self._signal_reason(table)

            row: dict[str, object] = {
                "report_date": report_date,
                "signal_date": signal_date,
                "date": signal_date,
                "exec_date": exec_date,
                "total_score": np.nan if total is None else float(total),
                "verdict": verdict,
                "position": int(verdict == VERDICT_BULLISH),
                "confidence": float(conf),
                "data_status": "OK" if not block_reason else "BLOCKED",
                "reason": reason,
                "block_reason": None if not block_reason else block_reason,
            }
            self._serialize_diag_columns(row, table)
            rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return self._error_signal_frame(f"{self.asset}: no signal dates overlap with available price history")
        return out.sort_values("signal_date").reset_index(drop=True)

    def score_asset_asof(
        self,
        dfs: dict[str, pd.DataFrame],
        as_of: pd.Timestamp,
        *,
        use_publication_lag: bool = True,
    ):
        diags: list[FactorDiagnostic] = []
        if self.cfg.vix.enabled:
            diags.append(self.score_vix(dfs, as_of, use_publication_lag=use_publication_lag))
        if self.cfg.cot.index.enabled:
            diags.append(self.score_cot_index(dfs, as_of, use_publication_lag=use_publication_lag))
        if self.cfg.cot.z_score.enabled:
            diags.append(self.score_cot_zscore(dfs, as_of, use_publication_lag=use_publication_lag))

        table = self._diag_table(diags)
        ok_scores = [d.score for d in diags if d.status == STATUS_OK and d.score is not None and not pd.isna(d.score)]
        blocked = any(d.status != STATUS_OK for d in diags)
        total = None if blocked or not ok_scores else float(np.sum(ok_scores))

        if total is not None and self._interaction_boost_applies(
            table.set_index("Factor")["Score"].get("VIX Risk Regime"),
            table.set_index("Factor")["Score"].get("COT Index Composite"),
        ):
            total *= float(self.cfg.interaction.boost)

        confidence = 0.0 if blocked else 1.0
        verdict = self.get_verdict(total=total, confidence=confidence)
        narrative = "\n".join(
            f"- **{d.name}** [{d.status}]: {d.rationale} (score {self._fmt_score(d.score)})" for d in diags
        )
        return table, (None if total is None else round(float(total), 2)), verdict, confidence, narrative

    def _cot_lookup_date(self, as_of: pd.Timestamp, use_publication_lag: bool) -> pd.Timestamp:
        as_of = pd.Timestamp(as_of).normalize()
        return as_of - pd.Timedelta(days=int(self.cfg.cot.publication_lag_days)) if use_publication_lag else as_of

    @staticmethod
    def _market_lookup_date(as_of: pd.Timestamp, use_publication_lag: bool) -> pd.Timestamp:
        as_of = pd.Timestamp(as_of).normalize()
        return as_of - pd.Timedelta(days=1) if use_publication_lag else as_of

    def score_vix(self, dfs: dict[str, pd.DataFrame], as_of: pd.Timestamp, *, use_publication_lag: bool = True) -> FactorDiagnostic:
        name = "VIX Risk Regime"
        vix = dfs.get("vix")
        if vix is None or vix.empty:
            return FactorDiagnostic(name, "MISSING", None, as_of, None, "", None, "VIX dataset unavailable")

        lookup_date = self._market_lookup_date(as_of, use_publication_lag)
        hist = slice_until(vix, lookup_date)
        if hist.empty:
            return FactorDiagnostic(name, "MISSING", None, lookup_date, None, "", None, "No VIX rows up to lookup date")

        tol = pd.Timedelta(days=int(self.cfg.vix.asof_tolerance_days))
        dev = asof_value(hist, as_of=lookup_date, value_col="deviation_pct", tolerance=tol)
        levels = get_deviation_levels(
            hist["deviation_pct"],
            sigma_levels=list(self.cfg.vix.sigma_levels),
            lookback_points=int(self.cfg.vix.levels_lookback_points),
        )
        if not levels:
            return FactorDiagnostic(name, "INVALID", None, lookup_date, dev.source_date, "", None, "VIX levels unavailable")

        thresholds = "; ".join(f"{x['key']}={x['value']:.2f}" for x in get_vix_scoring_levels(levels, self.cfg.vix.sigma_levels))
        if dev.status != STATUS_OK:
            return FactorDiagnostic(name, dev.status, None, lookup_date, dev.source_date, thresholds, None, dev.message)

        score, text = vix_score(float(dev.value), levels, self.cfg.vix.scoring, self.cfg.vix.sigma_levels)
        score *= float(self.cfg.vix.weight) * float(self.cfg.vix.scale)
        rationale = text if not dev.message else f"{text}. {dev.message}"
        return FactorDiagnostic(name, STATUS_OK, round(float(dev.value), 2), lookup_date, dev.source_date, thresholds, score, rationale)

    def score_cot_index(self, dfs: dict[str, pd.DataFrame], as_of: pd.Timestamp, *, use_publication_lag: bool = True) -> FactorDiagnostic:
        name = "COT Index Composite"
        df = dfs.get(f"{self.asset_key}_cot")
        if df is None or df.empty:
            return FactorDiagnostic(name, "MISSING", None, as_of, None, "", None, "COT dataset unavailable")

        eff = self._cot_lookup_date(as_of, use_publication_lag)
        hist = slice_until(df, eff)
        if hist.empty:
            return FactorDiagnostic(name, "MISSING", None, eff, None, "", None, "No COT rows up to lookup date")

        q = self.cfg.cot.index.quantiles
        qmap = {"p5": float(q.p5), "p10": float(q.p10), "p90": float(q.p90), "p95": float(q.p95)}
        t_comm = get_quantile_thresholds(
            hist["COT_Index_Comm"], lookback_points=int(self.cfg.cot.index.quantile_lookback_points), quantiles=qmap
        ) or {}
        t_large = get_quantile_thresholds(
            hist["COT_Index_Large_Inverted"], lookback_points=int(self.cfg.cot.index.quantile_lookback_points), quantiles=qmap
        ) or {}

        c_abs = self.cfg.cot.index.scoring.comm.abs_thresholds
        l_abs = self.cfg.cot.index.scoring.large_inv.abs_thresholds
        thresholds = (
            f"Comm abs[{c_abs.strong_bear:.0f},{c_abs.bear:.0f},{c_abs.bull:.0f},{c_abs.strong_bull:.0f}] "
            f"q[{t_comm.get('p5')},{t_comm.get('p10')},{t_comm.get('p90')},{t_comm.get('p95')}] | "
            f"LargeInv abs[{l_abs.strong_bear:.0f},{l_abs.bear:.0f},{l_abs.bull:.0f},{l_abs.strong_bull:.0f}] "
            f"q[{t_large.get('p5')},{t_large.get('p10')},{t_large.get('p90')},{t_large.get('p95')}]"
        )

        tol = pd.Timedelta(days=int(self.cfg.cot.asof_tolerance_days))
        row, source_date, row_status, row_message = self._latest_row_with_tolerance(hist, lookup_date=eff, tolerance=tol)
        if row_status != STATUS_OK:
            return FactorDiagnostic(name, row_status, None, eff, source_date, thresholds, None, row_message)

        comm_status = str(row.get("COT_Index_Comm_Status", STATUS_OK))
        large_status = str(row.get("COT_Index_Large_Inverted_Status", row.get("COT_Index_Large_Status", STATUS_OK)))
        value_text = (
            f"Comm={self._fmt_value(row.get('COT_Index_Comm'))}; "
            f"LargeInv={self._fmt_value(row.get('COT_Index_Large_Inverted'))}"
        )

        problems: list[str] = []
        final_status = STATUS_OK
        if comm_status != STATUS_OK:
            final_status = comm_status
            problems.append(self._status_reason(comm_status, field="COT_Index_Comm"))
        if large_status != STATUS_OK:
            final_status = large_status if final_status == STATUS_OK else final_status
            problems.append(self._status_reason(large_status, field="COT_Index_Large_Inverted"))

        if problems:
            return FactorDiagnostic(
                name,
                final_status,
                value_text,
                eff,
                source_date,
                thresholds,
                None,
                " | ".join(problems) + f". {row_message}",
            )

        try:
            comm_val = float(row["COT_Index_Comm"])
            large_val = float(row["COT_Index_Large_Inverted"])
        except Exception:
            return FactorDiagnostic(
                name,
                "INVALID",
                value_text,
                eff,
                source_date,
                thresholds,
                None,
                f"Failed to parse COT index values from prepared row. {row_message}",
            )

        score, text = calculate_cot_index_composite(
            comm_idx=comm_val,
            thresholds_comm=t_comm,
            large_inv_idx=large_val,
            thresholds_large=t_large,
            cfg=self.cfg.cot.index,
        )
        rationale = f"{text}. Быки используют максимум(абс., квант.); медведям нужно абс. + квант. подтверждение. {row_message}"
        return FactorDiagnostic(
            name,
            STATUS_OK,
            value_text,
            eff,
            source_date,
            thresholds,
            float(score) * float(self.cfg.cot.index.weight),
            rationale,
        )

    def score_cot_zscore(self, dfs: dict[str, pd.DataFrame], as_of: pd.Timestamp, *, use_publication_lag: bool = True) -> FactorDiagnostic:
        name = "COT Z-Score"
        df = dfs.get(f"{self.asset_key}_cot")
        if df is None or df.empty:
            return FactorDiagnostic(name, "MISSING", None, as_of, None, "", None, "COT dataset unavailable")

        eff = self._cot_lookup_date(as_of, use_publication_lag)
        hist = slice_until(df, eff)
        if hist.empty:
            return FactorDiagnostic(name, "MISSING", None, eff, None, "", None, "No COT rows up to lookup date")

        tol = pd.Timedelta(days=int(self.cfg.cot.asof_tolerance_days))
        row, source_date, row_status, row_message = self._latest_row_with_tolerance(hist, lookup_date=eff, tolerance=tol)

        thr = self.cfg.cot.z_score.thresholds
        thresholds = f"strong_bear={float(thr.strong_bear):.2f}; strong_bull={float(thr.strong_bull):.2f}"

        if row_status != STATUS_OK:
            return FactorDiagnostic(name, row_status, None, eff, source_date, thresholds, None, row_message)

        z_status = str(row.get("Z_Score_Comm_Status", STATUS_OK))
        z_value = row.get("Z_Score_Comm")
        shown_value = None
        try:
            if z_value is not None and not pd.isna(z_value):
                shown_value = round(float(z_value), 3)
        except Exception:
            shown_value = self._fmt_value(z_value)

        if z_status != STATUS_OK:
            return FactorDiagnostic(
                name,
                z_status,
                shown_value,
                eff,
                source_date,
                thresholds,
                None,
                f"{self._status_reason(z_status, field='Z_Score_Comm')}. {row_message}",
            )

        try:
            z_val = float(row["Z_Score_Comm"])
        except Exception:
            return FactorDiagnostic(
                name,
                "INVALID",
                shown_value,
                eff,
                source_date,
                thresholds,
                None,
                f"Failed to parse Z_Score_Comm from prepared row. {row_message}",
            )

        score, text = calculate_cot_zscore_score(z_comm=z_val, cfg=self.cfg.cot.z_score)
        return FactorDiagnostic(
            name,
            STATUS_OK,
            round(z_val, 3),
            eff,
            source_date,
            thresholds,
            float(score) * float(self.cfg.cot.z_score.weight),
            f"{text}. {row_message}",
        )

    def get_verdict(self, *, total: float | None, confidence: float) -> str:
        if total is None or confidence < float(self.cfg.min_confidence_for_verdict):
            return VERDICT_NO_DATA
        buy_thr = float(self.cfg.verdict_thresholds.verdict_buy)
        sell_thr = float(self.cfg.verdict_thresholds.verdict_sell)
        if total >= buy_thr:
            return VERDICT_BULLISH
        if total <= -sell_thr:
            return VERDICT_BEARISH
        return VERDICT_NEUTRAL

    def _interaction_boost_applies(self, vix_s: float | None, cot_s: float | None) -> bool:
        inter = self.cfg.interaction
        if not inter.enabled or vix_s is None or cot_s is None or pd.isna(vix_s) or pd.isna(cot_s):
            return False
        if abs(float(vix_s)) < float(inter.vix_strong) or abs(float(cot_s)) < float(inter.cot_strong):
            return False
        same = (float(vix_s) >= 0) == (float(cot_s) >= 0)
        return same if str(inter.mode).lower() == "same" else not same

    @staticmethod
    def _fmt_score(v: float | None) -> str:
        if v is None:
            return "No data"
        try:
            if pd.isna(v):
                return "No data"
        except Exception:
            pass
        return f"{float(v):+.2f}"