from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from src.config.settings import COTIndexComponentScoring, COTIndexSettings, COTZScoreSettings


def _cot_abs_score(idx: float, *, label: str, component: COTIndexComponentScoring) -> Tuple[float, str]:
    thr = component.abs_thresholds
    sc = component.scores

    if idx >= float(thr.strong_bull):
        return float(sc.strong_bull), f"{label} ≥{float(thr.strong_bull):.0f} (абс.) → Сильный бычий"
    if idx >= float(thr.bull):
        return float(sc.bull), f"{label} ≥{float(thr.bull):.0f} (абс.) → Бычий"
    if idx <= 0 or idx <= float(thr.strong_bear):
        return float(sc.strong_bear), f"{label} ≤{float(thr.strong_bear):.0f}/≤0 (abs) → Сильный медвежий"
    if idx <= float(thr.bear):
        return float(sc.bear), f"{label} ≤{float(thr.bear):.0f} (абс.) → Медвежий"
    return 0.0, f"{label} нейтральный (абс.)"


def _cot_quant_score(
    idx: float,
    thresholds: Optional[Dict[str, float]],
    *,
    label: str,
    component: COTIndexComponentScoring,
) -> Tuple[float, str]:
    sc = component.scores
    if not thresholds:
        return 0.0, f"{label}: квантильные пороги недоступны"

    if idx >= thresholds["p95"]:
        return float(sc.strong_bull), f"{label} ≥95p → Сильный бычий"
    if idx >= thresholds["p90"]:
        return float(sc.bull), f"{label} ≥90p → Бычий"
    if idx <= thresholds["p5"] or idx <= 0:
        return float(sc.strong_bear), f"{label} ≤5p/≤0 → Сильный медвежий"
    if idx <= thresholds["p10"]:
        return float(sc.bear), f"{label} ≤10p → Медвежий"
    return 0.0, f"{label} нейтральный"


def _pick_stronger(a: Tuple[float, str], b: Tuple[float, str]) -> Tuple[float, str]:
    sa, sb = float(a[0]), float(b[0])
    if abs(sa) > abs(sb):
        return a
    if abs(sb) > abs(sa):
        return b
    if "p" in b[1] or "≥" in b[1] or "≤" in b[1]:
        return b
    return a


def _directional_cot_score(
    abs_pair: Tuple[float, str],
    quant_pair: Tuple[float, str],
    *,
    label: str,
) -> Tuple[float, str]:
    abs_score, quant_score = float(abs_pair[0]), float(quant_pair[0])

    if abs_score > 0 or quant_score > 0:
        bulls = [x for x in (abs_pair, quant_pair) if float(x[0]) > 0]
        if len(bulls) == 2:
            return _pick_stronger(bulls[0], bulls[1])
        return bulls[0]

    if abs_score < 0 and quant_score < 0:
        stronger = _pick_stronger(abs_pair, quant_pair)
        other = quant_pair if stronger == abs_pair else abs_pair
        return float(stronger[0]), f"{stronger[1]} | подтверждение: {other[1]}"

    if abs_score < 0 or quant_score < 0:
        return 0.0, f"{label} медвежий отклонён: нужно абс. + квантильное подтверждение"

    return 0.0, f"{label} нейтальный"


def calculate_cot_index_composite(
    *,
    comm_idx: float,
    thresholds_comm: Optional[Dict[str, float]],
    large_inv_idx: Optional[float],
    thresholds_large: Optional[Dict[str, float]],
    cfg: COTIndexSettings,
) -> Tuple[float, str]:
    score = 0.0
    parts: list[str] = []

    comm_abs = _cot_abs_score(float(comm_idx), label="Comm", component=cfg.scoring.comm)
    comm_quant = _cot_quant_score(float(comm_idx), thresholds_comm, label="Comm", component=cfg.scoring.comm)
    comm_s, comm_txt = _directional_cot_score(comm_abs, comm_quant, label="Comm")
    score += float(comm_s)
    parts.append(comm_txt)

    if large_inv_idx is None or pd.isna(large_inv_idx):
        parts.append("LargeInv: нет данных")
    else:
        li = float(large_inv_idx)
        li_abs = _cot_abs_score(li, label="LargeInv", component=cfg.scoring.large_inv)
        li_quant = _cot_quant_score(li, thresholds_large, label="LargeInv", component=cfg.scoring.large_inv)
        li_s, li_txt = _directional_cot_score(li_abs, li_quant, label="LargeInv")
        score += float(li_s)
        parts.append(li_txt)

    return round(score, 2), (" | ".join(parts) if parts else "COT индекс нейтральный")


def calculate_cot_zscore_score(*, z_comm: float, cfg: COTZScoreSettings) -> Tuple[float, str]:
    z = float(z_comm)
    thr = cfg.thresholds
    sc = cfg.scores

    if z >= float(thr.strong_bull):
        return float(sc.strong_bull), f"Comm Z-score ≥{float(thr.strong_bull):.2f} → Сильное бычье усиление"
    if z <= float(thr.strong_bear):
        return float(sc.strong_bear), f"Comm Z-score ≤{float(thr.strong_bear):.2f} → Сильный медвежий штраф"
    return 0.0, "Comm Z-score нейтрально"