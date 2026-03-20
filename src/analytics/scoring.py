from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from src.config.settings import VixScoringSettings


def get_vix_scoring_levels(levels: Dict[str, float], sigma_levels: Iterable[int]) -> list[dict[str, Any]]:
    sigmas = sorted({int(x) for x in sigma_levels})
    if not sigmas:
        return []

    tiers: list[tuple[int, str]]
    if len(sigmas) == 1:
        tiers = [(sigmas[0], "risk")]
    elif len(sigmas) == 2:
        tiers = [(sigmas[0], "risk"), (sigmas[1], "strong")]
    else:
        tiers = [(sigmas[0], "risk"), (sigmas[1], "strong"), (sigmas[2], "very_strong")]

    out: list[dict[str, Any]] = []
    for sigma, tier in tiers:
        for sign in ("+", "-"):
            key = f"{sign}{sigma}σ"
            if key in levels:
                out.append({"key": key, "sigma": sigma, "tier": tier, "value": float(levels[key])})
    return out


def vix_score(
    dev_pct: float,
    levels: Dict[str, float],
    scores: VixScoringSettings,
    sigma_levels: Iterable[int],
) -> Tuple[float, str]:
    active = get_vix_scoring_levels(levels, sigma_levels)

    pos = sorted((x for x in active if x["key"].startswith("+")), key=lambda x: x["sigma"], reverse=True)
    neg = sorted((x for x in active if x["key"].startswith("-")), key=lambda x: x["sigma"], reverse=True)

    for item in pos:
        if dev_pct >= item["value"]:
            if item["tier"] == "very_strong":
                return float(scores.very_strong_risk_on_score), f"VIX ≥ {item['key']} → Экстремальный страх"
            if item["tier"] == "strong":
                return float(scores.strong_risk_on_score), f"VIX ≥ {item['key']} → Сильный страх"
            return float(scores.risk_on_score), f"VIX ≥ {item['key']} → Страх"

    for item in neg:
        if dev_pct <= item["value"]:
            if item["tier"] == "very_strong":
                return float(scores.very_strong_risk_off_score), f"VIX ≤ {item['key']} → Эйфория"
            if item["tier"] == "strong":
                return float(scores.strong_risk_off_score), f"VIX ≤ {item['key']} → Сильный позитив"
            return float(scores.risk_off_score), f"VIX ≤ {item['key']} → Позитив"

    return 0.0, "VIX нейтральный"