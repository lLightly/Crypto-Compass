from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path("config.yaml")


def _as_date(v: Any, default: dt.date, *, field: str) -> dt.date:
    if v is None:
        return default
    if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
        return v
    try:
        return dt.date.fromisoformat(str(v))
    except Exception as e:
        raise ValueError(f"Invalid date for {field}: {v!r}") from e


def _num(v: Any, default: float, *, field: str) -> float:
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception as e:
        raise ValueError(f"Invalid number for {field}: {v!r}") from e


def _int(v: Any, default: int, *, field: str) -> int:
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"Invalid integer for {field}: {v!r}") from e


def _bool(v: Any, default: bool, *, field: str) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {field}: {v!r}")


def _sigma_levels(v: Any, *, field: str) -> list[int]:
    if not isinstance(v, list) or not v:
        raise ValueError(f"{field} must be a non-empty list of ints")
    out = sorted({int(x) for x in v})
    if any(x <= 0 for x in out):
        raise ValueError(f"{field} must contain positive integers")
    if len(out) > 3:
        raise ValueError(f"{field} supports at most 3 scoring levels")
    return out


@dataclass(frozen=True)
class UISettings:
    plot_padding_days: int
    default_years: int
    slider_step_days: int


@dataclass(frozen=True)
class CompassSettings:
    trend_horizon_months: int


@dataclass(frozen=True)
class BacktestSettings:
    fee_pct: float
    slippage_pct: float
    rf_rate: float
    periods_per_year: int
    trailing_stop_pct: float


@dataclass(frozen=True)
class VerdictThresholds:
    buy: float
    sell: float

    @property
    def verdict_buy(self) -> float:
        return float(self.buy)

    @property
    def verdict_sell(self) -> float:
        return float(self.sell)


@dataclass(frozen=True)
class VixScoringSettings:
    very_strong_risk_off_score: float
    strong_risk_off_score: float
    risk_off_score: float
    very_strong_risk_on_score: float
    strong_risk_on_score: float
    risk_on_score: float


@dataclass(frozen=True)
class VixEngineSettings:
    enabled: bool
    weight: float
    scale: float
    asof_tolerance_days: int
    rolling_window_days: int
    levels_lookback_points: int
    sigma_levels: list[int]
    scoring: VixScoringSettings


@dataclass(frozen=True)
class COTAbsThresholds:
    bull: float
    strong_bull: float
    bear: float
    strong_bear: float


@dataclass(frozen=True)
class COTComponentScores:
    bull: float
    strong_bull: float
    bear: float
    strong_bear: float


@dataclass(frozen=True)
class COTIndexComponentScoring:
    abs_thresholds: COTAbsThresholds
    scores: COTComponentScores


@dataclass(frozen=True)
class COTIndexScoringSettings:
    comm: COTIndexComponentScoring
    large_inv: COTIndexComponentScoring


@dataclass(frozen=True)
class COTIndexQuantiles:
    p5: float
    p10: float
    p90: float
    p95: float


@dataclass(frozen=True)
class COTIndexSettings:
    enabled: bool
    weight: float
    window_weeks: int
    quantile_lookback_points: int
    quantiles: COTIndexQuantiles
    scoring: COTIndexScoringSettings


@dataclass(frozen=True)
class COTZScoreThresholds:
    strong_bull: float
    strong_bear: float


@dataclass(frozen=True)
class COTZScoreScores:
    strong_bull: float
    strong_bear: float


@dataclass(frozen=True)
class COTZScoreSettings:
    enabled: bool
    weight: float
    window_weeks: int
    thresholds: COTZScoreThresholds
    scores: COTZScoreScores


@dataclass(frozen=True)
class COTEngineSettings:
    asof_tolerance_days: int
    publication_lag_days: int
    index: COTIndexSettings
    z_score: COTZScoreSettings


@dataclass(frozen=True)
class InteractionSettings:
    enabled: bool
    mode: str
    vix_strong: float
    cot_strong: float
    boost: float


@dataclass(frozen=True)
class AssetEngineSettings:
    verdict_thresholds: VerdictThresholds
    min_confidence_for_verdict: float
    vix: VixEngineSettings
    cot: COTEngineSettings
    interaction: InteractionSettings


@dataclass(frozen=True)
class SingleAssetSettings:
    price_start: str
    cot_min_date: dt.date
    engine: AssetEngineSettings


@dataclass(frozen=True)
class AssetsSettings:
    btc: SingleAssetSettings
    eth: SingleAssetSettings
    macro_min_date: dt.date
    conclusion_min_date: dt.date

    @property
    def btc_price_start(self) -> str:
        return str(self.btc.price_start)

    @property
    def eth_price_start(self) -> str:
        return str(self.eth.price_start)

    @property
    def btc_cot_min_date(self) -> dt.date:
        return self.btc.cot_min_date

    @property
    def eth_cot_min_date(self) -> dt.date:
        return self.eth.cot_min_date

    def for_asset(self, asset: str) -> SingleAssetSettings:
        key = str(asset).strip().lower()
        if key == "btc":
            return self.btc
        if key == "eth":
            return self.eth
        raise ValueError(f"Unknown asset: {asset!r}")


@dataclass(frozen=True)
class COTDefaultsSettings:
    weeks_in_year: int
    default_years: int

    @property
    def default_weeks(self) -> int:
        return int(self.weeks_in_year * self.default_years)


@dataclass(frozen=False)
class Settings:
    raw: dict[str, Any]
    data_dir: str
    files: dict[str, str]
    ui: UISettings
    assets: AssetsSettings
    cot: COTDefaultsSettings
    compass: CompassSettings
    backtest: BacktestSettings

    def engine_for(self, asset: str) -> AssetEngineSettings:
        return self.assets.for_asset(asset).engine


_SETTINGS: Settings | None = None


def _parse_component_scoring(x: dict[str, Any], *, base: str) -> COTIndexComponentScoring:
    thr = x.get("abs_thresholds") or {}
    sc = x.get("scores") or {}
    return COTIndexComponentScoring(
        abs_thresholds=COTAbsThresholds(
            bull=_num(thr.get("bull"), 80.0, field=f"{base}.abs_thresholds.bull"),
            strong_bull=_num(thr.get("strong_bull"), 90.0, field=f"{base}.abs_thresholds.strong_bull"),
            bear=_num(thr.get("bear"), 20.0, field=f"{base}.abs_thresholds.bear"),
            strong_bear=_num(thr.get("strong_bear"), 10.0, field=f"{base}.abs_thresholds.strong_bear"),
        ),
        scores=COTComponentScores(
            bull=_num(sc.get("bull"), 1.0, field=f"{base}.scores.bull"),
            strong_bull=_num(sc.get("strong_bull"), 2.0, field=f"{base}.scores.strong_bull"),
            bear=_num(sc.get("bear"), -1.0, field=f"{base}.scores.bear"),
            strong_bear=_num(sc.get("strong_bear"), -2.0, field=f"{base}.scores.strong_bear"),
        ),
    )


def _parse_engine(asset_raw: dict[str, Any], *, asset_name: str) -> AssetEngineSettings:
    eng = asset_raw.get("engine") or {}
    vt = eng.get("verdict_thresholds") or {}
    vix_raw = eng.get("vix") or {}
    vix_sc = vix_raw.get("scoring") or {}
    cot_raw = eng.get("cot") or {}
    idx_raw = cot_raw.get("index") or {}
    idx_sc = idx_raw.get("scoring") or {}
    z_raw = cot_raw.get("z_score") or {}
    z_thr = z_raw.get("thresholds") or {}
    z_sc = z_raw.get("scores") or {}
    inter_raw = eng.get("interaction") or {}
    q_raw = idx_raw.get("quantiles") or {}

    buy = _num(vt.get("buy"), 1.5, field=f"assets.{asset_name}.engine.verdict_thresholds.buy")
    sell = _num(vt.get("sell"), buy, field=f"assets.{asset_name}.engine.verdict_thresholds.sell")

    return AssetEngineSettings(
        verdict_thresholds=VerdictThresholds(buy=buy, sell=sell),
        min_confidence_for_verdict=_num(
            eng.get("min_confidence_for_verdict"),
            1.0,
            field=f"assets.{asset_name}.engine.min_confidence_for_verdict",
        ),
        vix=VixEngineSettings(
            enabled=_bool(vix_raw.get("enabled"), True, field=f"assets.{asset_name}.engine.vix.enabled"),
            weight=_num(vix_raw.get("weight"), 1.0, field=f"assets.{asset_name}.engine.vix.weight"),
            scale=_num(vix_raw.get("scale"), 1.0, field=f"assets.{asset_name}.engine.vix.scale"),
            asof_tolerance_days=_int(
                vix_raw.get("asof_tolerance_days"), 7, field=f"assets.{asset_name}.engine.vix.asof_tolerance_days"
            ),
            rolling_window_days=_int(
                vix_raw.get("rolling_window_days"), 252, field=f"assets.{asset_name}.engine.vix.rolling_window_days"
            ),
            levels_lookback_points=_int(
                vix_raw.get("levels_lookback_points"),
                756,
                field=f"assets.{asset_name}.engine.vix.levels_lookback_points",
            ),
            sigma_levels=_sigma_levels(vix_raw.get("sigma_levels"), field=f"assets.{asset_name}.engine.vix.sigma_levels"),
            scoring=VixScoringSettings(
                very_strong_risk_off_score=_num(
                    vix_sc.get("very_strong_risk_off_score"),
                    -8.0,
                    field=f"assets.{asset_name}.engine.vix.scoring.very_strong_risk_off_score",
                ),
                strong_risk_off_score=_num(
                    vix_sc.get("strong_risk_off_score"),
                    -3.0,
                    field=f"assets.{asset_name}.engine.vix.scoring.strong_risk_off_score",
                ),
                risk_off_score=_num(
                    vix_sc.get("risk_off_score"),
                    -1.0,
                    field=f"assets.{asset_name}.engine.vix.scoring.risk_off_score",
                ),
                very_strong_risk_on_score=_num(
                    vix_sc.get("very_strong_risk_on_score"),
                    8.0,
                    field=f"assets.{asset_name}.engine.vix.scoring.very_strong_risk_on_score",
                ),
                strong_risk_on_score=_num(
                    vix_sc.get("strong_risk_on_score"),
                    3.0,
                    field=f"assets.{asset_name}.engine.vix.scoring.strong_risk_on_score",
                ),
                risk_on_score=_num(
                    vix_sc.get("risk_on_score"),
                    1.0,
                    field=f"assets.{asset_name}.engine.vix.scoring.risk_on_score",
                ),
            ),
        ),
        cot=COTEngineSettings(
            asof_tolerance_days=_int(
                cot_raw.get("asof_tolerance_days"), 14, field=f"assets.{asset_name}.engine.cot.asof_tolerance_days"
            ),
            publication_lag_days=_int(
                cot_raw.get("publication_lag_days"), 3, field=f"assets.{asset_name}.engine.cot.publication_lag_days"
            ),
            index=COTIndexSettings(
                enabled=_bool(idx_raw.get("enabled"), True, field=f"assets.{asset_name}.engine.cot.index.enabled"),
                weight=_num(idx_raw.get("weight"), 1.0, field=f"assets.{asset_name}.engine.cot.index.weight"),
                window_weeks=_int(
                    idx_raw.get("window_weeks"), 26, field=f"assets.{asset_name}.engine.cot.index.window_weeks"
                ),
                quantile_lookback_points=_int(
                    idx_raw.get("quantile_lookback_points"),
                    104,
                    field=f"assets.{asset_name}.engine.cot.index.quantile_lookback_points",
                ),
                quantiles=COTIndexQuantiles(
                    p5=_num(q_raw.get("p5"), 0.05, field=f"assets.{asset_name}.engine.cot.index.quantiles.p5"),
                    p10=_num(q_raw.get("p10"), 0.10, field=f"assets.{asset_name}.engine.cot.index.quantiles.p10"),
                    p90=_num(q_raw.get("p90"), 0.90, field=f"assets.{asset_name}.engine.cot.index.quantiles.p90"),
                    p95=_num(q_raw.get("p95"), 0.95, field=f"assets.{asset_name}.engine.cot.index.quantiles.p95"),
                ),
                scoring=COTIndexScoringSettings(
                    comm=_parse_component_scoring(idx_sc.get("comm") or {}, base=f"assets.{asset_name}.engine.cot.index.scoring.comm"),
                    large_inv=_parse_component_scoring(
                        idx_sc.get("large_inv") or {}, base=f"assets.{asset_name}.engine.cot.index.scoring.large_inv"
                    ),
                ),
            ),
            z_score=COTZScoreSettings(
                enabled=_bool(z_raw.get("enabled"), True, field=f"assets.{asset_name}.engine.cot.z_score.enabled"),
                weight=_num(z_raw.get("weight"), 1.0, field=f"assets.{asset_name}.engine.cot.z_score.weight"),
                window_weeks=_int(
                    z_raw.get("window_weeks"), 104, field=f"assets.{asset_name}.engine.cot.z_score.window_weeks"
                ),
                thresholds=COTZScoreThresholds(
                    strong_bull=_num(
                        z_thr.get("strong_bull"), 3.0, field=f"assets.{asset_name}.engine.cot.z_score.thresholds.strong_bull"
                    ),
                    strong_bear=_num(
                        z_thr.get("strong_bear"), -3.0, field=f"assets.{asset_name}.engine.cot.z_score.thresholds.strong_bear"
                    ),
                ),
                scores=COTZScoreScores(
                    strong_bull=_num(
                        z_sc.get("strong_bull"), 1.5, field=f"assets.{asset_name}.engine.cot.z_score.scores.strong_bull"
                    ),
                    strong_bear=_num(
                        z_sc.get("strong_bear"), -1.5, field=f"assets.{asset_name}.engine.cot.z_score.scores.strong_bear"
                    ),
                ),
            ),
        ),
        interaction=InteractionSettings(
            enabled=_bool(inter_raw.get("enabled"), False, field=f"assets.{asset_name}.engine.interaction.enabled"),
            mode=str(inter_raw.get("mode", "same")).strip().lower(),
            vix_strong=_num(inter_raw.get("vix_strong"), 1.0, field=f"assets.{asset_name}.engine.interaction.vix_strong"),
            cot_strong=_num(inter_raw.get("cot_strong"), 1.0, field=f"assets.{asset_name}.engine.interaction.cot_strong"),
            boost=_num(inter_raw.get("boost"), 1.0, field=f"assets.{asset_name}.engine.interaction.boost"),
        ),
    )


def _build_settings(raw: dict[str, Any]) -> Settings:
    ui_raw = raw.get("ui") or {}
    assets_raw = raw.get("assets") or {}
    cot_raw = raw.get("cot") or {}
    compass_raw = raw.get("compass") or {}
    backtest_raw = raw.get("backtest") or {}

    return Settings(
        raw=raw,
        data_dir=str(raw.get("data_dir", "data/processed")),
        files=dict(raw.get("files") or {}),
        ui=UISettings(
            plot_padding_days=_int(ui_raw.get("plot_padding_days"), 7, field="ui.plot_padding_days"),
            default_years=_int(ui_raw.get("default_years"), 3, field="ui.default_years"),
            slider_step_days=_int(ui_raw.get("slider_step_days"), 7, field="ui.slider_step_days"),
        ),
        assets=AssetsSettings(
            btc=SingleAssetSettings(
                price_start=str((assets_raw.get("btc") or {}).get("price_start", "2020-05-12")),
                cot_min_date=_as_date(
                    (assets_raw.get("btc") or {}).get("cot_min_date"),
                    dt.date(2020, 5, 12),
                    field="assets.btc.cot_min_date",
                ),
                engine=_parse_engine(assets_raw.get("btc") or {}, asset_name="btc"),
            ),
            eth=SingleAssetSettings(
                price_start=str((assets_raw.get("eth") or {}).get("price_start", "2023-03-28")),
                cot_min_date=_as_date(
                    (assets_raw.get("eth") or {}).get("cot_min_date"),
                    dt.date(2023, 3, 28),
                    field="assets.eth.cot_min_date",
                ),
                engine=_parse_engine(assets_raw.get("eth") or {}, asset_name="eth"),
            ),
            macro_min_date=_as_date(assets_raw.get("macro_min_date"), dt.date(2020, 5, 12), field="assets.macro_min_date"),
            conclusion_min_date=_as_date(
                assets_raw.get("conclusion_min_date"), dt.date(2020, 5, 12), field="assets.conclusion_min_date"
            ),
        ),
        cot=COTDefaultsSettings(
            weeks_in_year=_int(cot_raw.get("weeks_in_year"), 52, field="cot.weeks_in_year"),
            default_years=_int(cot_raw.get("default_years"), 3, field="cot.default_years"),
        ),
        compass=CompassSettings(
            trend_horizon_months=_int(compass_raw.get("trend_horizon_months"), 3, field="compass.trend_horizon_months")
        ),
        backtest=BacktestSettings(
            fee_pct=_num(backtest_raw.get("fee_pct"), 0.001, field="backtest.fee_pct"),
            slippage_pct=_num(backtest_raw.get("slippage_pct"), 0.0005, field="backtest.slippage_pct"),
            rf_rate=_num(backtest_raw.get("rf_rate"), 0.05, field="backtest.rf_rate"),
            periods_per_year=_int(backtest_raw.get("periods_per_year"), 365, field="backtest.periods_per_year"),
            trailing_stop_pct=_num(backtest_raw.get("trailing_stop_pct"), 0.0, field="backtest.trailing_stop_pct"),
        ),
    )


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    _SETTINGS = _build_settings(raw)
    return _SETTINGS


def reset_settings_cache() -> None:
    global _SETTINGS
    _SETTINGS = None


def set_settings_from_raw(raw: dict[str, Any]) -> Settings:
    global _SETTINGS
    _SETTINGS = _build_settings(raw)
    return _SETTINGS


def reload_settings(path: Path = CONFIG_PATH) -> Settings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return set_settings_from_raw(raw)