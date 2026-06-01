from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.settings import get_settings
from src.data_fetchers import finance_api
from src.data_fetchers.cot_parser import fetch_cot_raw, preprocess
from src.utils.helpers import save_csv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpdateFailure:
    dataset: str
    stage: str
    error: str

    def as_row(self) -> dict[str, str]:
        return {
            "Dataset": self.dataset,
            "Stage": self.stage,
            "Error": self.error,
        }


@dataclass(frozen=True)
class UpdateResult:
    updated: tuple[str, ...]
    failures: tuple[UpdateFailure, ...]

    @property
    def ok(self) -> bool:
        return not self.failures

    def error_summary(self) -> str:
        return " | ".join(f"{f.dataset}/{f.stage}: {f.error}" for f in self.failures)


def _save_processed_dataset(name: str, df: pd.DataFrame, proc_dir: Path) -> None:
    s = get_settings()
    rel_path = s.files.get(name)
    if not rel_path:
        raise ValueError(f"Dataset is not configured: {name}")
    save_csv(df, str(proc_dir / rel_path))


def _save_price_datasets(price_data: dict[str, pd.DataFrame], raw_dir: Path, proc_dir: Path) -> list[str]:
    updated: list[str] = []

    for name, df in price_data.items():
        if df is None or df.empty:
            raise RuntimeError(f"Price dataset is empty: {name}")

        if name == "vix":
            save_csv(df, str(raw_dir / "vix.csv"))

        _save_processed_dataset(name, df, proc_dir)
        updated.append(name)

    return updated


def _update_cot(asset: str, raw_dir: Path, proc_dir: Path) -> list[str]:
    cot_raw = fetch_cot_raw(asset)
    if cot_raw is None or cot_raw.empty:
        raise RuntimeError(f"COT raw dataset is empty for {asset}")

    asset_key = asset.lower()
    save_csv(cot_raw, str(raw_dir / f"{asset_key}_cot_raw.csv"))

    cot = preprocess(cot_raw)
    if cot.empty:
        raise RuntimeError(f"COT preprocessing produced an empty dataframe for {asset}")

    _save_processed_dataset(f"{asset_key}_cot", cot.sort_values("date"), proc_dir)
    return [f"{asset_key}_cot"]


def update_all_data() -> UpdateResult:
    raw_dir = Path("data/raw")
    proc_dir = Path(get_settings().data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    updated: list[str] = []
    failures: list[UpdateFailure] = []

    try:
        price_data = finance_api.fetch_all_prices()
        updated.extend(_save_price_datasets(price_data, raw_dir, proc_dir))
    except Exception as exc:
        logger.exception("Failed to update Yahoo price datasets: %s", exc)
        failures.append(UpdateFailure("yahoo_prices", "download", str(exc)))

    for asset in ("BTC", "ETH"):
        try:
            updated.extend(_update_cot(asset, raw_dir, proc_dir))
        except Exception as exc:
            logger.exception("Failed to update COT dataset %s: %s", asset, exc)
            failures.append(UpdateFailure(f"{asset.lower()}_cot", "download", str(exc)))

    return UpdateResult(updated=tuple(updated), failures=tuple(failures))