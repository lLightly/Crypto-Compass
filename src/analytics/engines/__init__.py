from __future__ import annotations

from typing import Dict, Type

from src.analytics.engines.base import BaseAssetEngine
from src.analytics.engines.btc_engine import BTCAssetEngine
from src.analytics.engines.eth_engine import ETHAssetEngine

_ENGINE_MAP: Dict[str, Type[BaseAssetEngine]] = {
    "BTC": BTCAssetEngine,
    "ETH": ETHAssetEngine,
}


def get_asset_engine(asset: str) -> BaseAssetEngine:
    a = str(asset).strip().upper()
    cls = _ENGINE_MAP.get(a)
    if cls is None:
        raise ValueError(f"Unknown asset: {asset!r}")
    return cls()