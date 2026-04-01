from __future__ import annotations

from src.analytics.engines.base import BaseAssetEngine
from src.config.settings import get_settings


class ETHAssetEngine(BaseAssetEngine):
    def __init__(self):
        super().__init__("ETH", get_settings().assets.eth.engine)