from __future__ import annotations

from src.analytics.engines.base import BaseAssetEngine
from src.config.settings import get_settings


class BTCAssetEngine(BaseAssetEngine):
    def __init__(self):
        super().__init__("BTC", get_settings().assets.btc.engine)