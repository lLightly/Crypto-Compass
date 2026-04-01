from __future__ import annotations

import logging

from src.services.updater import update_all_data
from src.utils.logging_config import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    update_all_data()
    logger.info("All data updated.")


if __name__ == "__main__":
    main()