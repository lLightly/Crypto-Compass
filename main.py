from __future__ import annotations

import logging

from src.services.updater import UpdateResult, update_all_data
from src.utils.logging_config import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> UpdateResult:
    result = update_all_data()
    if result.ok:
        logger.info("All data updated: %s", ", ".join(result.updated))
    else:
        logger.error("Data update completed with errors: %s", result.error_summary())
    return result


if __name__ == "__main__":
    update_result = main()
    if not update_result.ok:
        raise SystemExit(1)