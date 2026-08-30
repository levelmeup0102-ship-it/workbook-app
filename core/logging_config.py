import logging, sys

from core.settings import settings

def setup_logging() -> None:

    log_level = settings.LOG_LEVEL

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout
    )