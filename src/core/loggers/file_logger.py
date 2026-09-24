"""Structured JSON logger.

Emits one JSON object per line, each carrying a timestamp, level, service name
and the message. Structured lines are ready for a collector later without
committing to one now, and the `Logger` interface is unchanged, so call sites
keep passing a plain message string.

No correlation id: one process runs one session, so every line in a run belongs
to the same one and a field repeating that would say nothing.

Unlike a server's logger, this one writes to the file only and never to stdout:
the terminal belongs to the Rich renderer, and a log line printed into it would
land in the middle of a streaming answer.
"""

import json
import logging
from datetime import datetime, timezone

from src.core.config import settings
from src.core.loggers.logger import Logger

_SERVICE_NAME = "local-agent"


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "service": _SERVICE_NAME,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class FileLogger(Logger):
    def __init__(self, name: str, log_file: str) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)


file_logger = FileLogger(__name__, str(settings.log_path()))
