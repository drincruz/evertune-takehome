import json
import logging
import os

_EXTRA_FIELDS = ("status_code", "finish_reason", "attempt_number")

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)

def configure_logging(log_dir: str = "tmp", filename: str = "load_test.log", level: int | None = None) -> None:
    root_logger = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        return

    resolved_level = level
    if resolved_level is None:
        resolved_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, filename), mode="w")
    handler.setFormatter(JsonFormatter())

    root_logger.addHandler(handler)
    root_logger.setLevel(resolved_level)
