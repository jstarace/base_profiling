import paths
import json
import threading
from dataclasses import asdict, is_dataclass

LOG_PATH = paths.ROOT / "temp_explore" / "logs" / "cleaning_log.jsonl"
_LOG_LOCK = threading.Lock()


def log_record(record, path=LOG_PATH):
    """Append one record (a CleaningRecord dataclass or a dict) to the JSONL log. Thread-safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record) if is_dataclass(record) else record
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")