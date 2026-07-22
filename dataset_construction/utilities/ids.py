import paths  # noqa: F401  — loads .env / project root, consistent with other utilities
import uuid
from datetime import datetime, timezone


def generate_id() -> str:
    """Globally-unique, time-sortable row id: '<utc-timestamp>_<uuid4hex>'.

    The timestamp (compact ISO-8601 UTC, microsecond precision) makes ids sort
    by import time; the uuid4 guarantees uniqueness even for rows stamped in the
    same microsecond.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{ts}_{uuid.uuid4().hex}"


def add_row_ids(df, column: str = "id"):
    """Add a column of fresh unique ids, one per row. Mutates and returns df."""
    df[column] = [generate_id() for _ in range(len(df))]
    return df