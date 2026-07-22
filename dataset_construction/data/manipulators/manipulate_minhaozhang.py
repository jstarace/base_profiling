"""minhaozhang/mbti (Reddit) manipulator -> common schema (consolidates comments by author, chunked)."""
import csv
import io
import paths  # noqa: F401
import pandas as pd

from dataset_construction.data import schema, chunking
from dataset_construction.data.manipulators.base import Manipulator

_DERIVED = ["classification", "Extravert", "Introvert", "Intuition", "Sensing",
            "Feeling", "Thinking", "Judging", "Perceiving", "derived_PTYPE"]


def _parse_comments(body):
    """Split a minhaozhang body into clean comments. Bodies are CSV-quoted (fields quoted
    with " and separated by ;), so a real CSV parse handles the ""-escaping and any embedded
    ; correctly — naive split('";"') disagrees ~21% of the time. Falls back to the whole
    body as one comment if parsing ever fails."""
    try:
        rows = list(csv.reader(io.StringIO(body), delimiter=";", quotechar='"'))
    except Exception:
        return [body]
    return [f for r in rows for f in r if f.strip()]


class Minhaozhang(Manipulator):
    DATASET = "minhaozhang/mbti"
    CAP = chunking.DEFAULT_CAP

    def manipulate(self, raw):
        records, skipped = [], 0
        for author, grp in raw.groupby("author", sort=False):
            mbti = str(grp["mbti"].iloc[0]).strip().upper()
            try:
                axes = schema.code_to_axes(mbti)
            except ValueError:
                skipped += 1
                continue
            derived = schema.derive_mbti_columns(axes["O"], axes["C"], axes["E"], axes["A"])
            base = {
                "O": axes["O"], "C": axes["C"], "E": axes["E"], "A": axes["A"],
                "ptype": derived["derived_PTYPE"],
                **{k: derived[k] for k in _DERIVED},
                "MATCH": derived["classification"] == mbti,
                "author_id": schema.hash_author(author),
            }
            comments = [c for body in grp["body"].astype(str) for c in _parse_comments(body)]
            chunks = chunking.chunk_units(comments, cap=self.CAP, join="|||")
            for text, n, tot in chunking.with_counts(chunks):
                records.append({"text": text, "#": n, "Tot": tot, **base})

        if skipped:
            print(f"[manipulate] {self.DATASET}: skipped {skipped} authors with a malformed mbti code")
        return self.finalize(pd.DataFrame(records))
