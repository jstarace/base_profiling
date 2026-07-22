import re
import paths  # noqa: F401
from dataset_construction.cleaning.types import SentenceRecord


def majority(n):
    """Votes needed for a majority of n judges (holds as judges are added)."""
    return n // 2 + 1


def standardize(canonical, verdicts, needed):
    """Tally the panel's index-votes onto the shared canonical sentences.

    Each judge flagged canonical indices, each carrying a short validation quote. A flag
    whose quote is not found in its claimed canonical sentence is a known-wrong index — a
    silent mis-count — so it is DROPPED from the tally (never trusted) and recorded in
    `mismatches` for human review.

    Returns (records, mismatches):
      - records: one SentenceRecord per flagged canonical index (validated votes only),
        sorted by index. vote_count is judges-per-index, anonymized — no judge identity,
        to avoid biasing the final judge. Sub-majority indices are gate-assigned
        action 'no_action'; majority indices are left action=None for the final judge.
      - mismatches: the dropped flags, so we can see how often judges mis-index.
    """
    by_index = {seg.index: seg for seg in canonical}
    tally = {}          # idx -> {"votes", "justifications", "components", "confidences"}
    mismatches = []

    for verdict in verdicts:
        # best validated flag per index for THIS judge (one vote per judge per index)
        best = {}
        for flag in (verdict.flagged or []):
            seg = by_index.get(flag.sentence_index)
            if seg is None or not _quote_matches(flag.quote, seg.text):
                mismatches.append({
                    "sentence_index": flag.sentence_index,
                    "quote": flag.quote,
                    "reason": "index out of range" if seg is None else "quote not in canonical sentence",
                })
                continue
            prev = best.get(flag.sentence_index)
            if prev is None or flag.confidence > prev.confidence:
                best[flag.sentence_index] = flag

        for idx, flag in best.items():
            entry = tally.setdefault(idx, {"votes": 0, "justifications": [], "components": [], "confidences": []})
            entry["votes"] += 1
            entry["justifications"].append(flag.justification)
            entry["components"].append(flag.component)
            entry["confidences"].append(flag.confidence)

    records = []
    for idx in sorted(tally):
        entry = tally[idx]
        seg = by_index[idx]
        votes = entry["votes"]
        records.append(SentenceRecord(
            sentence_index=idx,
            sentence=seg.text,
            start=seg.start,
            end=seg.end,
            vote_count=votes,
            justifications=entry["justifications"],
            components=entry["components"],
            max_confidence=max(entry["confidences"]) if entry["confidences"] else None,
            action="no_action" if votes < needed else None,
        ))
    return records, mismatches


def _quote_matches(quote, sentence):
    """True if the judge's validation quote really comes from this canonical sentence.
    Whitespace-normalized substring — forgiving of spacing, strict on wording, so a wrong
    index is still caught."""
    if not quote:
        return False
    q = re.sub(r"\s+", " ", quote).strip()
    s = re.sub(r"\s+", " ", sentence).strip()
    return bool(q) and q in s
