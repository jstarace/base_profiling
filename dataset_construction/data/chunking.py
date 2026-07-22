"""Pack text units into <=cap-char chunks without splitting a sentence, labeling each # of Tot."""
from syntok import segmenter

DEFAULT_CAP = 11_000  # chars; just above current mbti_profiles' largest row (10,090)


def chunk_units(units, cap=DEFAULT_CAP, join="\n"):
    """Pack `units` (list[str], each sentence-safe) into <=cap-char chunks, joined by
    `join`. Returns list[str] chunk texts in order. Never splits a sentence; a lone unit
    or sentence longer than `cap` is emitted whole (oversized) rather than broken."""
    pieces = []
    for u in units:
        if len(u) <= cap:
            pieces.append(u)
        else:
            pieces.extend(_sentences(u))  # only the oversized unit is segmented

    chunks, cur, cur_len = [], [], 0
    for p in pieces:
        extra = len(p) + (len(join) if cur else 0)
        if cur and cur_len + extra > cap:
            chunks.append(join.join(cur))
            cur, cur_len = [p], len(p)
        else:
            cur.append(p)
            cur_len += extra
    if cur:
        chunks.append(join.join(cur))
    return chunks


def with_counts(chunks):
    """[(text, #, Tot), ...] from a list of chunk texts — the record-keeping columns."""
    tot = len(chunks)
    return [(text, i + 1, tot) for i, text in enumerate(chunks)]


def _sentences(text):
    """Sentence strings of `text` via syntok, each sliced from its own offsets so the
    original spacing is preserved. Used only to split an over-cap unit on real boundaries."""
    out = []
    for paragraph in segmenter.analyze(text):
        for sentence in paragraph:
            toks = list(sentence)
            if toks:
                out.append(text[toks[0].offset: toks[-1].offset + len(toks[-1].value)])
    return out
