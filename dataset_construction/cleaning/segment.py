import re

import paths  # noqa: F401
from syntok import segmenter
from dataset_construction.cleaning.types import Segment

# Runs of 2+ dots are STYLISTIC in this corpus, not sentence-terminal. syntok otherwise splits on
# them, shattering one real sentence into fragments and corrupting the canonical coordinate system.
_ELLIPSIS = re.compile(r"\.{2,}")
_MASK = chr(0xF8FF)  # same-length non-terminal placeholder (private-use char)

# Hard, dataset-specific delimiters, split BEFORE sentence tokenization and keyed by the
# row's `dataset` string. The delimiter joins independent posts, so no sentence spans it;
# splitting first keeps syntok from merging two posts into one sentence. Researchers: add
# your source's delimiter here. Unknown sources (None) are tokenized whole.
SOURCE_DELIMITERS = {
    "jingjietan/kaggle-mbti": "|||",
    "minhaozhang/mbti": "|||",  # manipulator joins CSV-parsed comments with |||
}


def segment(text, dataset=None):
    """Split `text` into the canonical Segment array — the shared coordinate system every
    judge and the final judge index into.

    Two passes: (1) split on the dataset's hard delimiter (pluggable per source),
    (2) sentence-tokenize each chunk with syntok. Offsets are absolute in `text`, and each
    Segment's text is sliced straight from those offsets, so text[seg.start:seg.end] ==
    seg.text always holds — the invariant the transformer relies on to edit by span.
    """
    segments = []
    idx = 0
    for chunk, base in _split_on_delimiter(text, SOURCE_DELIMITERS.get(dataset)):
        for start, end in _sentence_spans(chunk):
            seg_text = chunk[start:end]
            if not seg_text.strip():
                continue
            segments.append(Segment(index=idx, text=seg_text, start=base + start, end=base + end))
            idx += 1
    return segments


def _split_on_delimiter(text, delimiter):
    """Yield (chunk, base_offset) where base_offset is the chunk's start in `text`."""
    if not delimiter:
        return [(text, 0)]
    chunks = []
    pos = 0
    for part in text.split(delimiter):
        chunks.append((part, pos))
        pos += len(part) + len(delimiter)
    return chunks


def _sentence_spans(chunk):
    """(start, end) char spans of each sentence in `chunk`, from syntok token offsets.
    syntok.analyze preserves offsets into the string it's given, so a sentence spans from
    its first token's offset to the end of its last token.

    Ellipsis runs are masked to a SAME-LENGTH placeholder before syntok so they are not treated
    as sentence boundaries; spans are computed on the masked copy but the caller slices the
    ORIGINAL text, so offsets are preserved and a real terminal '.' still splits."""
    masked = _ELLIPSIS.sub(lambda m: _MASK * len(m.group()), chunk)
    spans = []
    for paragraph in segmenter.analyze(masked):
        for sentence in paragraph:
            tokens = list(sentence)
            if not tokens:
                continue
            start = tokens[0].offset
            last = tokens[-1]
            spans.append((start, last.offset + len(last.value)))
    return spans
