"""Common schema for the pre-clean corpus — the column order every dataset manipulator conforms to."""
import hashlib

# Canonical column order of processed_data/pre_clean_data.parquet.
COLUMNS = [
    "text",              # str: the (possibly chunked) content
    "classification",    # str: 4-letter MBTI code, e.g. "INTP"
    "ptype",             # int 0-15: numeric type (source's stated type)
    "O", "C", "E", "A",  # int 0/1: MBTI axis values (jingjietan's encoding)
    "Extravert", "Introvert", "Intuition", "Sensing",
    "Feeling", "Thinking", "Judging", "Perceiving",  # int 0/1: named axis labels
    "derived_PTYPE",     # int 0-15: ptype recomputed from O/C/E/A
    "MATCH",             # bool: does the source's stated type agree with our derivation
    "author_id",         # str|None: hashed author handle; None if the source is anonymous
    "#",                 # int: chunk index within the author (1-based)
    "Tot",               # int: total chunks for the author
    "dataset",           # str: source dataset id
    "id",                # str: unique, time-sortable per-row id
]

# Axis conventions (identical to the original importer — do NOT change; mbti_profiles is
# the validation baseline). Each axis: value 1 -> first entry, 0 -> second.
_LABELS = {  # axis -> (label_when_1, label_when_0)
    "E": ("Extravert", "Introvert"),
    "O": ("Intuition", "Sensing"),
    "A": ("Feeling", "Thinking"),
    "C": ("Judging", "Perceiving"),
}
_LETTER = {  # axis -> (letter_when_1, letter_when_0)
    "E": ("E", "I"),
    "O": ("N", "S"),
    "A": ("F", "T"),
    "C": ("J", "P"),
}
_CODE_ORDER = ["E", "O", "A", "C"]  # MBTI string reads (E/I)(N/S)(F/T)(J/P)
_LETTER_TO_AXIS = {  # inverse, for datasets that give a code/letters instead of axes
    "E": ("E", 1), "I": ("E", 0),
    "N": ("O", 1), "S": ("O", 0),
    "F": ("A", 1), "T": ("A", 0),
    "J": ("C", 1), "P": ("C", 0),
}


def derive_mbti_columns(O, C, E, A):
    """Four axis 0/1 values -> the derived MBTI columns, exactly as the original importer:
    the 8 named label flags, the 4-letter `classification`, and `derived_PTYPE`. Shared by
    every manipulator so all datasets encode type identically."""
    axes = {"O": O, "C": C, "E": E, "A": A}
    out = {}
    for axis, (one, zero) in _LABELS.items():
        out[one] = 1 if axes[axis] == 1 else 0
        out[zero] = 1 if axes[axis] == 0 else 0
    out["classification"] = "".join(
        _LETTER[a][0] if axes[a] == 1 else _LETTER[a][1] for a in _CODE_ORDER
    )
    out["derived_PTYPE"] = 8 * O + 4 * C + 2 * E + 1 * A
    return out


def code_to_axes(code):
    """4-letter MBTI code (e.g. "INFP") -> {O,C,E,A} 0/1. Raises on a malformed code."""
    code = code.strip().upper()
    if len(code) != 4 or any(ch not in _LETTER_TO_AXIS for ch in code):
        raise ValueError(f"malformed MBTI code: {code!r}")
    axes = {}
    for ch in code:
        axis, val = _LETTER_TO_AXIS[ch]
        axes[axis] = val
    return axes  # keys: O, C, E, A


def hash_author(handle, length=16):
    """Deterministic, non-reversible author id from a source handle (username stripped)."""
    return hashlib.sha256(handle.encode("utf-8")).hexdigest()[:length]
