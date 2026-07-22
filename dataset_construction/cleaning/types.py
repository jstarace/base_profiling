import paths  # noqa: F401 — loads .env / project root, consistent with other modules
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Segment:
    """One canonical sentence with its character span in the ORIGINAL text.
    Offsets are the source of truth: original_text[start:end] == text. This is what
    lets the transformer edit by span instead of by fragile string matching."""
    index: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Flag:
    """One judge flagging one canonical index. `quote` is validation-only: checked
    against the canonical sentence to catch mis-indexing, then discarded — never stored."""
    sentence_index: int
    quote: str
    component: str        # "explicit" | "implicit"
    confidence: float
    justification: str


@dataclass(frozen=True)
class InitialVerdict:
    """A panel judge's output: the overall verdict plus the canonical indices it flags."""
    verdict: bool
    justification: str
    flagged: list = field(default_factory=list)   # list[Flag]


@dataclass
class SentenceRecord:
    """A canonical sentence after the panel votes — anonymized for the final judge.
    Carries the vote COUNT (not which judges) so the final judge isn't biased by identity;
    justifications/components are pooled without attribution. `action` is the single
    per-sentence verdict:
        no_action              gate-assigned, sub-majority — never reviewed
        cut | rephrase | veto  final-judge-assigned, majority sentences only
    no_action and veto both leave text unchanged but stay distinct: no_action was never
    seen, veto was reviewed and overturned."""
    sentence_index: int
    sentence: str
    start: int
    end: int
    vote_count: int
    justifications: list = field(default_factory=list)
    components: list = field(default_factory=list)
    max_confidence: float | None = None
    action: str | None = None            # None = majority, pending the final judge


@dataclass
class FinalDecision:
    """Final judge's action for one majority sentence: 'cut' | 'rephrase' | 'veto'."""
    sentence_index: int
    sentence: str
    action: str
    justification: str


@dataclass
class Rephrase:
    """Transformer output; only produced when action == 'rephrase'. `justification`
    is the model explaining what self-reference it removed and how it kept the meaning."""
    sentence_index: int
    original: str
    rephrased: str
    justification: str = ""


@dataclass
class CleaningRecord:
    """The full lifecycle of one row through the pipeline, logged as a single JSONL record.
    Capture everything; trim later."""
    id: str                          # the row uuid (dataset 'id' column)
    source: str                      # which dataset (dataset column)
    created_at: str                  # when this record was produced (UTC ISO)
    original_text: str               # the row as ingested
    initial_verdicts: list           # per-judge InitialVerdicts (quotes stripped), asdict'd
    sentences: list = field(default_factory=list)        # ALL flagged SentenceRecords: index, votes, action
    flag_mismatches: list = field(default_factory=list)  # dropped flags whose quote missed its index
    panel_decision: str | None = None                    # "needs_cleaning" | "clean"
    all_no_action: bool = False                          # passed row gate but no sentence reached majority
    final_decisions: list = field(default_factory=list)  # reviewed/acted sentences only
    rephrases: list = field(default_factory=list)        # executed rewrites
    final_text: str | None = None                        # cleaned result after actions
