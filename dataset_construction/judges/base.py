import os
import json
import paths  # noqa: F401
from abc import ABC
from dataset_construction.cleaning.types import InitialVerdict, Flag, FinalDecision

_ACTIONS = {"cut", "rephrase", "veto"}
_JSON_NUDGE = "\n\nReturn ONLY the JSON object described, nothing else."


class Judge(ABC):
    """Role-agnostic base for every model adapter — fills the panel, final-judge, or
    transformer role depending on how the harness wires it."""

    DEFAULT_MODEL = None

    def __init__(self, model=None):
        # env var is CLASSNAME + "_API_KEY", matching .env: Anthropic -> ANTHROPIC_API_KEY,
        # Deepseek -> DEEPSEEK_API_KEY, OpenAI -> OPENAI_API_KEY, Gemini -> GEMINI_API_KEY.
        self.api_key = os.getenv(f"{self.__class__.__name__.upper()}_API_KEY")
        self.model = model or self.DEFAULT_MODEL

    # --- panel role (adapters that sit on the panel override this) ---
    def detect_profile_mention(self, text, canonical) -> InitialVerdict:
        raise NotImplementedError(f"{self.__class__.__name__} does not fill the panel role")

    # --- final-judge role (generic; any adapter with _complete_text can be the final judge) ---
    def decide(self, text, review_records):
        """One FinalDecision per majority sentence: 'cut' | 'rephrase' | 'veto'. Decides
        only; never mutates text. Uses THIS model's FINAL_JUDGE_PROMPT. A sentence the model
        omits or returns an unknown action for is left unchanged (veto) with a note — never
        silently dropped or guessed."""
        data = self.complete_json(self.FINAL_JUDGE_PROMPT, _render_records(text, review_records))
        by_index = {
            d["sentence_index"]: d
            for d in (data.get("decisions") or [])
            if isinstance(d, dict) and isinstance(d.get("sentence_index"), int)
        }
        decisions = []
        for record in review_records:
            entry = by_index.get(record.sentence_index) or {}
            action = entry.get("action")
            if action in _ACTIONS:
                decisions.append(FinalDecision(
                    sentence_index=record.sentence_index,
                    sentence=record.sentence,
                    action=action,
                    justification=(entry.get("justification") or "").strip(),
                ))
            else:
                decisions.append(FinalDecision(
                    sentence_index=record.sentence_index,
                    sentence=record.sentence,
                    action="veto",
                    justification="final judge returned no valid action for this sentence; left unchanged",
                ))
        return decisions

    # --- shared model primitive ---
    def complete_json(self, system_prompt, user_content):
        """Prompt the model for JSON, parse it robustly, retry once. Providers implement
        only _complete_text; JSON parsing is handled here so it's uniform across models."""
        for attempt in range(2):
            content = self._complete_text(system_prompt, user_content if attempt == 0 else user_content + _JSON_NUDGE)
            data = _find_json(content)
            if data is not None:
                return data
        return {}

    def _complete_text(self, system_prompt, user_content) -> str:
        """One-shot text completion via this provider's SDK. Each adapter implements it;
        this is the ONLY provider-specific method the final-judge/transformer roles need."""
        raise NotImplementedError(f"{self.__class__.__name__} has no _complete_text")

    def test_setup(self):
        """Optional quick config/connectivity check for a new judge."""
        raise NotImplementedError


# ---- shared plumbing ----

# One flagged canonical sentence. Providers that accept `additionalProperties` add it
# themselves (Gemini's schema dialect rejects the key), so it's omitted from the shared item.
FLAGGED_ITEM = {
    "type": "object",
    "properties": {
        "sentence_index": {"type": "integer"},
        "quote": {"type": "string"},
        "component": {"type": "string"},
        "confidence": {"type": "number"},
        "justification": {"type": "string"},
    },
    "required": ["sentence_index", "quote", "component", "confidence", "justification"],
}


def numbered_sentences(canonical):
    """Render the canonical array as the judge sees it: one indexed line per sentence."""
    return "\n".join(f"[{seg.index}] {seg.text}" for seg in canonical)


def user_message(text, canonical):
    """The judge reads meaning from the raw text but may ONLY flag by canonical index."""
    return (
        "RAW TEXT (read for meaning and context):\n"
        f"{text}\n\n"
        "NUMBERED CANONICAL SENTENCES (flag ONLY by these indices):\n"
        f"{numbered_sentences(canonical)}"
    )


def build_verdict(data):
    """Turn a panel judge's raw tool/JSON output into an InitialVerdict of Flags. Tolerant
    of loosely-shaped output (e.g. DeepSeek's JSON mode, which has no strict schema): a
    malformed flag entry is skipped rather than crashing the whole row."""
    flags = []
    for f in (data.get("flagged") or []):
        if not isinstance(f, dict):
            continue
        try:
            flags.append(Flag(
                sentence_index=int(f["sentence_index"]),
                quote=str(f.get("quote", "")),
                component=str(f.get("component", "")),
                confidence=float(f.get("confidence", 0.0)),
                justification=str(f.get("justification", "")),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return InitialVerdict(
        verdict=bool(data.get("verdict", False)),
        justification=str(data.get("justification", "")),
        flagged=flags,
    )


def _render_records(text, records):
    """The final judge's input: the full post + the majority sentences (anonymized)."""
    lines = ["FLAGGED SENTENCES:"]
    for r in records:
        justifications = " | ".join(j for j in (r.justifications or []) if j)
        confidence = f"{r.max_confidence:.2f}" if r.max_confidence is not None else "n/a"
        lines.append(f"[{r.sentence_index}] (judges_flagged={r.vote_count}, top_confidence={confidence})")
        lines.append(f"    sentence: {r.sentence}")
        lines.append(f"    panel_justifications: {justifications}")
    return f"RAW POST:\n{text}\n\n" + "\n".join(lines)


def _find_json(content):
    """Best-effort parse of a JSON object out of the model's text: raw, then de-fenced,
    then the substring between the outermost braces."""
    candidates = [content]
    stripped = content.strip()
    if stripped.startswith("```"):
        inner = stripped.strip("`")
        if inner.lower().startswith("json"):
            inner = inner[4:]
        candidates.append(inner)
    if "{" in content and "}" in content:
        candidates.append(content[content.index("{"): content.rindex("}") + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


# A tiny self-reference for test_setup smoke checks: index 0 flags, index 1 does not.
SMOKE_TEXT = "I'm definitely an INTP, that describes me perfectly. My sister is an ENFJ though."
