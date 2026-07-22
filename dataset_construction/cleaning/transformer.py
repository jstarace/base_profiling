import paths  # noqa: F401
from dataset_construction.cleaning.types import Rephrase


class Transformer:
    """Owns ALL text mutation. It edits by char-offset span — so an identical sentence
    elsewhere is never touched, and everything between edited spans stays byte-for-byte,
    including whitespace and the ||| delimiters. The rewriting model is INJECTED: the
    transformer holds a model adapter and uses that model's own TRANSFORMER_PROMPT, so
    swapping the post-panel model swaps the rephraser too."""

    def __init__(self, model):
        self.model = model   # a Judge adapter, used here as a JSON-completion client

    def apply(self, text, final_decisions, canonical):
        """Return (final_text, list[Rephrase]).

        cut removes the sentence's span, rephrase replaces it, veto leaves it. Edits are
        applied in DESCENDING offset order so an earlier edit never shifts a later span.
        Byte-faithful everywhere except the edited spans.
        """
        by_index = {seg.index: seg for seg in canonical}
        edits = []          # (start, end, replacement)
        rephrases = []

        for decision in final_decisions:
            seg = by_index.get(decision.sentence_index)
            if seg is None:
                continue
            if decision.action == "cut":
                edits.append((seg.start, seg.end, ""))
            elif decision.action == "rephrase":
                rephrase = self.rephrase(text, decision.sentence_index, seg.text, decision.justification)
                rephrases.append(rephrase)
                edits.append((seg.start, seg.end, rephrase.rephrased))
            # "veto" (and anything else) -> no change

        final_text = text
        for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
            final_text = final_text[:start] + replacement + final_text[end:]
        return final_text, rephrases

    def rephrase(self, text, sentence_index, sentence, justification):
        """Rewrite one sentence to drop the profile mention, preserving meaning. Falls back
        to the original sentence if the model returns nothing usable."""
        user = (
            f"POST:\n{text}\n\n"
            f"TARGET SENTENCE:\n{sentence}\n\n"
            f"REASON IT WAS FLAGGED (the self-reference to remove):\n{justification}"
        )
        data = self.model.complete_json(self.model.TRANSFORMER_PROMPT, user)
        rephrased = data.get("rephrased") or sentence
        return Rephrase(
            sentence_index=sentence_index,
            original=sentence,
            rephrased=rephrased,
            justification=(data.get("justification") or "").strip(),
        )
