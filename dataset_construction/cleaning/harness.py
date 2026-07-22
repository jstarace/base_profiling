import paths
import pandas as pd
from datetime import datetime, timezone
from dataclasses import asdict

from dataset_construction.judges.judge_gemini import Gemini
from dataset_construction.judges.judge_anthropic import Anthropic
from dataset_construction.judges.judge_openai import OpenAI
from dataset_construction.judges.judge_deepseek import Deepseek
from dataset_construction.cleaning.segment import segment
from dataset_construction.cleaning.standardize import standardize, majority
from dataset_construction.cleaning.transformer import Transformer
from dataset_construction.cleaning.types import CleaningRecord
from dataset_construction.utilities.run_log import log_record

PANEL_JUDGES = [Gemini, Anthropic, OpenAI]   # role: panel
FINAL_JUDGE = Deepseek                          # role: final judge (also drives the transformer)
FINAL_JUDGE_MODEL = None                        # None = the judge's own DEFAULT_MODEL


def main():
    sample = pd.read_parquet(paths.ROOT / "dataset_construction" / "data" / "mbti_sample_prompt_tuning.parquet")
    Harness().clean_data(sample)


class Harness:
    """Controls the panel of LLMs. Wiring only — it decides nothing about the text itself:
    the judges decide, the transformer edits, this just moves each row through the flow
    (segment -> panel -> gates -> final judge -> transform -> log)."""

    def __init__(self):
        self.panel = [judge() for judge in PANEL_JUDGES]
        final_judge = FINAL_JUDGE(model=FINAL_JUDGE_MODEL)   # same instance also drives the transformer
        self.final_judge = final_judge
        self.transformer = Transformer(final_judge)
        self.needed = majority(len(self.panel))

    def clean_row(self, row) -> dict:
        """Run one row through the full flow and return its cleaned row dict
        ({**row, "text": final_text}). Logs the full CleaningRecord as a side effect."""
        text = row.text
        dataset = row.get("dataset")

        # 1) canonical sentences: the one coordinate system every judge indexes into
        canonical = segment(text, dataset)

        # 2) panel votes by canonical index; each flag carries a validation quote
        verdicts = [judge.detect_profile_mention(text, canonical) for judge in self.panel]

        # 3) tally validated index-votes; a flag whose quote misses its index is dropped
        records, mismatches = standardize(canonical, verdicts, self.needed)

        # 4) row gate: on the OVERALL verdicts, not on whether a majority sentence exists
        flagged_count = sum(1 for v in verdicts if v.verdict)
        needs_cleaning = flagged_count >= self.needed

        # 5) sentence gate: only indices that reached majority go to the final judge
        majority_records = [r for r in records if r.vote_count >= self.needed]

        final_decisions = []
        rephrases = []
        final_text = text
        all_no_action = False

        if needs_cleaning and majority_records:
            # 6) final judge sees ONLY the majority sentences; cut/rephrase/veto each
            final_decisions = self.final_judge.decide(text, majority_records)
            actions = {d.sentence_index: d.action for d in final_decisions}
            for record in records:
                if record.sentence_index in actions:
                    record.action = actions[record.sentence_index]

            # 7) the transformer owns all mutation; it edits by offset span
            final_text, rephrases = self.transformer.apply(text, final_decisions, canonical)
        elif needs_cleaning:
            # passed the row gate but no sentence reached majority: nothing reviewed,
            # text unchanged, still logged as needs_cleaning. Expected-rare; surfaced.
            all_no_action = True

        # 8) log every row: full flagged set (with votes + actions) for provenance,
        #    acted set separately, dropped-flag mismatches for review
        record = CleaningRecord(
            id=row.get("id"),
            source=dataset,
            created_at=datetime.now(timezone.utc).isoformat(),
            original_text=text,
            initial_verdicts=[_verdict_for_log(v) for v in verdicts],
            sentences=[asdict(s) for s in records],
            flag_mismatches=mismatches,
            panel_decision="needs_cleaning" if needs_cleaning else "clean",
            all_no_action=all_no_action,
            final_decisions=[asdict(d) for d in final_decisions],
            rephrases=[asdict(r) for r in rephrases],
            final_text=final_text,
        )
        log_record(record)
        return {**row.to_dict(), "text": final_text}

    def clean_data(self, content: pd.DataFrame) -> pd.DataFrame:
        cleaned_rows = [self.clean_row(row) for _, row in content.iterrows()]
        cleaned_df = pd.DataFrame(cleaned_rows)
        cleaned_df.to_parquet(paths.ROOT / "dataset_construction" / "data" / "cleaned.parquet", index=False)
        return cleaned_df


def _verdict_for_log(verdict):
    """Log a judge's verdict WITHOUT the validation quotes — quotes are checked in
    standardize, then discarded, never stored."""
    payload = asdict(verdict)
    for flag in payload.get("flagged", []):
        flag.pop("quote", None)
    return payload


if __name__ == "__main__":
    main()
