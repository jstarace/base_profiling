import json
import paths  # noqa: F401
from dataclasses import asdict
from openai import OpenAI as OpenAICore
from dataset_construction.judges.base import Judge, FLAGGED_ITEM, user_message, build_verdict, SMOKE_TEXT
from dataset_construction.judges.prompts import DEFAULT_PANEL_PROMPT, DEFAULT_FINAL_JUDGE_PROMPT, DEFAULT_TRANSFORMER_PROMPT
from dataset_construction.cleaning.types import InitialVerdict

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "boolean"},
        "flagged": {"type": "array", "items": {**FLAGGED_ITEM, "additionalProperties": False}},
        "justification": {"type": "string"},
    },
    "required": ["verdict", "flagged", "justification"],
    "additionalProperties": False,
}


class OpenAI(Judge):
    DEFAULT_MODEL = "gpt-4o"

    # This model's role prompts. Override any default with a bespoke string to tune THIS
    # model for that role, without affecting any other model.
    PANEL_PROMPT = DEFAULT_PANEL_PROMPT
    FINAL_JUDGE_PROMPT = DEFAULT_FINAL_JUDGE_PROMPT
    TRANSFORMER_PROMPT = DEFAULT_TRANSFORMER_PROMPT

    def __init__(self, model=None):
        super().__init__(model)
        self.client = OpenAICore()

    def detect_profile_mention(self, text, canonical) -> InitialVerdict:
        return self._evaluate(self.PANEL_PROMPT, text, canonical)

    def test_setup(self):
        from dataset_construction.cleaning.segment import segment
        canonical = segment(SMOKE_TEXT)
        print(json.dumps(asdict(self.detect_profile_mention(SMOKE_TEXT, canonical)), indent=2, ensure_ascii=False))

    def _evaluate(self, prompt, text, canonical) -> InitialVerdict:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message(text, canonical)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "record_verdict", "schema": VERDICT_SCHEMA, "strict": True},
            },
        )
        message = completion.choices[0].message
        if message.refusal:
            raise ValueError(f"model refused: {message.refusal}")
        return build_verdict(json.loads(message.content))

    def _complete_text(self, system_prompt, user_content) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return completion.choices[0].message.content or ""
