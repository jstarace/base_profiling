import json
import paths  # noqa: F401
from dataclasses import asdict
from google import genai
from google.genai import types
from dataset_construction.judges.base import Judge, FLAGGED_ITEM, user_message, build_verdict, SMOKE_TEXT
from dataset_construction.judges.prompts import DEFAULT_PANEL_PROMPT, DEFAULT_FINAL_JUDGE_PROMPT, DEFAULT_TRANSFORMER_PROMPT
from dataset_construction.cleaning.types import InitialVerdict

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "boolean"},
        "flagged": {"type": "array", "items": FLAGGED_ITEM},
        "justification": {"type": "string"},
    },
    "required": ["verdict", "flagged", "justification"],
}


class Gemini(Judge):
    DEFAULT_MODEL = "gemini-2.5-flash"

    # This model's role prompts. Override any default with a bespoke string to tune THIS
    # model for that role, without affecting any other model.
    PANEL_PROMPT = DEFAULT_PANEL_PROMPT
    FINAL_JUDGE_PROMPT = DEFAULT_FINAL_JUDGE_PROMPT
    TRANSFORMER_PROMPT = DEFAULT_TRANSFORMER_PROMPT

    def __init__(self, model=None):
        super().__init__(model)
        self.client = genai.Client()

    def detect_profile_mention(self, text, canonical) -> InitialVerdict:
        return self._evaluate(self.PANEL_PROMPT, text, canonical)

    def test_setup(self):
        from dataset_construction.cleaning.segment import segment
        canonical = segment(SMOKE_TEXT)
        print(json.dumps(asdict(self.detect_profile_mention(SMOKE_TEXT, canonical)), indent=2, ensure_ascii=False))

    def _evaluate(self, prompt, text, canonical) -> InitialVerdict:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_message(text, canonical),
            config=types.GenerateContentConfig(
                system_instruction=prompt,
                response_mime_type="application/json",
                response_schema=VERDICT_SCHEMA,
                temperature=0.1,
            ),
        )
        if not response.text:
            raise ValueError("no verdict returned")
        return build_verdict(json.loads(response.text))

    def _complete_text(self, system_prompt, user_content) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return response.text or ""
