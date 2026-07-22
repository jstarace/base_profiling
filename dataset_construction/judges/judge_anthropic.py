import json
import paths  # noqa: F401
from dataclasses import asdict
from anthropic import Anthropic as AnthropicCore
from dataset_construction.judges.base import Judge, FLAGGED_ITEM, user_message, build_verdict, SMOKE_TEXT
from dataset_construction.judges.prompts import DEFAULT_PANEL_PROMPT, DEFAULT_FINAL_JUDGE_PROMPT, DEFAULT_TRANSFORMER_PROMPT
from dataset_construction.cleaning.types import InitialVerdict

VERDICT_TOOL = {
    "name": "record_verdict",
    "description": "Record the judge's verdict on the input text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "boolean"},
            "flagged": {"type": "array", "items": {**FLAGGED_ITEM, "additionalProperties": False}},
            "justification": {"type": "string"},
        },
        "required": ["verdict", "flagged", "justification"],
        "additionalProperties": False,
    },
}


class Anthropic(Judge):
    DEFAULT_MODEL = "claude-haiku-4-5"

    # This model's role prompts. Override any default with a bespoke string to tune THIS
    # model for that role, without affecting any other model.
    PANEL_PROMPT = DEFAULT_PANEL_PROMPT
    FINAL_JUDGE_PROMPT = DEFAULT_FINAL_JUDGE_PROMPT
    TRANSFORMER_PROMPT = DEFAULT_TRANSFORMER_PROMPT

    def __init__(self, model=None):
        super().__init__(model)
        self.client = AnthropicCore(api_key=self.api_key)

    def detect_profile_mention(self, text, canonical) -> InitialVerdict:
        return self._evaluate(self.PANEL_PROMPT, text, canonical)

    def test_setup(self):
        from dataset_construction.cleaning.segment import segment
        canonical = segment(SMOKE_TEXT)
        print(json.dumps(asdict(self.detect_profile_mention(SMOKE_TEXT, canonical)), indent=2, ensure_ascii=False))

    def _evaluate(self, prompt, text, canonical) -> InitialVerdict:
        message = self.client.messages.create(
            model=self.model,
            system=prompt,
            max_tokens=15360,
            tools=[VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "record_verdict", "disable_parallel_tool_use": True},
            messages=[{"role": "user", "content": user_message(text, canonical)}],
        )
        for block in message.content:
            if block.type == "tool_use" and block.name == "record_verdict":
                return build_verdict(block.input)
        raise ValueError("no verdict returned")

    def _complete_text(self, system_prompt, user_content) -> str:
        message = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_content}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
