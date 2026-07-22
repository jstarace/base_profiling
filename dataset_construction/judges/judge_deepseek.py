import json
import paths  # noqa: F401
from dataclasses import asdict
from openai import OpenAI as OpenAICore
from dataset_construction.judges.base import Judge, user_message, build_verdict, SMOKE_TEXT
from dataset_construction.judges.prompts import DEFAULT_PANEL_PROMPT, DEFAULT_FINAL_JUDGE_PROMPT, DEFAULT_TRANSFORMER_PROMPT

# DeepSeek has JSON mode but no strict schema, so the panel output shape is pinned in-prompt
# and validated downstream (the quote check in standardize). The literal "JSON" here also
# satisfies deepseek's json_object requirement that the word appear in the prompt.
PANEL_JSON_SPEC = (
    "Respond with ONLY a JSON object of this exact shape, nothing else:\n"
    '{"verdict": true|false, '
    '"flagged": [{"sentence_index": <int>, "quote": "<a few words copied verbatim from that sentence>", '
    '"component": "explicit"|"implicit", "confidence": <number 0-1>, "justification": "<why>"}], '
    '"justification": "<overall justification>"}\n'
    "When verdict is false, flagged is an empty list."
)


class Deepseek(Judge):
    """DeepSeek model adapter (OpenAI-compatible endpoint). A FULL peer: it fills the panel
    role (detect_profile_mention), the final-judge role (inherited decide), and drives the
    transformer (inherited complete_json) — so it can be swapped into any slot. deepseek-chat
    has JSON mode but no strict schema, so the panel output shape is pinned in-prompt and the
    verdict parse is tolerant."""

    DEFAULT_MODEL = "deepseek-chat"

    # This model's role prompts. Override any default with a bespoke string to tune THIS
    # model for that role, without affecting any other model.
    PANEL_PROMPT = DEFAULT_PANEL_PROMPT
    FINAL_JUDGE_PROMPT = DEFAULT_FINAL_JUDGE_PROMPT
    TRANSFORMER_PROMPT = DEFAULT_TRANSFORMER_PROMPT

    def __init__(self, model=None):
        super().__init__(model)
        self.client = OpenAICore(base_url="https://api.deepseek.com", api_key=self.api_key)

    def detect_profile_mention(self, text, canonical):
        data = self.complete_json(self.PANEL_PROMPT, f"{user_message(text, canonical)}\n\n{PANEL_JSON_SPEC}")
        return build_verdict(data)

    def test_setup(self):
        from dataset_construction.cleaning.segment import segment
        canonical = segment(SMOKE_TEXT)
        print(json.dumps(asdict(self.detect_profile_mention(SMOKE_TEXT, canonical)), indent=2, ensure_ascii=False))

    def _complete_text(self, system_prompt, user_content) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        # reasoner has no JSON mode and ignores temperature; chat supports both.
        if "reasoner" not in self.model:
            kwargs["temperature"] = 0.2
            kwargs["response_format"] = {"type": "json_object"}
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""
