"""TEMPLATE — copy to add a model adapter: fill in the SDK client and _complete_text."""
import paths  # noqa: F401
from dataset_construction.judges.base import Judge, user_message, build_verdict, SMOKE_TEXT  # noqa: F401
from dataset_construction.judges.prompts import DEFAULT_PANEL_PROMPT, DEFAULT_FINAL_JUDGE_PROMPT, DEFAULT_TRANSFORMER_PROMPT


class YourModel(Judge):
    DEFAULT_MODEL = "provider-model-id"          # e.g. "gpt-4o", "claude-haiku-4-5"

    # This model's role prompts. Replace a default with a bespoke triple-quoted string to
    # tune THIS model for that role, without touching any other model.
    PANEL_PROMPT = DEFAULT_PANEL_PROMPT
    FINAL_JUDGE_PROMPT = DEFAULT_FINAL_JUDGE_PROMPT
    TRANSFORMER_PROMPT = DEFAULT_TRANSFORMER_PROMPT

    def __init__(self, model=None):
        super().__init__(model)   # sets self.api_key (from <CLASSNAME>_API_KEY) and self.model
        # self.client = ProviderSDK(api_key=self.api_key)
        raise NotImplementedError("wire up the provider SDK client here")

    def _complete_text(self, system_prompt, user_content) -> str:
        """The ONE provider-specific call the final-judge and transformer roles need: send
        system + user, return the model's text. Role code parses the JSON out of it."""
        raise NotImplementedError

    # Implement ONLY if this model will sit on the panel:
    # def detect_profile_mention(self, text, canonical):
    #     raw = <call your model with self.PANEL_PROMPT + user_message(text, canonical),
    #            asking for the flagged-by-index schema>
    #     return build_verdict(raw)
