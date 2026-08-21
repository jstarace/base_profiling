"""Full-profile drift study infrastructure."""

BASE_MODEL = "meta-llama/Llama-3.1-8B"
BASE_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
TRAITS = tuple("OCEAN")
WEIGHTS = (16, 8, 4, 2, 1)
PTYPES = tuple(range(32))
LAYERS = tuple(range(32))
POOLS = ("final_token", "mean_tokens")
