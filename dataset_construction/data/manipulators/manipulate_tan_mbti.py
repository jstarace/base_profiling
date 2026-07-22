"""jingjietan/kaggle-mbti manipulator -> common schema (reproduces the mbti_profiles baseline)."""
import paths  # noqa: F401
import pandas as pd

from dataset_construction.data import schema
from dataset_construction.data.manipulators.base import Manipulator

_DERIVED = ["classification", "Extravert", "Introvert", "Intuition", "Sensing",
            "Feeling", "Thinking", "Judging", "Perceiving", "derived_PTYPE"]


class TanMbti(Manipulator):
    DATASET = "jingjietan/kaggle-mbti"

    def manipulate(self, raw):
        df = pd.DataFrame({"text": raw["text"].astype(str)})
        for ax in ["O", "C", "E", "A"]:
            df[ax] = raw[ax].astype(int)
        df["ptype"] = raw["ptype"].astype(int)

        derived = [schema.derive_mbti_columns(o, c, e, a)
                   for o, c, e, a in zip(df["O"], df["C"], df["E"], df["A"])]
        for col in _DERIVED:
            df[col] = [d[col] for d in derived]

        df["MATCH"] = df["derived_PTYPE"] == df["ptype"]
        df["author_id"] = None
        df["#"] = 1
        df["Tot"] = 1
        return self.finalize(df)
