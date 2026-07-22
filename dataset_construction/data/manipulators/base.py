"""Base for dataset manipulators: raw source DataFrame -> rows conforming to schema.COLUMNS."""
import paths  # noqa: F401
from abc import ABC, abstractmethod

from dataset_construction.data import schema
from dataset_construction.utilities.ids import add_row_ids


class Manipulator(ABC):
    DATASET = None

    @abstractmethod
    def manipulate(self, raw):
        """raw (source DataFrame) -> DataFrame carrying every schema column except the
        provenance pair (`dataset`, `id`), which `finalize` stamps. Call `finalize` last."""

    def finalize(self, df):
        """Stamp `dataset` + a fresh `id` per row, enforce the schema exactly, order columns."""
        df = df.copy()
        df["dataset"] = self.DATASET
        add_row_ids(df)
        missing = [c for c in schema.COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in schema.COLUMNS]
        if missing:
            raise ValueError(f"{self.DATASET}: missing schema columns {missing}")
        if extra:
            raise ValueError(f"{self.DATASET}: unexpected columns {extra}")
        return df[schema.COLUMNS]
