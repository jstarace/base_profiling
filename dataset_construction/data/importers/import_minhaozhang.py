"""Download-only importer: minhaozhang/mbti (Reddit) -> raw_data/minhaozhang.parquet."""
import paths  # noqa: F401
from datasets import load_dataset, concatenate_datasets

DATASET = "minhaozhang/mbti"
RAW = paths.ROOT / "dataset_construction" / "data" / "raw_data" / "minhaozhang.parquet"


def main():
    if RAW.exists():
        print(f"[import] {DATASET}: raw present, skip")
        return RAW
    ds = load_dataset(DATASET)
    df = concatenate_datasets(list(ds.values())).to_pandas()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW)
    print(f"[import] {DATASET}: {len(df):,} rows -> {RAW}")
    return RAW


if __name__ == "__main__":
    main()
