"""Download-only importer: jingjietan/essays-big5 -> raw_data/tan_ocean.parquet."""
import paths  # noqa: F401
from datasets import load_dataset, concatenate_datasets

DATASET = "jingjietan/essays-big5"
RAW = paths.ROOT / "dataset_construction" / "data" / "raw_data" / "tan_ocean.parquet"


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
