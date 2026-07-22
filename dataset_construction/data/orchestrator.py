"""Download missing raw sets -> manipulate each -> ONE combined processed_data/pre_clean_data.parquet."""
import paths  # noqa: F401
import pandas as pd

from dataset_construction.data.importers import import_tan_mbti, import_minhaozhang
from dataset_construction.data.manipulators.manipulate_tan_mbti import TanMbti
from dataset_construction.data.manipulators.manipulate_minhaozhang import Minhaozhang

PROCESSED = paths.ROOT / "dataset_construction" / "data" / "processed_data" / "pre_clean_data.parquet"

# dataset name -> (importer module, manipulator instance)
PIPELINE = {
    import_tan_mbti.DATASET: (import_tan_mbti, TanMbti()),
    import_minhaozhang.DATASET: (import_minhaozhang, Minhaozhang()),
}


def run(datasets=None, write=True):
    targets = PIPELINE if datasets is None else {d: PIPELINE[d] for d in datasets}
    frames = []
    for name, (importer, manip) in targets.items():
        importer.main()                       # download only if missing
        raw = pd.read_parquet(importer.RAW)
        print(f"[manipulate] {name}: {len(raw):,} raw rows ...")
        out = manip.manipulate(raw)
        print(f"[manipulate] {name}: -> {len(out):,} rows")
        frames.append(out)
    combined = pd.concat(frames, ignore_index=True)
    if write:
        PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(PROCESSED)
        print(f"[orchestrator] wrote {len(combined):,} rows -> {PROCESSED}")
    return combined


def main():
    run()


if __name__ == "__main__":
    main()
