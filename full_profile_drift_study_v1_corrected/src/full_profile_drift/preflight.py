"""Read-only integrity and resource preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from full_profile_drift import BASE_MODEL, BASE_REVISION
from full_profile_drift.io import atomic_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, default=Path("/workspace"))
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--model-snapshot", type=Path, required=True)
    p.add_argument("--adapter-root", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--verification", type=Path, required=True)
    args = p.parse_args()
    import pandas as pd
    catalog = pd.read_csv(args.catalog)
    verification = json.loads(args.verification.read_text())
    free = shutil.disk_usage(args.workspace).free
    checks = {
        "minimum_free_bytes": 60 * 1024 ** 3,
        "available_free_bytes": free,
        "disk_space_pass": free >= 60 * 1024 ** 3,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "model_snapshot_path": str(args.model_snapshot),
        "model_snapshot_present": args.model_snapshot.is_dir() and args.model_snapshot.name == BASE_REVISION,
        "adapter_directory_count": len(list(args.adapter_root.glob("ptype_*"))),
        "adapters": [],
    }
    for row in catalog.itertuples():
        path = args.adapter_root / row.adapter / "adapter_model.safetensors"
        checks["adapters"].append({"adapter":row.adapter,"profile":row.human_readable_profile,"path":str(path),"present":path.is_file(),"expected_sha256":row.adapter_sha256,"actual_sha256":sha256(path) if path.is_file() else None})
    checks["adapter_hashes_pass"] = all(x["present"] and x["expected_sha256"] == x["actual_sha256"] for x in checks["adapters"])
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    checks["tokenizer"] = {name:{"base":sha256(args.model_snapshot/name),"adapter_0":sha256(args.adapter_root/"ptype_0"/name),
        "all_adapter_hashes_identical":len({sha256(args.adapter_root/f"ptype_{ptype}"/name) for ptype in range(32)}) == 1} for name in tokenizer_files}
    semantic = [row["tokenizer"]["semantically_matches_base"] for row in verification["adapters"]]
    checks["tokenizer_semantically_matches_base_all_32"] = all(semantic)
    checks["tokenizer_files_identical_across_adapters"] = all(item["all_adapter_hashes_identical"] for item in checks["tokenizer"].values())
    checks["tokenizer_match"] = checks["tokenizer_semantically_matches_base_all_32"] and checks["tokenizer_files_identical_across_adapters"]
    checks["pass"] = all((checks["disk_space_pass"], checks["model_snapshot_present"], checks["adapter_directory_count"] == 32, checks["adapter_hashes_pass"], checks["tokenizer_match"]))
    (args.project / "audit").mkdir(parents=True, exist_ok=True)
    atomic_json(args.project / "audit/preflight.json", checks)
    if not checks["pass"]:
        raise SystemExit("PREFLIGHT_FAILED")
    print(json.dumps({"status":"PASS","free_bytes":free,"adapters":32}))


if __name__ == "__main__":
    main()
