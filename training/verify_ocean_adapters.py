"""Read-only structural and functional verification for the 32 OCEAN LoRA adapters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "meta-llama/Llama-3.1-8B"
PROMPT = "Write a short reflective paragraph about planning a difficult project."
GENERATION_CONFIG = {
    "do_sample": False,
    "max_new_tokens": 48,
    "num_beams": 1,
    "use_cache": True,
}
EXPECTED_PTYPES = list(range(32))
EXPECTED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def main() -> None:
    args = parse_args()
    adapters_dir = args.adapters_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    discovered = discover_adapters(adapters_dir)
    discovery_ok = sorted(discovered) == EXPECTED_PTYPES
    revision = resolve_revision(args.cache_dir, args.base_model)
    snapshot = snapshot_path(args.cache_dir, args.base_model, revision)
    environment = environment_info(revision, snapshot)

    report = {
        "scope": (
            "Functional artifact verification only. No claim is made that an adapter "
            "represents its intended personality profile."
        ),
        "started_utc": utc_now(),
        "adapters_dir": str(adapters_dir),
        "output_dir": str(output_dir),
        "expected_ptypes": EXPECTED_PTYPES,
        "discovered_ptypes": sorted(discovered),
        "discovery_exact": discovery_ok,
        "missing_ptypes": sorted(set(EXPECTED_PTYPES) - set(discovered)),
        "extra_ptypes": sorted(set(discovered) - set(EXPECTED_PTYPES)),
        "prompt": PROMPT,
        "generation_config": GENERATION_CONFIG,
        "environment": environment,
        "base_inference": {},
        "base_contamination_check": {},
        "adapters": [],
    }

    structural_reference = None
    for ptype in EXPECTED_PTYPES:
        row = new_row(ptype, discovered.get(ptype))
        if ptype not in discovered:
            row["warnings"].append("adapter directory missing")
            report["adapters"].append(row)
            continue
        try:
            row["structural"] = inspect_safetensors(discovered[ptype])
            signature = row["structural"]["tensor_signature"]
            if structural_reference is None:
                structural_reference = signature
            row["structural"]["names_shapes_consistent"] = signature == structural_reference
            row["tokenizer"] = tokenizer_file_check(discovered[ptype], snapshot)
        except Exception as exc:
            row["structural"]["error"] = format_exception(exc)
            row["warnings"].append("structural validation failed")
        report["adapters"].append(row)
        write_reports(report, output_dir)

    if not discovery_ok:
        report["fatal_error"] = "Adapter discovery did not match exactly ptype_0 through ptype_31."
        finish(report, output_dir, started)
        raise SystemExit(2)

    base_tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(discovered[0], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    semantic_tokenizer = semantic_tokenizer_check(base_tokenizer, tokenizer)
    report["tokenizer_verification"] = semantic_tokenizer
    for row in report["adapters"]:
        row["tokenizer"]["semantically_matches_base"] = semantic_tokenizer["semantically_matches_base"]
        row["tokenizer"]["identical_across_all_adapters"] = all(
            report["adapters"][0]["tokenizer"].get(name, {}).get("adapter_sha256")
            == row["tokenizer"].get(name, {}).get("adapter_sha256")
            for name in ("tokenizer.json", "tokenizer_config.json")
        )
    inputs = tokenizer(PROMPT, return_tensors="pt")

    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    ).eval()
    inputs = {name: tensor.to("cuda:0") for name, tensor in inputs.items()}

    with torch.inference_mode():
        base_logits = model(**inputs).logits[:, -1, :].float().cpu()
        base_ids = model.generate(**inputs, **generation_kwargs(tokenizer))
    base_completion = decode_completion(tokenizer, inputs["input_ids"], base_ids)
    report["base_inference"] = generation_record(base_completion, base_logits)
    write_reports(report, output_dir)

    for row in report["adapters"]:
        ptype = row["ptype"]
        adapter_dir = Path(row["path"])
        try:
            torch.cuda.empty_cache()
            row["cuda_memory"]["before_adapter_attachment"] = cuda_memory()
            config = PeftConfig.from_pretrained(adapter_dir)
            target_modules = set(config.target_modules or [])
            row["peft"]["configured_target_modules"] = sorted(target_modules)
            row["peft"]["target_module_mismatches"] = sorted(
                target_modules.symmetric_difference(EXPECTED_TARGETS)
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                peft_model = PeftModel(model, config, adapter_name="default").eval()
                weights = load_peft_weights(str(adapter_dir), device="cuda:0")
                load_result = set_peft_model_state_dict(peft_model, weights, adapter_name="default")
            row["peft"]["warnings"] = [str(item.message) for item in caught]
            raw_missing = list(load_result.missing_keys)
            row["peft"]["raw_loader_missing_keys"] = raw_missing
            row["peft"]["expected_base_keys_absent_from_adapter_state"] = [
                key for key in raw_missing if "lora_" not in key
            ]
            row["peft"]["missing_keys"] = [key for key in raw_missing if "lora_" in key]
            row["peft"]["unexpected_keys"] = list(load_result.unexpected_keys)
            row["peft"]["lora_parameter_count"] = sum(
                parameter.numel()
                for name, parameter in peft_model.named_parameters()
                if "lora_" in name
            )
            row["peft"]["load_success"] = True

            with torch.inference_mode():
                adapter_logits = peft_model(**inputs).logits[:, -1, :].float().cpu()
                generated = peft_model.generate(**inputs, **generation_kwargs(tokenizer))
            completion = decode_completion(tokenizer, inputs["input_ids"], generated)
            row["inference"] = generation_record(completion, adapter_logits)
            row["logit_difference"] = logit_difference(base_logits, adapter_logits)
            row["cuda_memory"]["after_adapter_inference"] = cuda_memory()

            model = peft_model.unload().eval()
            del peft_model, weights, adapter_logits, generated
            torch.cuda.empty_cache()
            row["cuda_memory"]["after_adapter_unload_and_empty_cache"] = cuda_memory()
        except Exception as exc:
            row["peft"]["load_success"] = False
            row["peft"]["error"] = format_exception(exc)
            row["warnings"].append("PEFT loading or inference failed")
            if "peft_model" in locals():
                try:
                    model = peft_model.unload().eval()
                except Exception:
                    pass
            torch.cuda.empty_cache()
        write_reports(report, output_dir)

    with torch.inference_mode():
        final_base_logits = model(**inputs).logits[:, -1, :].float().cpu()
    contamination = logit_difference(base_logits, final_base_logits)
    contamination["contamination_warning"] = contamination["nonzero"]
    report["base_contamination_check"] = contamination
    del final_base_logits
    finish(report, output_dir, started)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters-dir", type=Path, default=Path("/workspace/adapters"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/adapter_verification"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace/.hf-cache/hub"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    return parser.parse_args()


def discover_adapters(root: Path) -> dict[int, Path]:
    found = {}
    if not root.is_dir():
        return found
    for path in root.iterdir():
        match = re.fullmatch(r"ptype_(\d+)", path.name)
        if path.is_dir() and match:
            found[int(match.group(1))] = path
    return found


def inspect_safetensors(adapter_dir: Path) -> dict:
    path = adapter_dir / "adapter_model.safetensors"
    stats_by_target = defaultdict(lambda: {"tensor_count": 0, "parameter_count": 0, "l2_sq": 0.0})
    stats_by_layer = defaultdict(lambda: {"tensor_count": 0, "parameter_count": 0, "l2_sq": 0.0})
    signature = []
    all_finite = True
    all_zero_names = []
    total_parameters = 0
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank = int(adapter_config["r"])
    lora_alpha = float(adapter_config["lora_alpha"])
    scale = lora_alpha / rank

    with safe_open(path, framework="pt", device="cpu") as handle:
        names = sorted(handle.keys())
        for name in names:
            tensor = handle.get_tensor(name)
            signature.append([name, list(tensor.shape), str(tensor.dtype)])
            finite = bool(torch.isfinite(tensor).all().item())
            nonzero = bool(torch.count_nonzero(tensor).item())
            all_finite = all_finite and finite
            if not nonzero:
                all_zero_names.append(name)
            count = tensor.numel()
            norm = float(torch.linalg.vector_norm(tensor.float()).item())
            total_parameters += count
            target = target_from_name(name)
            layer = layer_from_name(name)
            for bucket, key in ((stats_by_target, target), (stats_by_layer, layer)):
                bucket[key]["tensor_count"] += 1
                bucket[key]["parameter_count"] += count
                bucket[key]["l2_sq"] += norm * norm
        effective_updates = effective_update_stats(handle, names, scale, rank, lora_alpha)

    for bucket in (stats_by_target, stats_by_layer):
        for value in bucket.values():
            value["l2_norm"] = math.sqrt(value.pop("l2_sq"))

    lora_names = [name for name, _, _ in signature if ".lora_A." in name or ".lora_B." in name]
    targets = {target_from_name(name) for name in lora_names}
    return {
        "file": str(path),
        "file_size_bytes": path.stat().st_size,
        "readable": True,
        "tensor_count": len(signature),
        "total_parameters": total_parameters,
        "all_tensors_finite": all_finite,
        "expected_lora_tensors_present": bool(lora_names) and targets == EXPECTED_TARGETS,
        "present_target_modules": sorted(targets),
        "uniformly_zero_tensor_count": len(all_zero_names),
        "uniformly_zero_tensor_names": all_zero_names,
        "adapter_uniformly_zero": len(all_zero_names) == len(signature),
        "tensor_signature": signature,
        "stats_by_target_module": dict(sorted(stats_by_target.items())),
        "stats_by_layer": dict(sorted(stats_by_layer.items(), key=lambda item: int(item[0]))),
        "effective_updates": effective_updates,
    }


def effective_update_stats(handle, names: list[str], scale: float, rank: int, lora_alpha: float) -> list[dict]:
    records = []
    a_names = sorted(name for name in names if ".lora_A." in name)
    for a_name in a_names:
        b_name = a_name.replace(".lora_A.", ".lora_B.")
        if b_name not in names:
            raise ValueError(f"Missing LoRA B matrix paired with {a_name}")
        a = handle.get_tensor(a_name).double()
        b = handle.get_tensor(b_name).double()
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(f"Incompatible LoRA pair: {a_name} {tuple(a.shape)}, {b_name} {tuple(b.shape)}")

        gram_a = a @ a.T
        gram_b = b.T @ b
        raw_frobenius_sq = torch.trace(gram_b @ gram_a).clamp_min(0)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram_a)
        sqrt_gram_a = (eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)) @ eigenvectors.T
        spectral_sq = torch.linalg.eigvalsh(sqrt_gram_a @ gram_b @ sqrt_gram_a).max().clamp_min(0)
        raw_frobenius = float(raw_frobenius_sq.sqrt().item())
        raw_spectral = float(spectral_sq.sqrt().item())
        delta_frobenius = abs(scale) * raw_frobenius
        delta_spectral = abs(scale) * raw_spectral
        records.append({
            "layer": int(layer_from_name(a_name)),
            "target_module": target_from_name(a_name),
            "a_tensor_name": a_name,
            "b_tensor_name": b_name,
            "rank": rank,
            "lora_alpha": lora_alpha,
            "scale_lora_alpha_over_rank": scale,
            "matrix_shape": [int(b.shape[0]), int(a.shape[1])],
            "a_norm": float(torch.linalg.vector_norm(a).item()),
            "b_norm": float(torch.linalg.vector_norm(b).item()),
            "unscaled_b_at_a_frobenius_norm": raw_frobenius,
            "unscaled_b_at_a_spectral_norm": raw_spectral,
            "frobenius_norm": delta_frobenius,
            "spectral_norm": delta_spectral,
            "effective_update_norm": delta_frobenius,
        })
    return sorted(records, key=lambda item: (item["layer"], item["target_module"]))


def tokenizer_file_check(adapter_dir: Path, snapshot: Path) -> dict:
    result = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        adapter_file = adapter_dir / name
        base_file = snapshot / name
        result[name] = {
            "adapter_sha256": sha256(adapter_file),
            "base_sha256": sha256(base_file),
            "matches_base": sha256(adapter_file) == sha256(base_file),
        }
    result["all_files_match_base"] = all(item["matches_base"] for item in result.values())
    return result


def semantic_tokenizer_check(base_tokenizer, adapter_tokenizer) -> dict:
    probes = [
        PROMPT,
        "Hello, world!",
        "OCEAN: openness and conscientiousness.",
        "Unicode café — test.",
    ]
    encodings = [
        {
            "text": text,
            "base_ids": base_tokenizer.encode(text),
            "adapter_ids": adapter_tokenizer.encode(text),
        }
        for text in probes
    ]
    vocab_equal = base_tokenizer.get_vocab() == adapter_tokenizer.get_vocab()
    encodings_equal = all(item["base_ids"] == item["adapter_ids"] for item in encodings)
    return {
        "inference_tokenizer_source": "/workspace/adapters/ptype_0",
        "vocabulary_equal": vocab_equal,
        "vocabulary_size_base": len(base_tokenizer.get_vocab()),
        "vocabulary_size_adapter": len(adapter_tokenizer.get_vocab()),
        "probe_encodings_equal": encodings_equal,
        "probe_encodings": encodings,
        "base_special_tokens": base_tokenizer.special_tokens_map,
        "adapter_special_tokens": adapter_tokenizer.special_tokens_map,
        "difference_explained": "Adapter tokenizer adds EOS as pad token; vocabulary and probe encodings are identical.",
        "semantically_matches_base": vocab_equal and encodings_equal,
    }


def resolve_revision(cache_dir: Path, model_id: str) -> str:
    model_cache = cache_dir / f"models--{model_id.replace('/', '--')}"
    return (model_cache / "refs" / "main").read_text().strip()


def snapshot_path(cache_dir: Path, model_id: str, revision: str) -> Path:
    path = cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def environment_info(revision: str, snapshot: Path) -> dict:
    packages = ["torch", "transformers", "peft", "accelerate", "safetensors", "huggingface_hub", "pandas", "numpy"]
    gpu = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_model": gpu.name,
        "gpu_total_memory_bytes": gpu.total_memory,
        "nvidia_driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).strip(),
        "base_model_id": BASE_MODEL,
        "resolved_base_model_revision": revision,
        "base_model_snapshot": str(snapshot),
        "torch_dtype": "bfloat16",
    }


def generation_kwargs(tokenizer) -> dict:
    return {
        **GENERATION_CONFIG,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def decode_completion(tokenizer, input_ids: torch.Tensor, generated: torch.Tensor) -> str:
    new_tokens = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generation_record(completion: str, logits: torch.Tensor) -> dict:
    tokens = completion.split()
    repetition = 0.0 if not tokens else 1.0 - len(set(tokens)) / len(tokens)
    return {
        "completion": completion,
        "empty": not completion.strip(),
        "repetitive": len(tokens) >= 8 and repetition > 0.5,
        "word_repetition_ratio": repetition,
        "corrupted": "�" in completion or "\x00" in completion,
        "numerically_stable": bool(torch.isfinite(logits).all().item()),
    }


def logit_difference(base: torch.Tensor, adapter: torch.Tensor) -> dict:
    difference = adapter - base
    return {
        "maximum_absolute": float(difference.abs().max().item()),
        "mean_absolute": float(difference.abs().mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(difference).item()),
        "nonzero": bool(torch.count_nonzero(difference).item()),
        "compared_position": "last input token",
    }


def new_row(ptype: int, path: Path | None) -> dict:
    return {
        "ptype": ptype,
        "adapter": f"ptype_{ptype}",
        "path": str(path) if path else None,
        "structural": {},
        "tokenizer": {},
        "peft": {
            "load_success": False,
            "missing_keys": [],
            "unexpected_keys": [],
            "target_module_mismatches": [],
            "warnings": [],
        },
        "inference": {},
        "logit_difference": {},
        "cuda_memory": {
            "before_adapter_attachment": {},
            "after_adapter_inference": {},
            "after_adapter_unload_and_empty_cache": {},
        },
        "warnings": [],
    }


def write_reports(report: dict, output_dir: Path) -> None:
    json_path = output_dir / "adapter_verification.json"
    csv_path = output_dir / "adapter_verification.csv"
    md_path = output_dir / "adapter_verification.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    fields = [
        "ptype", "adapter", "structural_readable", "all_tensors_finite",
        "expected_lora_tensors_present", "names_shapes_consistent",
        "uniformly_zero_tensor_count", "total_parameters", "tokenizer_matches_base",
        "peft_load_success", "missing_key_count", "unexpected_key_count",
        "target_module_mismatch_count", "lora_parameter_count", "completion",
        "empty", "repetitive", "corrupted", "numerically_stable",
        "max_abs_logit_difference", "mean_abs_logit_difference",
        "l2_logit_difference", "logit_difference_nonzero", "warnings",
        "cuda_allocated_before", "cuda_reserved_before",
        "cuda_allocated_after_inference", "cuda_reserved_after_inference",
        "cuda_allocated_after_unload", "cuda_reserved_after_unload",
        "effective_updates_json", "base_contamination_max_abs",
        "base_contamination_mean_abs", "base_contamination_l2",
        "base_contamination_warning",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["adapters"]:
            structural, peft = row["structural"], row["peft"]
            inference, difference = row["inference"], row["logit_difference"]
            memory = row["cuda_memory"]
            contamination = report.get("base_contamination_check", {})
            writer.writerow({
                "ptype": row["ptype"], "adapter": row["adapter"],
                "structural_readable": structural.get("readable"),
                "all_tensors_finite": structural.get("all_tensors_finite"),
                "expected_lora_tensors_present": structural.get("expected_lora_tensors_present"),
                "names_shapes_consistent": structural.get("names_shapes_consistent"),
                "uniformly_zero_tensor_count": structural.get("uniformly_zero_tensor_count"),
                "total_parameters": structural.get("total_parameters"),
                "tokenizer_matches_base": row["tokenizer"].get("semantically_matches_base"),
                "peft_load_success": peft.get("load_success"),
                "missing_key_count": len(peft.get("missing_keys", [])),
                "unexpected_key_count": len(peft.get("unexpected_keys", [])),
                "target_module_mismatch_count": len(peft.get("target_module_mismatches", [])),
                "lora_parameter_count": peft.get("lora_parameter_count"),
                "completion": inference.get("completion"), "empty": inference.get("empty"),
                "repetitive": inference.get("repetitive"), "corrupted": inference.get("corrupted"),
                "numerically_stable": inference.get("numerically_stable"),
                "max_abs_logit_difference": difference.get("maximum_absolute"),
                "mean_abs_logit_difference": difference.get("mean_absolute"),
                "l2_logit_difference": difference.get("l2_norm"),
                "logit_difference_nonzero": difference.get("nonzero"),
                "warnings": " | ".join(row["warnings"] + peft.get("warnings", [])),
                "cuda_allocated_before": memory["before_adapter_attachment"].get("allocated_bytes"),
                "cuda_reserved_before": memory["before_adapter_attachment"].get("reserved_bytes"),
                "cuda_allocated_after_inference": memory["after_adapter_inference"].get("allocated_bytes"),
                "cuda_reserved_after_inference": memory["after_adapter_inference"].get("reserved_bytes"),
                "cuda_allocated_after_unload": memory["after_adapter_unload_and_empty_cache"].get("allocated_bytes"),
                "cuda_reserved_after_unload": memory["after_adapter_unload_and_empty_cache"].get("reserved_bytes"),
                "effective_updates_json": json.dumps(structural.get("effective_updates", []), separators=(",", ":")),
                "base_contamination_max_abs": contamination.get("maximum_absolute"),
                "base_contamination_mean_abs": contamination.get("mean_absolute"),
                "base_contamination_l2": contamination.get("l2_norm"),
                "base_contamination_warning": contamination.get("contamination_warning"),
            })

    lines = [
        "# OCEAN LoRA adapter verification", "",
        "> Functional artifact verification only; this report does not establish personality validity.", "",
        f"- Discovery exact: **{report['discovery_exact']}**",
        f"- Base model: `{report['environment']['base_model_id']}`",
        f"- Resolved revision: `{report['environment']['resolved_base_model_revision']}`",
        f"- GPU: {report['environment']['gpu_model']}",
        f"- Prompt: `{report['prompt']}`", "",
        f"- Base contamination warning: **{report.get('base_contamination_check', {}).get('contamination_warning', 'pending')}**",
        f"- Final-vs-initial base max absolute logit difference: `{report.get('base_contamination_check', {}).get('maximum_absolute', 'pending')}`", "",
        "| Adapter | Structural | PEFT load | Finite | Nonzero effect | Max abs logit Δ | Completion flags |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["adapters"]:
        structural, peft = row["structural"], row["peft"]
        inference, difference = row["inference"], row["logit_difference"]
        flags = [key for key in ("empty", "repetitive", "corrupted") if inference.get(key)]
        lines.append(
            f"| {row['adapter']} | {structural.get('readable', '')} | {peft.get('load_success', '')} | "
            f"{structural.get('all_tensors_finite', '')} | {difference.get('nonzero', '')} | "
            f"{difference.get('maximum_absolute', '')} | {', '.join(flags) or 'none'} |"
        )
    lines.extend(["", "## Base completion", "", "```text", report.get("base_inference", {}).get("completion", ""), "```", ""])
    for row in report["adapters"]:
        lines.extend([
            f"## {row['adapter']}", "",
            f"- Structural error: `{row['structural'].get('error')}`" if row["structural"].get("error") else "- Structural validation completed.",
            f"- Missing keys: {len(row['peft'].get('missing_keys', []))}",
            f"- Unexpected keys: {len(row['peft'].get('unexpected_keys', []))}",
            f"- Target mismatches: `{row['peft'].get('target_module_mismatches', [])}`", "",
            "```text", row["inference"].get("completion", ""), "```", "",
        ])
    md_path.write_text("\n".join(lines))


def finish(report: dict, output_dir: Path, started: float) -> None:
    report["finished_utc"] = utc_now()
    report["elapsed_seconds"] = time.time() - started
    write_reports(report, output_dir)


def target_from_name(name: str) -> str:
    for target in EXPECTED_TARGETS:
        if f".{target}." in name:
            return target
    return "unknown"


def layer_from_name(name: str) -> str:
    match = LAYER_RE.search(name)
    return match.group(1) if match else "-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def cuda_memory() -> dict:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def format_exception(exc: Exception) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


if __name__ == "__main__":
    main()
