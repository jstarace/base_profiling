"""Train one LoRA adapter per OCEAN ptype class for a single partition group.

Self-contained for a fresh RunPod pod: pulls pandora-big5 from HuggingFace, trains every
ptype in the group sequentially, writes each adapter to $ADAPTER_OUT_DIR/ptype_<N>/, exits.
Resume-safe: a ptype whose adapter already exists on the volume is skipped.

    python training/train_ocean_adapters.py --group 0

Requires HF_TOKEN in the environment: the base checkpoint is a gated repo, and so the
dataset pull and the model pull both authenticate with it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import time
from datetime import timedelta

import paths  # noqa: F401  (loads .env for local runs; pods use real env vars)

BASE_MODEL = "meta-llama/Llama-3.1-8B"
DATASET = "jingjietan/pandora-big5"
PARTITION = Path(__file__).resolve().parent / "ocean_partition.json"
DEFAULT_OUT_DIR = "/workspace/adapters"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ADAPTER_WEIGHTS = "adapter_model.safetensors"


def main():
    args = parse_args()
    out_root = Path(os.environ.get("ADAPTER_OUT_DIR", DEFAULT_OUT_DIR))
    out_root.mkdir(parents=True, exist_ok=True)

    ptypes = load_group(args.partition, args.group)
    pending = [p for p in ptypes if not is_done(out_root, p)]
    print(f"[group {args.group}] {len(ptypes)} classes: {ptypes}")
    print(f"[group {args.group}] {len(ptypes) - len(pending)} already on volume, {len(pending)} to train")
    print(f"[group {args.group}] adapters -> {out_root}")
    if not pending:
        print(f"[group {args.group}] nothing to do")
        return

    ds = load_pandora(args.dataset, args.limit)
    tokenizer = load_tokenizer(args.base_model)

    group_start = time.time()
    for i, ptype in enumerate(pending, 1):
        # The verified upstream integer is the complete class identity. Training
        # never reconstructs it from dataframe column order. The same integer is
        # preserved in the directory name so ptype N always maps to ptype_N.
        out_dir = out_root / f"ptype_{ptype}"
        rows = ds.filter(lambda ex: ex["ptype"] == ptype, num_proc=args.num_proc)
        start = time.time()
        print(f"\n[ptype {ptype}] ({i}/{len(pending)}) start {stamp()} | {len(rows):,} rows")

        if args.dry_run:
            print(f"[ptype {ptype}] dry-run, no training")
            continue

        train_adapter(rows, tokenizer, out_dir, args)
        elapsed = time.time() - start
        print(f"[ptype {ptype}] end {stamp()} | elapsed {timedelta(seconds=int(elapsed))} "
              f"| saved -> {out_dir}")

    print(f"\n[group {args.group}] done | total {timedelta(seconds=int(time.time() - group_start))}")


def train_adapter(rows, tokenizer, out_dir, args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_len)

    tokenized = rows.map(tokenize, batched=True, num_proc=args.num_proc,
                         remove_columns=rows.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        **quantization_kwargs(args),
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "_checkpoints"),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            max_grad_norm=1.0,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=args.logging_steps,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    del trainer, model
    torch.cuda.empty_cache()


def quantization_kwargs(args):
    if not args.load_4bit:
        return {}
    import torch
    from transformers import BitsAndBytesConfig
    return {"quantization_config": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )}


def load_pandora(name, limit):
    from datasets import concatenate_datasets, load_dataset
    ds = load_dataset(name, token=os.environ.get("HF_TOKEN"))
    # The adapters were trained on every published split as one corpus. Original
    # train, validation, and test boundaries are intentionally not retained.
    merged = concatenate_datasets(list(ds.values()))
    if limit:
        merged = merged.select(range(min(limit, len(merged))))
    print(f"[data] {name}: {len(merged):,} rows")
    return merged


def load_tokenizer(base_model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # Llama base ships no pad token
    return tok


def load_group(partition_path, group):
    payload = json.loads(Path(partition_path).read_text())
    groups = payload["groups"]
    if not 0 <= group < len(groups):
        raise SystemExit(f"group {group} out of range: partition has {len(groups)} groups")
    return groups[group]["ptypes"]


def is_done(out_root, ptype):
    # Resume uses the same immutable ptype-to-directory contract as training.
    return (out_root / f"ptype_{ptype}" / ADAPTER_WEIGHTS).exists()


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", type=int, required=True, help="partition group index")
    p.add_argument("--partition", type=Path, default=PARTITION)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="cap optimizer steps per class; -1 uses --epochs")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-4bit", action="store_true", help="QLoRA; needs bitsandbytes")
    p.add_argument("--limit", type=int, default=0, help="debug: cap rows loaded before filtering")
    p.add_argument("--dry-run", action="store_true", help="resolve group and row counts, no training")
    return p.parse_args()


if __name__ == "__main__":
    main()
