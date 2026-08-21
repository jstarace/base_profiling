"""Atomic activation shards, checksums, resume validation, and shard indexing."""
from __future__ import annotations

import hashlib,json,os,tempfile
from pathlib import Path

import numpy as np


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def atomic_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fd,temp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp,path)
    except BaseException:
        try: os.unlink(temp)
        except FileNotFoundError: pass
        raise


def atomic_npz(path,**arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fd,temp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as handle: np.savez(handle,**arrays); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp,path)
    except BaseException:
        try: os.unlink(temp)
        except FileNotFoundError: pass
        raise


def shard_paths(root,model_key,group,pooling):
    stem=Path(root)/"shards"/model_key/group/pooling
    return stem.with_suffix(".npz"),stem.with_suffix(".metadata.json")


def validate_completed_shard(data_path,metadata_path,expected):
    if not data_path.exists() and not metadata_path.exists(): return False
    if not data_path.exists() or not metadata_path.exists(): raise ValueError("partial shard exists")
    metadata=json.loads(metadata_path.read_text())
    for key,value in expected.items():
        if metadata.get(key)!=value: raise ValueError(f"resume metadata mismatch {key}")
    if metadata.get("data_sha256")!=sha256(data_path): raise ValueError("resume shard checksum mismatch")
    with np.load(data_path) as shard:
        if shard["raw"].shape!=tuple(metadata["array_shape"]) or shard["delta"].shape!=tuple(metadata["array_shape"]): raise ValueError("resume shape mismatch")
    return True

