import json

import numpy as np
import pytest

from full_profile_drift.capture import valid_shard
from full_profile_drift.io import atomic_json,atomic_npz,sha256


def test_atomic_npz_and_resume_validation(tmp_path):
    data=tmp_path/"x.npz"; metadata=tmp_path/"x.metadata.json"; raw=np.ones((2,3,4),dtype="float32")
    atomic_npz(data,raw=raw,delta=np.zeros_like(raw)); payload={"model_key":"base","array_shape":[2,3,4],"data_sha256":sha256(data)}; atomic_json(metadata,payload)
    assert valid_shard(data,metadata,{"model_key":"base"})
    with pytest.raises(ValueError): valid_shard(data,metadata,{"model_key":"ptype_0"})


def test_atomic_json_preserves_completed_file_on_interruption(tmp_path,monkeypatch):
    path=tmp_path/"x.json"; atomic_json(path,{"complete":True})
    import full_profile_drift.io as io
    monkeypatch.setattr(json,"dump",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("interrupt")))
    with pytest.raises(RuntimeError): io.atomic_json(path,{"complete":False})
    assert json.loads(path.read_text())=={"complete":True} and not list(tmp_path.glob("*.tmp"))


def test_delta_is_float32_finite_and_exact():
    base=np.arange(24,dtype="float32").reshape(2,3,4); adapter=base+0.25; delta=(adapter-base).astype("float32")
    assert delta.dtype==np.float32 and np.isfinite(delta).all() and np.array_equal(delta,adapter-base)
