import json
from collections import Counter

import numpy as np

from full_profile_drift.analyze import factorial_design


def test_walsh_hadamard_design_is_exactly_orthogonal():
    design,terms=factorial_design(5)
    assert design.shape==(32,32) and len(terms)==32
    assert np.array_equal(design.T@design,32*np.eye(32))


def test_prompt_manifest_frozen_coverage():
    manifest=json.load(open("prompt_manifest/prompt_manifest.json"))
    records=manifest["records"]
    assert len(records)==len({r["prompt_id"] for r in records})==len({r["text"] for r in records})==1080
    assert Counter(r["group"] for r in records)=={"naturalistic_behavioral":720,"neutral_controls":240,"ipip_stems":120}
    natural=Counter(r["category"] for r in records if r["group"]=="naturalistic_behavioral")
    assert len(natural)==12 and set(natural.values())=={60}
    assert sum(r["legacy_core"] for r in records)==360


def test_exact_low_rank_inner_product_identity():
    rng=np.random.default_rng(7)
    ai=rng.normal(size=(3,5)); bi=rng.normal(size=(7,3)); aj=rng.normal(size=(3,5)); bj=rng.normal(size=(7,3))
    low_rank=np.sum((bi.T@bj)*(ai@aj.T))
    dense=np.sum((bi@ai)*(bj@aj))
    assert np.allclose(low_rank,dense,rtol=1e-12,atol=1e-12)
