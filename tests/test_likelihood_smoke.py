"""Smoke test that the model-loading API resolves real Hugging Face paths.

Skipped unless ``transformers`` is installed (``pip install plasmidrl[gen]``).
A full forward pass is skipped when no GPU is available, since the 124M
PlasmidGPT checkpoints are large enough that CPU-only loading takes long
enough to risk false-positive CI timeouts.
"""
import importlib.util

import pytest

from plasmidrl import models


def test_known_model_paths():
    assert models.path("Base") == "UCL-CSSB/PlasmidGPT"
    assert models.path("SFT") == "UCL-CSSB/PlasmidGPT-SFT"
    assert models.path("RL") == "UCL-CSSB/PlasmidGPT-GRPO"


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        models.path("Foobar")


def test_optimal_temperatures_set():
    assert models.OPTIMAL_T == {"Base": 1.00, "SFT": 1.00, "RL": 1.15}


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="transformers not installed (use plasmidrl[gen])")
def test_load_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no GPU; skipping full forward-pass smoke")
    model, tok = models.load("RL")
    seq = "ATGAAACGTGGTTTAGCATGCATGTAA"
    ids = tok(seq, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids)
    assert out.logits.shape[1] == ids["input_ids"].shape[1]
