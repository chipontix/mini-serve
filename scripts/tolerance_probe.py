"""Why the logit tolerances in tests/test_logits.py are what they are, measured from both sides.

    .venv/bin/python -m scripts.tolerance_probe   # writes results/tolerance-probe.txt

Prints the max |engine - transformers| over the reference's top-20 logits of every prompt
position for the correct engine (fp32 and fp16 KV), and for two small, plausible bugs injected
through the config: RMSNorm eps 1e-6 instead of 1e-5, and RoPE theta off by 1%. The tolerance
must sit above the first and below the second.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from mini_serve.engine import DEFAULT_WEIGHTS, ROOT, Runner

REF = np.load(ROOT / "tests" / "data" / "reference.npz")


def max_err(runner):
    worst = scale = 0.0
    for i in range(5):
        ids = REF[f"ids{i}"].tolist()
        L = len(ids)
        t = list(range(-(-L // 16)))
        logits = runner.forward(ids, list(range(L)), t * L, len(t), list(range(L)))
        topi, topv = REF[f"topi{i}"], REF[f"topv{i}"]
        worst = max(worst, float(np.abs(np.take_along_axis(logits, topi, 1) - topv).max()))
        scale = max(scale, float(np.abs(topv).max()))
    return worst, scale


def with_config(**change):
    d = Path(tempfile.mkdtemp())
    cfg = json.loads((DEFAULT_WEIGHTS / "config.json").read_text())
    cfg.update(change)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "model.safetensors").symlink_to(DEFAULT_WEIGHTS / "model.safetensors")
    return d


def main():
    lines = []
    for label, kw in [("correct, fp32 KV", dict(kv_f16=False)),
                      ("correct, fp16 KV", dict(kv_f16=True)),
                      ("bug: rms_norm_eps 1e-6, fp32 KV", dict(kv_f16=False, weights=with_config(rms_norm_eps=1e-6))),
                      ("bug: rope_theta +1%, fp32 KV", dict(kv_f16=False, weights=with_config(rope_theta=101000.0)))]:
        err, scale = max_err(Runner(64, 16, **kw))
        lines.append(f"{label:34} max |logit error| {err:.2e}   (largest |logit| {scale:.1f})")
        print(lines[-1], flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "tolerance-probe.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
