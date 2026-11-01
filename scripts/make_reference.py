"""Reference logits from Hugging Face transformers on the CPU, for tests/test_logits.py.

    make reference   # .venv-ref with torch + transformers, writes tests/data/reference.npz
    (runs as: .venv-ref/bin/python -m scripts.make_reference)

The model is loaded in fp16 (the same rounding of the bf16 checkpoint as mini_serve.engine)
and run in fp32 with eager attention. Stored per prompt: token ids, the full logits of the
last position, the top-K logits of every position, and a 16-token greedy continuation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tokenizers
import torch
import transformers
from transformers import AutoModelForCausalLM

from mini_serve.engine import DEFAULT_WEIGHTS as WEIGHTS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "data" / "reference.npz"
TOPK, GEN = 20, 16

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "In a paged KV cache, the attention keys and values of each sequence are stored in fixed-size blocks",
    "1, 2, 3, 5, 8, 13,",
    "Once upon a time, in a small village by the sea, there lived an old fisherman who",
]


def main():
    torch.manual_seed(0)
    tok = tokenizers.Tokenizer.from_file(str(WEIGHTS / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(WEIGHTS, dtype=torch.float16, attn_implementation="eager")
    model = model.float().eval()
    data = {}
    with torch.no_grad():
        for i, text in enumerate(PROMPTS):
            ids = tok.encode(text).ids
            logits = model(torch.tensor([ids])).logits[0].numpy()        # [L, V]
            top = np.argsort(-logits, axis=1)[:, :TOPK]
            seq = list(ids)
            margins = []
            for _ in range(GEN):
                l = model(torch.tensor([seq])).logits[0, -1].numpy()
                a, b = np.sort(l)[-2:][::-1]
                margins.append(a - b)
                seq.append(int(l.argmax()))
            data[f"ids{i}"] = np.array(ids, dtype=np.int32)
            data[f"last{i}"] = logits[-1]
            data[f"topi{i}"] = top.astype(np.int32)
            data[f"topv{i}"] = np.take_along_axis(logits, top, axis=1)
            data[f"gen{i}"] = np.array(seq[len(ids):], dtype=np.int32)
            data[f"gen_margin{i}"] = np.array(margins, dtype=np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **data)
    meta = {"torch": torch.__version__, "transformers": transformers.__version__,
            "python": sys.version.split()[0], "prompts": PROMPTS}
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT}: {meta}")


if __name__ == "__main__":
    main()
