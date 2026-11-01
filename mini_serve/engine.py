"""Python side of the model runner: loads the weights, calls libminiserve through ctypes, samples."""

import ctypes
import json
import os
from array import array
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "weights" / "SmolLM2-135M"
DEFAULT_LIB = ROOT / "build" / "libminiserve.dylib"


class Config(ctypes.Structure):  # must match struct Config in src/engine.cpp
    _fields_ = [(name, ctypes.c_int) for name in
                ("n_layers", "hidden", "n_heads", "n_kv_heads", "head_dim", "inter", "vocab")] + \
               [("rope_theta", ctypes.c_float), ("rms_eps", ctypes.c_float)] + \
               [(name, ctypes.c_int) for name in ("num_blocks", "page_size", "kv_f16")]


def load_safetensors(path):
    """name -> float32 array of the weights rounded to fp16, i.e. the model in fp16. The file
    stores bf16; bf16 -> fp32 is exact, the fp32 -> fp16 rounding is the same as torch's
    `.to(torch.float16)`, so the reference (scripts/make_reference.py) sees identical values."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    header = json.loads(bytes(raw[8:8 + n]))
    out = {}
    for name, t in header.items():
        if name == "__metadata__":
            continue
        a, b = t["data_offsets"]
        buf = raw[8 + n + a:8 + n + b]
        if t["dtype"] == "BF16":
            w = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif t["dtype"] in ("F16", "F32"):
            w = np.frombuffer(buf, dtype="<f2" if t["dtype"] == "F16" else "<f4").astype(np.float32)
        else:
            raise ValueError(f"{name}: unsupported dtype {t['dtype']}")
        out[name] = w.astype(np.float16).astype(np.float32).reshape(t["shape"])
    return out


class Runner:
    """Owns the C++ engine (weights + the paged KV pool on the GPU)."""

    def __init__(self, num_blocks, page_size=16, kv_f16=True, weights=DEFAULT_WEIGHTS, lib=DEFAULT_LIB, seed=0):
        weights = Path(weights)
        cfg = json.loads((weights / "config.json").read_text())
        assert cfg["model_type"] == "llama" and cfg.get("rope_scaling") is None and cfg["tie_word_embeddings"]
        self.vocab = cfg["vocab_size"]
        self.num_blocks, self.page_size = num_blocks, page_size
        self.eos_id = cfg["eos_token_id"]
        self.lib = ctypes.CDLL(os.fspath(lib))
        L = self.lib
        L.ms_create.restype = ctypes.c_void_p
        L.ms_create.argtypes = [ctypes.POINTER(Config)]
        L.ms_destroy.argtypes = [ctypes.c_void_p]
        L.ms_last_error.restype = ctypes.c_char_p
        L.ms_set_weight.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int64]
        L.ms_forward.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 3 + \
                                [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        c = Config(cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"],
                   cfg["num_key_value_heads"], cfg["hidden_size"] // cfg["num_attention_heads"],
                   cfg["intermediate_size"], cfg["vocab_size"], cfg["rope_theta"], cfg["rms_norm_eps"],
                   num_blocks, page_size, int(kv_f16))
        self.h = L.ms_create(ctypes.byref(c))
        if not self.h:
            raise RuntimeError(L.ms_last_error().decode())
        for name, w in load_safetensors(weights / "model.safetensors").items():
            w = np.ascontiguousarray(w)
            self._check(L.ms_set_weight(self.h, name.encode(), w.ctypes.data, w.size))
        self.rng = np.random.default_rng(seed)

    def _check(self, rc):
        if rc != 0:
            raise RuntimeError(self.lib.ms_last_error().decode())

    def __del__(self):
        if getattr(self, "h", None):
            self.lib.ms_destroy(self.h)
            self.h = None

    def forward(self, tokens, positions, tables, max_blocks, out_rows):
        """Logits [len(out_rows), vocab]; see ms_forward in src/engine.cpp."""
        args = [array("i", x) for x in (tokens, positions, tables, out_rows)]
        logits = np.empty((len(out_rows), self.vocab), dtype=np.float32)
        ptr = [a.buffer_info()[0] for a in args]
        self._check(self.lib.ms_forward(self.h, len(tokens), ptr[0], ptr[1], ptr[2], max_blocks,
                                        len(out_rows), ptr[3], logits.ctypes.data))
        return logits

    def run(self, tokens, positions, tables, max_blocks, out_rows, seqs):
        logits = self.forward(tokens, positions, tables, max_blocks, out_rows)
        out = []
        for row, seq in zip(logits, seqs):
            if seq.temperature <= 0:
                out.append(int(row.argmax()))
            else:
                p = np.exp((row - row.max()) / seq.temperature)
                out.append(int(self.rng.choice(len(p), p=p / p.sum())))
        return out
