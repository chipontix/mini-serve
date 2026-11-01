"""The engine against transformers on the CPU (tests/data/reference.npz, scripts/make_reference.py).

Needs the weights (make weights) and the built library (make build).
"""

import unittest

import numpy as np

from mini_serve.blocks import BlockPool
from mini_serve.engine import ROOT, Runner
from mini_serve.llm import LLM
from mini_serve.scheduler import Scheduler, Sequence

REF = np.load(ROOT / "tests" / "data" / "reference.npz")
N_PROMPTS = 5
PAGE = 16
# Max |engine - reference| over the compared logits. See README, "Correctness".
TOL = {"f32": 2e-3, "f16": 5e-2}


def runner(kv):
    try:
        return Runner(num_blocks=64, page_size=PAGE, kv_f16=(kv == "f16"))
    except RuntimeError as e:
        if "no Metal device" in str(e):
            raise unittest.SkipTest("no Metal device")
        raise


def table(n_tokens, first_block=0):
    """A block table for one sequence: blocks in reverse order, so logical order != physical."""
    n = -(-n_tokens // PAGE)
    return list(range(first_block + n - 1, first_block - 1, -1))


def errors(logits, i, rows):
    """Max |diff| at the reference's top-K tokens of each position, and over the full last row."""
    topi, topv = REF[f"topi{i}"][rows], REF[f"topv{i}"][rows]
    err_top = np.abs(np.take_along_axis(logits, topi, axis=1) - topv).max()
    return err_top, topi[:, 0], topv


class LogitsTest(unittest.TestCase):
    worst = {}

    def check(self, kv, logits, i, rows, label):
        err, ref_arg, topv = errors(logits, i, rows)
        self.worst[(kv, label)] = max(self.worst.get((kv, label), 0.0), float(err))
        self.assertLessEqual(err, TOL[kv], f"prompt {i}, {label}")
        # Same argmax wherever the reference's top two are further apart than the error allows.
        clear = (topv[:, 0] - topv[:, 1]) > 2 * TOL[kv]
        np.testing.assert_array_equal(logits.argmax(1)[clear], ref_arg[clear])

    def test_prefill_all_positions(self):
        for kv in TOL:
            r = runner(kv)
            for i in range(N_PROMPTS):
                ids = REF[f"ids{i}"].tolist()
                L = len(ids)
                t = table(L)
                logits = r.forward(ids, list(range(L)), t * L, len(t), list(range(L)))
                self.check(kv, logits, i, np.arange(L), "prefill")
                last = np.abs(logits[-1] - REF[f"last{i}"]).max()
                self.assertLessEqual(last, TOL[kv], f"prompt {i}, full last row")

    def test_decode_token_by_token(self):
        """Prefill half the prompt, then one token per call: the decode path of the kernel."""
        for kv in TOL:
            r = runner(kv)
            for i in range(N_PROMPTS):
                ids = REF[f"ids{i}"].tolist()
                L, k = len(ids), len(ids) // 2
                t = table(L, first_block=10)
                rows = [r.forward(ids[:k], list(range(k)), t * k, len(t), [k - 1])[0]]
                for p in range(k, L):
                    rows.append(r.forward([ids[p]], [p], t, len(t), [0])[0])
                self.check(kv, np.stack(rows), i, np.arange(k - 1, L), "decode")

    def test_batched_prompts_and_greedy_generation(self):
        """All 5 prompts through the scheduler together (one batched prefill, batched decode):
        the 16 greedy tokens must equal transformers' greedy continuation."""
        r = runner("f16")
        llm = LLM(r, Scheduler(BlockPool(64, PAGE)))
        seqs = [Sequence(REF[f"ids{i}"].tolist(), max_tokens=16) for i in range(N_PROMPTS)]
        outs = llm.generate(seqs)
        for i, out in enumerate(outs):
            ref, margin = REF[f"gen{i}"].tolist(), REF[f"gen_margin{i}"]
            # Compare up to the first step where the reference itself was a near tie.
            n = next((j for j, m in enumerate(margin) if m <= 2 * TOL["f16"]), len(ref))
            self.assertEqual(out[:n], ref[:n], f"prompt {i}")

    @classmethod
    def tearDownClass(cls):
        for (kv, label), err in sorted(cls.worst.items()):
            print(f"  max |logit error| kv={kv} {label}: {err:.2e} (tolerance {TOL[kv]:.0e})")


if __name__ == "__main__":
    unittest.main()
