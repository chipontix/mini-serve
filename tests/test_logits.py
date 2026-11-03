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
TOL = {"f32": 5e-3, "f16": 3e-2}
# Chunked vs one-chunk prefill, both with an fp32 KV cache. The same K/V reach the kernel either
# way; Accelerate's GEMMs sum in a different order for a different row count (observed ~1e-4).
# The smallest greedy top-2 margin on these prompts is 3e-2, so outputs must match exactly too.
CHUNK_TOL = 1e-3


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

    def test_chunked_prefill_matches_one_chunk(self):
        """The 5 prompts through the scheduler with chunked prefill (7 tokens per step: chunks
        that cut pages and prompts mid-way, mixed with other sequences' decodes) against the same
        prompts prefilled in one chunk: the logits of every sampled token and the greedy outputs."""
        results = {}
        for chunk in (None, 7):
            r = runner("f32")
            logits = {}
            run = r.run

            def recording_run(tokens, positions, tables, max_blocks, out_rows, seqs):
                out = run(tokens, positions, tables, max_blocks, out_rows, seqs)
                rows = r.last_logits
                for i, (seq, row) in enumerate(zip(seqs, out_rows)):
                    if positions[row] == len(seq.tokens) - 1:   # the sampled row, not a partial chunk
                        logits.setdefault(seq.id, []).append(rows[i])
                return out

            r.run = recording_run
            llm = LLM(r, Scheduler(BlockPool(64, PAGE), chunk_size=chunk))
            seqs = [Sequence(REF[f"ids{i}"].tolist(), max_tokens=8, id=i) for i in range(N_PROMPTS)]
            outs = llm.generate(seqs)
            results[chunk] = outs, {i: np.stack(v) for i, v in logits.items()}
            if chunk:
                self.assertGreater(llm.stats["prefill_steps"], 3)
        (outs1, logits1), (outsc, logitsc) = results[None], results[7]
        self.assertEqual(outsc, outs1)
        err = max(np.abs(logitsc[i] - logits1[i]).max() for i in range(N_PROMPTS))
        self.worst[("f32", "chunked vs one chunk")] = float(err)
        self.assertLessEqual(err, CHUNK_TOL)

    @classmethod
    def tearDownClass(cls):
        for (kv, label), err in sorted(cls.worst.items()):
            tol = CHUNK_TOL if label.startswith("chunked") else TOL[kv]
            print(f"  max |logit error| kv={kv} {label}: {err:.2e} (tolerance {tol:.0e})")


if __name__ == "__main__":
    unittest.main()
