"""Injects known bugs into the block pool and the scheduler, one at a time, and checks that
tests/test_scheduler.py (or the test named with the mutation) fails for each. Writes
results/mutation.txt.

    .venv/bin/python scripts/mutation.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGITS_CHUNKED = "tests.test_logits.LogitsTest.test_chunked_prefill_matches_one_chunk"  # needs GPU + weights
MUTATIONS = [  # (file, original, replacement, what the bug is[, test to run])
    ("mini_serve/scheduler.py", "[:(len(tokens) - 1) // P]", "[:len(tokens) // P]",
     "fully cached prompt: no token left to compute"),
    ("mini_serve/scheduler.py", "if self.pool.num_free():", "if True:",
     "decode allocates without checking the pool, never preempts"),
    ("mini_serve/scheduler.py", "seq.num_computed = len(hit) * P\n", "seq.num_computed = len(hit) * P + 1\n",
     "off by one after a prefix hit"),
    ("mini_serve/scheduler.py", "victim = self.running[-1]", "victim = self.running[0]",
     "preempts the oldest sequence instead of the newest"),
    ("mini_serve/scheduler.py", "        self.running.remove(seq)\n        self._free(seq)\n        seq.preemptions",
     "        self.running.remove(seq)\n        seq.preemptions", "preemption keeps the victim's blocks"),
    ("mini_serve/scheduler.py", "hit_free = sum(1 for b in hit if pool.ref[b] == 0)", "hit_free = 0",
     "admission forgets that free cached blocks use free slots"),
    ("mini_serve/scheduler.py", "        if self.static and self.running:\n            return self._decode()\n", "",
     "static batch admits new requests while running"),
    ("mini_serve/blocks.py", "        h = self.hash_of.pop(b, None)\n        if h is not None:\n            del self.by_hash[h]\n", "",
     "reused block keeps its old cache entry (stale prefix hit)"),
    ("mini_serve/blocks.py", "            if self.ref[b] == 0:\n                del self.free[b]\n", "",
     "prefix hit on a free block leaves it on the free list"),
    ("mini_serve/blocks.py", "prev = hash((prev, tuple(", "prev = hash((None, tuple(",
     "block hash not chained to the prefix"),
    ("mini_serve/blocks.py", "self.free[b] = None  # most recently freed: evicted last", "self.free[b] = None; self.free.move_to_end(b, last=False)",
     "free list evicts the most recently freed block first"),
    ("mini_serve/scheduler.py", "            self.running.remove(seq)\n            self._free(seq)\n\n",
     "            self.running.remove(seq)\n\n", "abort of a running sequence keeps its blocks"),
    ("mini_serve/scheduler.py", "key=lambda s: left(s) > 1)", "key=lambda s: left(s) == 1)",
     "chunked step: prefill chunks before decodes"),
    ("mini_serve/scheduler.py", "            n = min(n, budget)\n", "",
     "chunked admission ignores the step budget"),
    ("mini_serve/scheduler.py", "            if not sample:", "            if False:",
     "partial prefill chunk samples a token (real model)", LOGITS_CHUNKED),
]


def run_tests(test="tests.test_scheduler"):
    r = subprocess.run([sys.executable, "-m", "unittest", test], cwd=ROOT,
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip().splitlines()[-1]


def main():
    for test in {m[4] for m in MUTATIONS if len(m) > 4} | {"tests.test_scheduler"}:
        assert run_tests(test)[0], f"{test} must pass before mutating"
    lines, missed = [], 0
    for path, old, new, what, *test in MUTATIONS:
        f = ROOT / path
        backup = f.read_text()
        assert backup.count(old) == 1, f"mutation does not apply: {what}"
        try:
            f.write_text(backup.replace(old, new))
            passed, summary = run_tests(*test)
        finally:
            f.write_text(backup)
        missed += passed
        lines.append(f"{'MISSED' if passed else 'caught'}  {what:62} {summary}")
        print(lines[-1], flush=True)
    lines.append(f"{len(MUTATIONS) - missed} of {len(MUTATIONS)} injected bugs caught")
    print(lines[-1])
    (ROOT / "results" / "mutation.txt").write_text("\n".join(lines) + "\n")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
