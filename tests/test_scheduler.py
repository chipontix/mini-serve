"""Block pool and scheduler tests. No model, no GPU: a fake runner stands in for the engine.

The fake runner keeps its own KV pool of (token, position) per slot and checks, for every row
it computes, what the real attention would read: the slots of positions 0..pos through the
block table must hold exactly that sequence's tokens. Its "model" then returns a token that
depends on the whole history it read. So a wrong block table, a slot overwritten by another
sequence, a stale shared block or a lost token after preemption all show up, either as an
assertion in the runner or as an output different from running the request alone.
"""

import random
import unittest

from mini_serve.blocks import BlockPool, block_hashes
from mini_serve.llm import LLM
from mini_serve.scheduler import Scheduler, Sequence

VOCAB = 1000


def next_token(history):
    h = 0
    for t in history:
        h = (h * 31 + t + 7) % 1_000_003
    return h % VOCAB


def reference(prompt, n):
    toks = list(prompt)
    for _ in range(n):
        toks.append(next_token(toks))
    return toks[len(prompt):]


class FakeRunner:
    def __init__(self, pool):
        self.pool = pool
        self.kv = {}
        self.rows_computed = 0

    def run(self, tokens, positions, tables, max_blocks, out_rows, seqs):
        P = self.pool.page_size
        row_seq = []
        for seq, start, end in self.items(seqs, out_rows):
            row_seq += [seq] * (end - start)
        for i, (tok, pos) in enumerate(zip(tokens, positions)):
            blk = tables[i * max_blocks + pos // P]
            assert self.pool.ref[blk] == 1, f"write into shared or free block {blk}"
            self.kv[(blk, pos % P)] = (tok, pos)
        for i, pos in enumerate(positions):
            history = []
            for p in range(pos + 1):
                tok, stored_pos = self.kv[(tables[i * max_blocks + p // P], p % P)]
                assert stored_pos == p, "slot holds a token of another position"
                history.append(tok)
            assert history == row_seq[i].tokens[:pos + 1], "attention would read the wrong history"
        self.rows_computed += len(tokens)
        return [next_token(row_seq[r].tokens[:positions[r] + 1]) for r in out_rows]

    @staticmethod
    def items(seqs, out_rows):
        start = 0
        for seq, last in zip(seqs, out_rows):
            yield seq, start, last + 1
            start = last + 1


def make(num_blocks=64, page=4, static=False, **kw):
    pool = BlockPool(num_blocks, page, prefix_cache=not static)
    sched = Scheduler(pool, static=static, **kw)
    return pool, sched, LLM(FakeRunner(pool), sched)


def workload(rng, n, system_len=10, page=4):
    system = [rng.randrange(VOCAB) for _ in range(system_len)]
    return [Sequence(system + [rng.randrange(VOCAB) for _ in range(rng.randrange(1, 9))],
                     max_tokens=rng.randrange(1, 12)) for _ in range(n)]


def assert_pool_clean(tc, pool):
    tc.assertEqual(pool.num_free(), pool.num_blocks)
    tc.assertTrue(all(r == 0 for r in pool.ref))


class BlockPoolTest(unittest.TestCase):
    def test_allocate_release_lru(self):
        pool = BlockPool(3, 4)
        a, b, c = pool.allocate(), pool.allocate(), pool.allocate()
        self.assertEqual(pool.num_free(), 0)
        with self.assertRaises(RuntimeError):
            pool.allocate()
        pool.release([b])
        pool.release([a])
        self.assertEqual(pool.allocate(), b)  # freed longest ago is reused first
        with self.assertRaises(AssertionError):
            pool.release([c, c])

    def test_prefix_hit_and_eviction(self):
        pool = BlockPool(2, 2)
        hs = block_hashes([1, 2, 3, 4], 2)
        b0, b1 = pool.allocate(), pool.allocate()
        pool.register(b0, hs[0])
        pool.register(b1, hs[1])
        self.assertEqual(pool.lookup(hs), [b0, b1])
        self.assertEqual(pool.lookup(block_hashes([1, 2, 9, 9], 2)), [b0])
        self.assertEqual(pool.lookup(block_hashes([9, 2, 3, 4], 2)), [])  # chained: same block 2, other prefix
        pool.take([b0])
        self.assertEqual(pool.ref[b0], 2)
        pool.release([b0, b0, b1])
        self.assertEqual(pool.lookup(hs), [b0, b1])  # freed blocks stay cached
        pool.take([b0])                               # a hit on a free block takes it off the free list
        self.assertEqual(pool.num_free(), 1)
        self.assertEqual(pool.allocate(), b1)         # reusing b1 evicts its cached content
        self.assertEqual(pool.lookup(hs), [b0])

    def test_repeated_block_is_not_a_hit(self):
        # [1, 2] at positions 0-1 and at 2-3 have different K/V (RoPE, context): the second
        # block must not resolve to the first one's cache entry.
        pool, sched, llm = make(page=2)
        a = Sequence([1, 2, 1, 2, 1, 2, 7], max_tokens=3)
        llm.generate([a])
        b = Sequence([1, 2, 1, 2, 9], max_tokens=3)
        llm.generate([b])
        self.assertEqual(b.cached_tokens, 4)
        self.assertEqual(a.output, reference(a.prompt, 3))
        self.assertEqual(b.output, reference(b.prompt, 3))

    def test_disabled_cache_never_hits(self):
        pool = BlockPool(2, 2, prefix_cache=False)
        b = pool.allocate()
        pool.register(b, 123)
        self.assertEqual(pool.lookup([123]), [])


class SchedulerTest(unittest.TestCase):
    def check_outputs(self, seqs):
        for s in seqs:
            self.assertEqual(s.output, reference(s.prompt, s.max_tokens), f"seq {s.id}")

    def test_continuous_matches_reference_and_shares_prefix(self):
        rng = random.Random(1)
        pool, _, llm = make()
        seqs = workload(rng, 12)
        # Blocks are registered after the step that computed them, so the first request goes
        # alone; the other 11 then find the system prompt in the cache.
        llm.generate(seqs[:1])
        llm.generate(seqs[1:])
        self.check_outputs(seqs)
        self.assertEqual(pool.hits, 11 * 2)  # 10-token system prompt = 2 full blocks of 4
        assert_pool_clean(self, pool)

    def test_static_matches_reference_without_cache(self):
        rng = random.Random(2)
        pool, _, llm = make(static=True, max_num_seqs=4)
        seqs = workload(rng, 10)
        llm.generate(seqs)
        self.check_outputs(seqs)
        self.assertEqual(pool.hits, 0)
        assert_pool_clean(self, pool)

    def test_preemption_recomputes_and_matches_reference(self):
        rng = random.Random(3)
        # 12 blocks of 4 tokens: far too few for 8 sequences of up to 30 tokens at once.
        pool, sched, llm = make(num_blocks=12, page=4)
        seqs = [Sequence([rng.randrange(VOCAB) for _ in range(rng.randrange(4, 12))], max_tokens=rng.randrange(8, 18))
                for _ in range(8)]
        llm.generate(seqs)
        self.check_outputs(seqs)
        self.assertGreater(sched.preemptions, 0)
        assert_pool_clean(self, pool)

    def test_preemption_takes_the_latest_admitted(self):
        pool, sched, llm = make(num_blocks=4, page=4)
        a = Sequence(list(range(1, 9)), max_tokens=8)    # 8 tokens = 2 blocks; the 9th needs a 3rd
        b = Sequence(list(range(11, 19)), max_tokens=8)
        sched.add(a)
        sched.add(b)
        llm.step()                                       # prefill both: 4 blocks used
        self.assertEqual(pool.num_free(), 0)
        llm.step()                                       # decode: a needs a new block, b is evicted
        self.assertEqual(sched.running, [a])
        self.assertEqual(list(sched.waiting), [b])
        self.assertEqual(b.preemptions, 1)
        self.assertEqual(b.blocks, [])
        while llm.step():
            pass
        self.check_outputs([a, b])

    def test_new_request_joins_between_decode_steps(self):
        pool, sched, llm = make()
        a = Sequence([1, 2, 3], max_tokens=6)
        sched.add(a)
        llm.step()
        llm.step()
        b = Sequence([4, 5], max_tokens=3)
        sched.add(b)
        batch, _ = llm.step()
        self.assertTrue(batch.prefill)
        self.assertEqual([s for s, _, _ in batch.items], [b])  # prefill only the newcomer
        batch, _ = llm.step()
        self.assertFalse(batch.prefill)
        self.assertEqual({s.id for s, _, _ in batch.items}, {a.id, b.id})
        while llm.step():
            pass
        self.check_outputs([a, b])

    def test_static_batch_does_not_admit_until_done(self):
        pool, sched, llm = make(static=True, max_num_seqs=8)
        a = Sequence([1, 2, 3], max_tokens=5)
        sched.add(a)
        llm.step()
        b = Sequence([4, 5], max_tokens=2)
        sched.add(b)
        seen = []
        while (out := llm.step()):
            seen.append({s.id for s, _, _ in out[0].items})
        self.assertEqual(seen[:4], [{a.id}] * 4)            # a's remaining decode steps, alone
        self.assertEqual(b.first_token_time > a.finish_time, True)
        self.check_outputs([a, b])

    def test_shared_prefix_blocks_are_physically_shared(self):
        pool, sched, llm = make(page=4)
        system = list(range(100, 112))                      # 3 full blocks
        a = Sequence(system + [1, 2], max_tokens=4)
        sched.add(a)
        llm.step()
        b = Sequence(system + [3], max_tokens=4)
        sched.add(b)
        before = llm.runner.rows_computed
        llm.step()
        self.assertEqual(b.blocks[:3], a.blocks[:3])
        self.assertEqual(b.cached_tokens, 12)
        self.assertEqual(llm.runner.rows_computed - before, 1)  # only the 1 uncached token
        self.assertEqual(pool.ref[a.blocks[0]], 2)
        while llm.step():
            pass
        self.check_outputs([a, b])
        assert_pool_clean(self, pool)

    def test_fully_cached_prompt_still_computes_last_token(self):
        pool, sched, llm = make(page=4)
        prompt = list(range(200, 208))                      # exactly 2 blocks
        a = Sequence(prompt, max_tokens=1)
        llm.generate([a])
        b = Sequence(list(prompt), max_tokens=3)
        llm.generate([b])
        self.assertEqual(b.cached_tokens, 4)                # block 2 is recomputed for the logits
        self.check_outputs([a, b])

    def test_prefill_budget_splits_admission(self):
        pool, sched, llm = make(max_prefill_tokens=10)
        seqs = [Sequence(list(range(i * 10, i * 10 + 6)), max_tokens=2) for i in range(3)]
        for s in seqs:
            sched.add(s)
        batch, _ = llm.step()
        self.assertEqual(len(batch.items), 1)               # 6 + 6 > 10
        while llm.step():
            pass
        self.check_outputs(seqs)

    def test_chunked_prefill_matches_reference(self):
        """Every chunk size, with prefix hits and with preemption: same tokens as running alone."""
        for chunk in (3, 5, 16):  # smaller chunks admit too slowly to run out of blocks
            rng = random.Random(4)
            pool, sched, llm = make(chunk_size=chunk)
            seqs = workload(rng, 12)
            llm.generate(seqs[:1])
            llm.generate(seqs[1:])
            self.check_outputs(seqs)
            self.assertEqual(pool.hits, 11 * 2, f"chunk {chunk}")
            assert_pool_clean(self, pool)
            pool, sched, llm = make(num_blocks=12, page=4, chunk_size=chunk)
            seqs = [Sequence([rng.randrange(VOCAB) for _ in range(rng.randrange(4, 12))],
                             max_tokens=rng.randrange(8, 18)) for _ in range(8)]
            llm.generate(seqs)
            self.check_outputs(seqs)
            self.assertGreater(sched.preemptions, 0, f"chunk {chunk}")
            assert_pool_clean(self, pool)

    def test_chunked_step_decodes_while_long_prompt_prefills(self):
        pool, sched, llm = make(chunk_size=4)
        a = Sequence([1, 2, 3], max_tokens=8)
        sched.add(a)
        llm.step()                                          # a's prompt: 3 tokens, first token sampled
        b = Sequence(list(range(10, 20)), max_tokens=2)     # 10 tokens: 3 chunks of at most 3
        sched.add(b)
        chunks = []
        while not b.output:
            n_a = len(a.output)
            batch, _ = llm.step()
            self.assertLessEqual(sum(e - s for _, s, e in batch.items), 4)
            self.assertEqual(batch.items[0], (a, len(a.tokens) - 2, len(a.tokens) - 1))  # decode first
            self.assertEqual(len(a.output), n_a + 1)                                     # a never waits
            chunks.append(batch.items[1][1:])
        self.assertEqual(chunks, [(0, 3), (3, 6), (6, 9), (9, 10)])
        while llm.step():
            pass
        self.check_outputs([a, b])
        assert_pool_clean(self, pool)

    def test_rejects_request_larger_than_pool(self):
        _, sched, _ = make(num_blocks=2, page=4)
        with self.assertRaises(ValueError):
            sched.add(Sequence([1] * 6, max_tokens=4))



def check_invariants(tc, pool, sched, batch=None):
    """What must hold between any two steps, whatever the policy and the history."""
    P = pool.page_size
    holders = {}
    for seq in sched.running:
        tc.assertEqual(len(set(seq.blocks)), len(seq.blocks), "a block twice in one table")
        tc.assertLessEqual(seq.num_computed, len(seq.tokens))
        tc.assertGreaterEqual(len(seq.blocks) * P, seq.num_computed)
        tc.assertLessEqual(len(seq.hashes), seq.num_computed // P)
        for i, b in enumerate(seq.blocks):
            holders.setdefault(b, []).append((seq, i))
    for seq in sched.waiting:
        tc.assertEqual((seq.blocks, seq.num_computed), ([], 0), "a waiting sequence holds blocks")
    # Reference counts are exactly the number of tables holding the block; the free list is
    # exactly the blocks nobody holds.
    for b in range(pool.num_blocks):
        tc.assertEqual(pool.ref[b], len(holders.get(b, [])), f"refcount of block {b}")
    tc.assertEqual(set(pool.free), {b for b in range(pool.num_blocks) if pool.ref[b] == 0})
    # A shared block is a full, registered prefix block at the same index with the same hash in
    # every table, and nobody writes into it.
    for b, hs in holders.items():
        if len(hs) > 1:
            tc.assertFalse(sched.static, "static batches share nothing")
            tc.assertEqual(len({i for _, i in hs}), 1)
            tc.assertEqual(len({seq.hashes[i] if i < len(seq.hashes) else None for seq, i in hs}), 1)
            tc.assertIsNotNone(hs[0][0].hashes[hs[0][1]] if hs[0][1] < len(hs[0][0].hashes) else None)
    for seq in sched.running:
        for b in seq.blocks[seq.num_computed // P:]:
            tc.assertEqual(pool.ref[b], 1, "a block still to be written is shared")
    tc.assertEqual({pool.by_hash[h] for h in pool.by_hash}, set(pool.hash_of))
    tc.assertTrue(all(pool.by_hash[h] == b for b, h in pool.hash_of.items()))
    tc.assertLessEqual(len(sched.running), sched.max_num_seqs)
    if batch is not None:
        n = sum(e - s for _, s, e in batch.items)
        if sched.chunk_size:
            tc.assertLessEqual(n, sched.chunk_size)
        elif batch.prefill and not sched.static:
            first = batch.items[0]
            tc.assertLessEqual(n, max(sched.max_prefill_tokens, first[2] - first[1]))


class SchedulerPropertyTest(unittest.TestCase):
    """Random pools, policies and request streams, with requests added and aborted between any
    two steps. After every step the invariants above hold, the fake runner has checked every
    row's attention history, and in the end every request that was not aborted has exactly the
    tokens it would get running alone, and every block is free."""

    def test_random_histories(self):
        for case in range(300):
            rng = random.Random(case)
            static = rng.random() < 0.2
            page = rng.randint(1, 5)
            num_blocks = rng.randint(4, 40)
            kw = dict(max_num_seqs=rng.randint(1, 6), max_prefill_tokens=rng.randint(1, 24))
            if not static and rng.random() < 0.5:
                kw["chunk_size"] = rng.randint(1, 10)
            pool = BlockPool(num_blocks, page, prefix_cache=not static and rng.random() < 0.8)
            sched = Scheduler(pool, static=static, **kw)
            llm = LLM(FakeRunner(pool), sched)
            prefixes = [[rng.randrange(20) for _ in range(rng.randint(0, 3 * page))] for _ in range(3)]
            seqs, aborted = [], set()
            msg = f"case {case}: static={static} page={page} blocks={num_blocks} {kw}"
            for _ in range(rng.randint(5, 120)):
                r = rng.random()
                if r < 0.25:
                    prompt = rng.choice(prefixes) + [rng.randrange(20) for _ in range(rng.randint(1, 2 * page + 3))]
                    seq = Sequence(prompt, max_tokens=rng.randint(1, 3 * page))
                    try:
                        sched.add(seq)
                    except ValueError:
                        continue
                    seqs.append(seq)
                elif r < 0.32 and seqs:
                    seq = rng.choice(seqs)
                    if not seq.finished:
                        sched.abort(seq)
                        aborted.add(seq.id)
                else:
                    busy = sched.has_work()
                    out = llm.step()
                    self.assertEqual(out is not None, busy, msg)  # never stuck with work
                check_invariants(self, pool, sched, out[0] if r >= 0.32 and out else None)
            while (out := llm.step()):
                check_invariants(self, pool, sched, out[0])
            for seq in seqs:
                if seq.id not in aborted:
                    self.assertEqual(seq.output, reference(seq.prompt, seq.max_tokens), msg)
            assert_pool_clean(self, pool)


if __name__ == "__main__":
    unittest.main()
