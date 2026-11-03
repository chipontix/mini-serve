"""Request scheduling: continuous batching with a prefix cache, and a static-batch baseline.

Every step runs one forward pass over a list of (sequence, tokens[start:end]) items and samples
one token for each sequence in it. A sequence's `num_computed` tokens already have their K/V in
its blocks; a step computes the rest, tokens[num_computed:], and samples the next token. So
prefill and decode are the same operation: a prefill computes the whole uncached prompt, a decode
computes the single token sampled last step.

Continuous policy (`static=False`):
  * Between steps, waiting requests are admitted (FCFS) while the pool has blocks for their
    uncached tokens and the prefill token budget allows. If any were admitted, the step is a
    prefill step of only those; running sequences wait one step. Prefill and decode are never
    mixed in one step.
  * Otherwise the step is a decode step of every running sequence. A sequence whose next token
    starts a new block gets one; if the pool is empty, the most recently admitted running
    sequence is preempted: its blocks are freed and it goes back to the front of the queue, to
    be recomputed from prompt + generated tokens when there is room (recompute, not swap).
  * Finished sequences free their blocks at once, so their slots go to waiting requests on the
    next step.

Chunked prefill (`chunk_size=N`, continuous only; off by default): one step computes at most N
tokens and mixes decode with prefill. Running sequences that decode get their one token first,
then sequences still in prefill get the next chunk of their prompt, then waiting requests are
admitted with whatever budget is left (all their blocks are taken at admission, as above). A
chunk that stops before the sequence's last token samples nothing. So a long prompt no longer
holds every running sequence back for a whole prefill step.

Static policy (`static=True`), the baseline: when nothing is running, take up to max_num_seqs
waiting requests, reserve every block they could need (prompt + max_tokens), prefill them
together and decode until the last one finishes. New requests wait for the whole batch. No
prefix cache, no preemption.
"""

import itertools
import time
from collections import deque
from dataclasses import dataclass, field

from .blocks import BlockPool, block_hashes


@dataclass(eq=False)  # compared by identity: `seq in running` must not compare every field
class Sequence:
    prompt: list
    max_tokens: int
    eos_id: int = -1           # -1: never stop on a token
    temperature: float = 0.0
    id: int = field(default_factory=itertools.count().__next__)
    output: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    num_computed: int = 0
    hashes: list = field(default_factory=list)   # hashes of this sequence's full, registered blocks
    arrival: float = field(default_factory=time.perf_counter)
    first_token_time: float = None
    finish_time: float = None
    preemptions: int = 0
    cached_tokens: int = 0     # prompt tokens served from the prefix cache at the last admission

    @property
    def tokens(self):
        return self.prompt + self.output

    @property
    def finished(self):
        return self.finish_time is not None


@dataclass
class Batch:
    items: list   # (seq, start, end): compute seq.tokens[start:end]
    prefill: bool
    sample: list = field(init=False)  # per item: it reaches the sequence's last token, so a token is sampled

    def __post_init__(self):
        self.sample = [end == len(seq.tokens) for seq, _, end in self.items]


class Scheduler:
    def __init__(self, pool: BlockPool, max_num_seqs=64, max_prefill_tokens=2048, static=False, chunk_size=None):
        assert not (static and chunk_size), "chunked prefill is a continuous-batching policy"
        assert chunk_size is None or chunk_size >= 1
        self.pool = pool
        self.max_num_seqs = max_num_seqs
        self.max_prefill_tokens = max_prefill_tokens
        self.static = static
        self.chunk_size = chunk_size
        self.waiting = deque()
        self.running = []          # admission order: the last one is preempted first
        self.preemptions = 0

    def add(self, seq: Sequence):
        P = self.pool.page_size
        if -(-(len(seq.prompt) + seq.max_tokens) // P) > self.pool.num_blocks:
            raise ValueError("prompt + max_tokens does not fit in the KV pool")
        if not seq.prompt or seq.max_tokens < 1:
            raise ValueError("empty prompt or max_tokens < 1")
        self.waiting.append(seq)

    def abort(self, seq: Sequence):
        """Drops a request (e.g. the client went away) and frees its blocks."""
        if seq in self.waiting:
            self.waiting.remove(seq)
        elif seq in self.running:
            self.running.remove(seq)
            self._free(seq)

    def has_work(self):
        return bool(self.waiting or self.running)

    def blocks_needed(self, n_tokens):
        return -(-n_tokens // self.pool.page_size)

    # ------------------------------------------------------------------------------ admission

    def _admit(self, seq, budget):
        """Give `seq` blocks for all its tokens, reusing cached prefix blocks. Returns the number
        of tokens to compute, or None if it does not fit (nothing is taken then)."""
        pool, P, tokens = self.pool, self.pool.page_size, seq.tokens
        if self.static:
            hit, need = [], self.blocks_needed(len(seq.prompt) + seq.max_tokens)
        else:
            # At least the last token is computed: its logits give the next token.
            hit = pool.lookup(block_hashes(tokens, P)[:(len(tokens) - 1) // P])
            need = self.blocks_needed(len(tokens)) - len(hit)
        hit_free = sum(1 for b in hit if pool.ref[b] == 0)  # cached but free: taking them uses free slots
        n_compute = len(tokens) - len(hit) * P
        if need + hit_free > pool.num_free() or n_compute > budget:
            return None
        pool.take(hit)
        seq.blocks = hit + [pool.allocate() for _ in range(need)]
        seq.hashes = block_hashes(tokens, P)[:len(hit)]
        seq.num_computed = len(hit) * P
        seq.cached_tokens = seq.num_computed
        return n_compute

    def _free(self, seq):
        self.pool.release(seq.blocks)
        seq.blocks, seq.hashes, seq.num_computed = [], [], 0

    def _preempt(self, seq):
        self.running.remove(seq)
        self._free(seq)
        seq.preemptions += 1
        self.preemptions += 1
        self.waiting.appendleft(seq)

    # ------------------------------------------------------------------------------ schedule

    def schedule(self):
        """The next step's batch, or None if there is nothing to run."""
        if self.static and self.running:
            return self._decode()
        if self.chunk_size:
            return self._schedule_chunked()
        admitted = []
        budget = float("inf") if self.static else self.max_prefill_tokens
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            # The first prompt is admitted even if it is longer than the budget.
            n = self._admit(seq, budget if admitted else max(budget, len(seq.tokens)))
            if n is None:
                break
            self.waiting.popleft()
            self.running.append(seq)
            admitted.append(seq)
            budget -= n
        if admitted:
            return Batch([(s, s.num_computed, len(s.tokens)) for s in admitted], prefill=True)
        return self._decode() if self.running else None

    def _schedule_chunked(self):
        self._grow()
        budget, items, prefill = self.chunk_size, [], False
        left = lambda s: len(s.tokens) - s.num_computed
        # Decodes first, so a long prompt never delays a running sequence's next token.
        for seq in sorted(self.running, key=lambda s: left(s) > 1):
            n = min(left(seq), budget)
            if n:
                items.append((seq, seq.num_computed, seq.num_computed + n))
                budget -= n
                prefill |= left(seq) > 1
        while budget and self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            n = self._admit(seq, float("inf"))
            if n is None:
                break
            self.waiting.popleft()
            self.running.append(seq)
            n = min(n, budget)
            items.append((seq, seq.num_computed, seq.num_computed + n))
            budget -= n
            prefill = True
        return Batch(items, prefill=prefill) if items else None

    def _decode(self):
        self._grow()
        if not self.running:
            return None
        return Batch([(s, s.num_computed, len(s.tokens)) for s in self.running], prefill=False)

    def _grow(self):
        """Gives every running sequence a block for its next token, preempting if the pool is empty."""
        P = self.pool.page_size
        for seq in list(self.running):
            if seq not in self.running:   # preempted earlier in this loop
                continue
            while len(seq.blocks) * P < len(seq.tokens):
                if self.pool.num_free():
                    seq.blocks.append(self.pool.allocate())
                else:
                    victim = self.running[-1]
                    self._preempt(victim)
                    if victim is seq:
                        break

    # ------------------------------------------------------------------------------ update

    def update(self, batch: Batch, next_tokens, now=None):
        """Record the sampled token of every item. Returns the sequences that finished."""
        now = time.perf_counter() if now is None else now
        P = self.pool.page_size
        done = []
        for (seq, _, end), tok, sample in zip(batch.items, next_tokens, batch.sample):
            seq.num_computed = end
            # Register blocks that just became full, so later requests can share them.
            new = block_hashes(seq.tokens[:end], P, start=len(seq.hashes),
                               prev=seq.hashes[-1] if seq.hashes else None)
            for i, h in enumerate(new, start=len(seq.hashes)):
                if not self.static:
                    self.pool.register(seq.blocks[i], h)
            seq.hashes += new
            if not sample:   # a prefill chunk short of the last token: its logits are not used
                continue
            seq.output.append(int(tok))
            if seq.first_token_time is None:
                seq.first_token_time = now
            if len(seq.output) >= seq.max_tokens or int(tok) == seq.eos_id:
                seq.finish_time = now
                self.running.remove(seq)
                self._free(seq)
                done.append(seq)
        return done
