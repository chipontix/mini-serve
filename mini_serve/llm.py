"""The step loop: scheduler -> flat token batch -> runner -> sampled tokens -> scheduler."""

import time


def build_inputs(batch):
    """Flattens a Batch into the runner's arguments: one row per token to compute, each with its
    position and its sequence's block table (padded to the longest table in the batch), plus the
    index of each sequence's last row, whose logits give its next token."""
    max_blocks = max(len(seq.blocks) for seq, _, _ in batch.items)
    tokens, positions, tables, out_rows = [], [], [], []
    for seq, start, end in batch.items:
        row = seq.blocks + [0] * (max_blocks - len(seq.blocks))
        all_tokens = seq.tokens
        for p in range(start, end):
            tokens.append(all_tokens[p])
            positions.append(p)
            tables.extend(row)
        out_rows.append(len(tokens) - 1)
    return tokens, positions, tables, max_blocks, out_rows


class LLM:
    def __init__(self, runner, scheduler):
        self.runner = runner
        self.scheduler = scheduler
        self.stats = {"prefill_steps": 0, "decode_steps": 0, "prefill_rows": 0, "decode_rows": 0,
                      "step_s": 0.0, "engine_s": 0.0, "matmul_s": 0.0, "attention_s": 0.0}

    def step(self):
        """Runs one step. Returns (batch, sampled tokens), or None if there is nothing to do."""
        t0 = time.perf_counter()
        batch = self.scheduler.schedule()
        if batch is None:
            return None
        seqs = [seq for seq, _, _ in batch.items]
        inputs = build_inputs(batch)
        next_tokens = self.runner.run(*inputs, seqs)
        self.scheduler.update(batch, next_tokens)
        kind = "prefill" if batch.prefill else "decode"
        self.stats[kind + "_steps"] += 1
        self.stats[kind + "_rows"] += len(inputs[0])
        self.stats["step_s"] += time.perf_counter() - t0
        if hasattr(self.runner, "profile"):
            for k, v in self.runner.profile().items():
                self.stats[("engine" if k == "total" else k) + "_s"] += v
        return batch, next_tokens

    def generate(self, seqs):
        """Offline use: runs `seqs` to completion, returns their outputs in order."""
        for s in seqs:
            self.scheduler.add(s)
        while self.step():
            pass
        return [s.output for s in seqs]
