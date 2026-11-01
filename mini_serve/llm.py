"""The step loop: scheduler -> flat token batch -> runner -> sampled tokens -> scheduler."""


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
        self.steps = {"prefill": 0, "decode": 0}

    def step(self):
        """Runs one step. Returns (batch, sampled tokens), or None if there is nothing to do."""
        batch = self.scheduler.schedule()
        if batch is None:
            return None
        seqs = [seq for seq, _, _ in batch.items]
        next_tokens = self.runner.run(*build_inputs(batch), seqs)
        self.scheduler.update(batch, next_tokens)
        self.steps["prefill" if batch.prefill else "decode"] += 1
        return batch, next_tokens

    def generate(self, seqs):
        """Offline use: runs `seqs` to completion, returns their outputs in order."""
        for s in seqs:
            self.scheduler.add(s)
        while self.step():
            pass
        return [s.output for s in seqs]
