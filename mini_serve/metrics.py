"""Prometheus text exposition (format 0.0.4) for GET /metrics, with no client library: counters,
gauges and fixed-bucket histograms, rendered from plain numbers.

Engine-side numbers (steps, preemptions, cache hits, pool occupancy, step time) come from the
engine process with each /stats snapshot; request latencies (TTFT, TPOT, end to end) are measured
in the HTTP process, where the client's clock-relevant events happen.
"""

import bisect
import threading

STEP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


class Histogram:
    def __init__(self, buckets):
        self.buckets = tuple(buckets)
        self.counts = [0] * (len(self.buckets) + 1)   # last one: above the largest bucket
        self.sum = 0.0

    def observe(self, v):
        self.counts[bisect.bisect_left(self.buckets, v)] += 1   # Prometheus buckets are "<= le"
        self.sum += v

    def snapshot(self):
        return {"buckets": list(self.buckets), "counts": list(self.counts), "sum": self.sum}


class RequestMetrics:
    """Per-request numbers of the HTTP process. Handler threads call these concurrently."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = {"ok": 0, "aborted": 0, "rejected": 0, "bad_request": 0}
        self.prompt_tokens = 0
        self.ttft = Histogram(LATENCY_BUCKETS)
        self.tpot = Histogram(STEP_BUCKETS)
        self.e2e = Histogram(LATENCY_BUCKETS)

    def finished(self, outcome, prompt_tokens=0, ttft=None, tpot=None, e2e=None):
        with self.lock:
            self.requests[outcome] += 1
            self.prompt_tokens += prompt_tokens
            for h, v in ((self.ttft, ttft), (self.tpot, tpot), (self.e2e, e2e)):
                if v is not None:
                    h.observe(v)

    def snapshot(self):
        with self.lock:
            return {"requests": dict(self.requests), "prompt_tokens": self.prompt_tokens,
                    "ttft": self.ttft.snapshot(), "tpot": self.tpot.snapshot(), "e2e": self.e2e.snapshot()}


def _labels(labels):
    return "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}" if labels else ""


def _num(v):
    return repr(float(v)) if isinstance(v, float) else str(v)


def render(engine, req):
    """`engine`: the engine's /stats dict; `req`: RequestMetrics.snapshot()."""
    out = []

    def family(name, kind, help_, samples):
        out.append(f"# HELP {name} {help_}")
        out.append(f"# TYPE {name} {kind}")
        for suffix, labels, value in samples:
            out.append(f"{name}{suffix}{_labels(labels)} {_num(value)}")

    def histogram(name, help_, snap):
        samples, total = [], 0
        for le, c in zip(snap["buckets"] + ["+Inf"], snap["counts"]):
            total += c
            samples.append(("_bucket", {"le": le}, total))
        samples += [("_sum", {}, float(snap["sum"])), ("_count", {}, total)]
        family(name, "histogram", help_, samples)

    p = "mini_serve_"
    family(p + "requests_total", "counter", "Completion requests by outcome.",
           [("", {"outcome": k}, v) for k, v in req["requests"].items()])
    family(p + "prompt_tokens_total", "counter", "Prompt tokens of accepted requests.",
           [("", {}, req["prompt_tokens"])])
    family(p + "generation_tokens_total", "counter", "Tokens sampled by the engine.",
           [("", {}, engine["sampled_tokens"])])
    family(p + "steps_total", "counter", "Engine steps by kind.",
           [("", {"kind": k}, engine[f"{k}_steps"]) for k in ("prefill", "decode")])
    family(p + "step_rows_total", "counter", "Token rows computed by engine steps, by kind.",
           [("", {"kind": k}, engine[f"{k}_rows"]) for k in ("prefill", "decode")])
    family(p + "preemptions_total", "counter", "Sequences preempted (blocks freed, recomputed later).",
           [("", {}, engine["preemptions"])])
    family(p + "prefix_cache_hit_blocks_total", "counter", "KV blocks taken from the prefix cache instead of computed.",
           [("", {}, engine["prefix_hit_blocks"])])
    family(p + "engine_seconds_total", "counter", "Time inside the C++ forward pass, and its matmul and attention parts.",
           [("", {"part": "forward"}, float(engine["engine_s"])), ("", {"part": "matmul"}, float(engine["matmul_s"])),
            ("", {"part": "attention"}, float(engine["attention_s"]))])
    family(p + "running_sequences", "gauge", "Sequences in the running batch.", [("", {}, engine["running"])])
    family(p + "waiting_sequences", "gauge", "Sequences waiting for admission.", [("", {}, engine["waiting"])])
    family(p + "kv_blocks_free", "gauge", "Free KV blocks (cached prefix blocks with no reference included).",
           [("", {}, engine["free_blocks"])])
    family(p + "kv_blocks_total", "gauge", "KV blocks in the pool.", [("", {}, engine["num_blocks"])])
    histogram(p + "step_seconds", "Wall time of one engine step.", engine["step_hist"])
    histogram(p + "time_to_first_token_seconds", "From request received to first token.", req["ttft"])
    histogram(p + "time_per_output_token_seconds", "Mean time between tokens of one request.", req["tpot"])
    histogram(p + "request_seconds", "From request received to last token.", req["e2e"])
    return "\n".join(out) + "\n"
