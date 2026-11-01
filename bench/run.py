"""Load generator and benchmark: the same request stream against each server configuration.

    python -m bench.run --label NAME [--quick]

For every configuration a server process is started, warmed up, and sent a Poisson stream of
streaming /v1/completions requests (seeded, so every configuration sees the same prompts,
lengths and arrival times). Each request records send time, the arrival time of every streamed
token, and the token count. Writes results/NAME/{env.txt,requests.csv,summary.csv}.

Workload: every prompt is one shared system prompt (~250 tokens) plus a short user question;
max_tokens is drawn from [32, 128] and the server is told to ignore EOS, so every configuration
generates exactly the same number of tokens. Greedy sampling.

Metrics per (configuration, rate):
  TTFT  time from sending the request to the first streamed token
  TPOT  (last token time - first token time) / (tokens - 1), per request
  throughput  generated tokens / (last token of the run - first request sent)
  goodput     requests that met the SLO (TTFT <= SLO_TTFT and TPOT <= SLO_TPOT) per second
"""

import argparse
import csv
import http.client
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLO_TTFT, SLO_TPOT = 1.0, 0.100   # seconds

SYSTEM = (
    "You are a helpful assistant for a small hardware store. You answer questions about tools, "
    "materials and home repairs. Keep answers short and practical, mention safety precautions "
    "when a task involves power tools, electricity, ladders or chemicals, and recommend calling a "
    "licensed professional for gas lines, structural work and main electrical panels. When the "
    "customer asks about a product, describe what it is used for, what to look for when buying "
    "it, and one common mistake people make with it. If you are not sure, say so instead of "
    "guessing. Prices change often, so never quote a price; suggest checking the shelf label or "
    "asking at the front desk. The store is open from eight in the morning to seven in the "
    "evening on weekdays and from nine to five on weekends. Returns are accepted within thirty "
    "days with a receipt. Customers can rent a carpet cleaner, a tile saw and a pressure washer "
    "by the day. Workshops for beginners run every Saturday morning.\n\n"
)
QUESTIONS = [
    "How do I fix a leaking kitchen faucet?", "What drill bit should I use for concrete?",
    "Can I paint over wallpaper?", "How do I choose a ladder for cleaning gutters?",
    "What is the difference between a nail and a screw for framing?", "How do I unclog a drain without chemicals?",
    "Which sandpaper grit should I start with on old furniture?", "How do I hang a heavy mirror on drywall?",
    "Is it safe to replace a light switch myself?", "What glue works on ceramic?",
    "How do I seal a bathtub with silicone?", "What size extension cord do I need for a table saw?",
    "How do I remove rust from garden tools?", "Can I rent the tile saw for a weekend?",
    "What kind of paint is best for a bathroom ceiling?", "How do I stop a door from squeaking?",
]

CONFIGS = {   # name -> server flags
    "static": ["--static"],
    "continuous": ["--no-prefix-cache"],
    "continuous+prefix": [],
}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(port, path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path)
    return json.loads(c.getresponse().read())


def start_server(flags, num_blocks, max_num_seqs):
    port = free_port()
    cmd = [sys.executable, "-m", "mini_serve.server", "--port", str(port), "--num-blocks", str(num_blocks),
           "--max-num-seqs", str(max_num_seqs)] + flags
    proc = subprocess.Popen(cmd, cwd=ROOT)
    for _ in range(600):
        try:
            get(port, "/v1/models")
            return proc, port
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("server did not start")


def one_request(port, prompt, max_tokens, rec):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    body = json.dumps({"model": "SmolLM2-135M", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True, "ignore_eos": True})
    rec["sent"] = time.perf_counter()
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    resp = c.getresponse()
    times = []
    for line in resp:
        if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
            times.append(time.perf_counter())
    rec["times"] = times


def workload(n, rate, seed):
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append((t, SYSTEM + "Customer: " + rng.choice(QUESTIONS) + "\nAssistant:", rng.randint(32, 128)))
    return out


def run_load(port, reqs):
    recs = [{} for _ in reqs]
    threads = []
    t0 = time.perf_counter()
    for (at, prompt, mt), rec in zip(reqs, recs):
        delay = t0 + at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        th = threading.Thread(target=one_request, args=(port, prompt, mt, rec))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    return recs


def summarize(recs, reqs):
    ttft = [r["times"][0] - r["sent"] for r in recs]
    tpot = [(r["times"][-1] - r["times"][0]) / (len(r["times"]) - 1) for r in recs]
    span = max(r["times"][-1] for r in recs) - min(r["sent"] for r in recs)
    ok = sum(1 for a, b in zip(ttft, tpot) if a <= SLO_TTFT and b <= SLO_TPOT)
    tokens = sum(len(r["times"]) for r in recs)
    assert tokens == sum(mt for _, _, mt in reqs), "a request got fewer tokens than max_tokens"
    q = lambda xs, p: statistics.quantiles(xs, n=100)[p - 1]
    return {"ttft_p50": statistics.median(ttft), "ttft_p99": q(ttft, 99),
            "tpot_p50": statistics.median(tpot), "tpot_p99": q(tpot, 99),
            "throughput_tok_s": tokens / span, "goodput_req_s": ok / span,
            "slo_met": ok / len(recs), "duration_s": span, "tokens": tokens}


def free_gb():
    st = os.statvfs(Path.home())
    return st.f_bavail * st.f_frsize / 1e9


class LoadSampler(threading.Thread):
    """Records load average and free disk every 30 s into load.csv, for judging the run."""

    def __init__(self, path, every=30.0):
        super().__init__(daemon=True)
        self.path, self.every, self.stop, self.max_load = path, every, threading.Event(), 0.0
        with open(path, "w") as f:
            f.write("unix_time,load1,load5,load15,free_disk_gb\n")

    def sample(self):
        l1, l5, l15 = os.getloadavg()
        self.max_load = max(self.max_load, l1)
        with open(self.path, "a") as f:
            f.write(f"{time.time():.0f},{l1:.2f},{l5:.2f},{l15:.2f},{free_gb():.1f}\n")
        # Whose load it is: this benchmark's own processes count too.
        top = subprocess.run("ps -Ao pcpu,comm -r | head -6", shell=True, capture_output=True, text=True).stdout
        with open(self.path.with_name("top.txt"), "a") as f:
            f.write(f"--- {time.strftime('%H:%M:%S')} load {l1:.2f}\n{top}")

    def run(self):
        self.sample()
        while not self.stop.wait(self.every):
            self.sample()


def env_text():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except subprocess.TimeoutExpired:
            return "timeout"
    return "\n".join([
        f"date: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"commit: {sh('git rev-parse --short HEAD')} {'(dirty)' if sh('git status --porcelain') else ''}",
        f"chip: {sh('sysctl -n machdep.cpu.brand_string')}, memory: {int(sh('sysctl -n hw.memsize')) >> 30} GiB",
        f"macOS: {sh('sw_vers -productVersion')}",
        f"power: {sh('pmset -g | grep -E \"powermode|lowpowermode\"')}",
        f"load before: {sh('uptime')}",
        f"free disk before: {free_gb():.1f} GB",
        "top processes:", sh("ps -Ao pcpu,comm -r | head -6"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--quick", action="store_true", help="one small rate, for debugging")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--rates", default="2,4,8,16,32")
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--pool-sweep", default="64,96,128,192,256,1024",
                    help="blocks, for the pool-size experiment (continuous+prefix and static)")
    ap.add_argument("--pool-rate", type=float, default=32)
    args = ap.parse_args()
    if args.quick:
        args.requests, args.rates, args.pool_sweep = 24, "4", ""

    out = ROOT / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.txt").write_text(env_text() + "\n")
    sampler = LoadSampler(out / "load.csv")
    sampler.start()
    rows, req_rows = [], []

    def measure(experiment, cfg, rate, num_blocks):
        reqs = workload(args.requests, rate, seed=1)
        proc, port = start_server(CONFIGS[cfg], num_blocks, args.max_num_seqs)
        try:
            run_load(port, workload(4, 100.0, seed=99))      # warm-up: shader compile, allocations
            before = get(port, "/stats")
            recs = run_load(port, reqs)
            after = get(port, "/stats")
        finally:
            proc.terminate()
            proc.wait()
        stats = {k: after[k] - before[k] for k in after}
        s = summarize(recs, reqs)
        s.update(experiment=experiment, config=cfg, rate=rate, num_blocks=num_blocks,
                 preemptions=stats["preemptions"], prefix_hit_blocks=stats["prefix_hit_blocks"],
                 mean_decode_batch=stats["decode_rows"] / max(1, stats["decode_steps"]),
                 step_s=stats["step_s"], engine_s=stats["engine_s"], matmul_s=stats["matmul_s"],
                 attention_s=stats["attention_s"], steps=stats["prefill_steps"] + stats["decode_steps"])
        rows.append(s)
        for i, r in enumerate(recs):
            req_rows.append({"experiment": experiment, "config": cfg, "rate": rate, "num_blocks": num_blocks,
                             "req": i, "max_tokens": reqs[i][2], "ttft": r["times"][0] - r["sent"],
                             "e2e": r["times"][-1] - r["sent"], "tokens": len(r["times"])})
        print(f"{experiment:5} {cfg:18} rate {rate:5} blocks {num_blocks:5}: "
              f"TTFT p50 {s['ttft_p50']*1e3:7.1f} ms  TPOT p50 {s['tpot_p50']*1e3:6.1f} ms  "
              f"{s['throughput_tok_s']:7.1f} tok/s  goodput {s['goodput_req_s']:5.2f} req/s  "
              f"preempt {s['preemptions']}", flush=True)

    for rate in [float(x) for x in args.rates.split(",")]:
        for cfg in CONFIGS:
            measure("rate", cfg, rate, args.num_blocks)
    for nb in [int(x) for x in args.pool_sweep.split(",") if x]:
        for cfg in ("static", "continuous+prefix"):
            measure("pool", cfg, args.pool_rate, nb)

    for name, data in (("summary.csv", rows), ("requests.csv", req_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    sampler.stop.set()
    sampler.sample()
    with open(out / "env.txt", "a") as f:
        f.write(f"load after: {subprocess.run('uptime', capture_output=True, text=True).stdout.strip()}\n")
        f.write(f"free disk after: {free_gb():.1f} GB\n")
        f.write(f"max 1-min load average sampled every 30 s: {sampler.max_load:.2f}"
                f"{'  ABOVE 16: repeat this run' if sampler.max_load > 16 else ''}\n")
        f.write(f"command: {' '.join(sys.argv)}\nSLO: TTFT <= {SLO_TTFT} s, TPOT <= {SLO_TPOT} s\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
