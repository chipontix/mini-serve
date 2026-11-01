"""Markdown tables for the README, from results/LABEL/summary.csv.

    python3 scripts/tables.py LABEL
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(label):
    with open(ROOT / "results" / label / "summary.csv") as f:
        return list(csv.DictReader(f))


def ms(x):
    return f"{float(x) * 1e3:.0f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(label):
    rows = load(label)
    rate = [r for r in rows if r["experiment"] == "rate"]
    pool = [r for r in rows if r["experiment"] == "pool"]
    print(f"**Same request stream at each rate** ({label}, {rate[0]['num_blocks']} blocks)\n")
    print(table(["req/s", "config", "TTFT p50 / p99, ms", "TPOT p50 / p99, ms", "tok/s", "goodput, req/s",
                 "SLO met", "mean decode batch", "prefix-hit blocks"],
                [[f"{float(r['rate']):g}", r["config"], f"{ms(r['ttft_p50'])} / {ms(r['ttft_p99'])}",
                  f"{ms(r['tpot_p50'])} / {ms(r['tpot_p99'])}", f"{float(r['throughput_tok_s']):.0f}",
                  f"{float(r['goodput_req_s']):.2f}", f"{float(r['slo_met']):.0%}",
                  f"{float(r['mean_decode_batch']):.1f}", r["prefix_hit_blocks"]] for r in rate]))
    if pool:
        print(f"\n**KV pool size at {float(pool[0]['rate']):g} req/s**\n")
        print(table(["blocks (tokens)", "config", "tok/s", "TTFT p50, ms", "TPOT p50, ms", "goodput, req/s",
                     "preemptions", "mean decode batch"],
                    [[f"{r['num_blocks']} ({int(r['num_blocks']) * 16})", r["config"],
                      f"{float(r['throughput_tok_s']):.0f}", ms(r["ttft_p50"]), ms(r["tpot_p50"]),
                      f"{float(r['goodput_req_s']):.2f}", r["preemptions"], f"{float(r['mean_decode_batch']):.1f}"]
                     for r in pool]))
    print("\n**Where a step's time goes** (sums over the measured run)\n")
    print(table(["req/s", "config", "steps", "ms per step", "matmul (CPU)", "attention (GPU, incl. wait)",
                 "rest of engine", "Python scheduler + sampling"],
                [[f"{float(r['rate']):g}", r["config"], r["steps"],
                  f"{float(r['step_s']) / int(r['steps']) * 1e3:.1f}",
                  f"{float(r['matmul_s']) / float(r['step_s']):.0%}",
                  f"{float(r['attention_s']) / float(r['step_s']):.0%}",
                  f"{(float(r['engine_s']) - float(r['matmul_s']) - float(r['attention_s'])) / float(r['step_s']):.0%}",
                  f"{(float(r['step_s']) - float(r['engine_s'])) / float(r['step_s']):.0%}"] for r in rate]))


if __name__ == "__main__":
    main(sys.argv[1])
