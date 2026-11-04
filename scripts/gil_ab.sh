#!/usr/bin/env bash
# A/B for README "What I learned": engine loop in a thread of the HTTP process (commit dae6341)
# vs in its own process (this checkout). Same quick load (24 requests, 4 req/s, 3 configs),
# rounds interleaved. Writes results/gil-ab/{thread,process}-N.csv and the step-time tables.
# dae6341 also called tok.get_vocab_size() once per prompt token (~1 s with the GIL held per
# request, fixed later); the same fix is applied to it here, so the A/B isolates thread vs process.
#   scripts/gil_ab.sh [ROUNDS]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROUNDS="${1:-2}"
OLD="$(mktemp -d)/mini-serve-dae6341"
git -C "$ROOT" worktree add -q --detach "$OLD" dae6341
trap 'git -C "$ROOT" worktree remove --force "$OLD"' EXIT
sed -i '' 's/0 <= t < tok.get_vocab_size() for t in ids/0 <= t < 49152 for t in ids/' "$OLD/mini_serve/server.py"
grep -q "0 <= t < 49152 for t in ids" "$OLD/mini_serve/server.py"
mkdir -p "$OLD/weights"  # dae6341 read the weights from weights/ in the checkout
ln -s "$("$ROOT/scripts/fetch_weights.sh" 2>/dev/null)" "$OLD/weights/SmolLM2-135M"
ln -s "$ROOT/.venv" "$OLD/.venv"
cmake -S "$OLD" -B "$OLD/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$OLD/build" -j >/dev/null
OUT="$ROOT/results/gil-ab"
mkdir -p "$OUT"
for i in $(seq "$ROUNDS"); do
  for side in thread process; do
    dir=$([ "$side" = thread ] && echo "$OLD" || echo "$ROOT")
    (cd "$dir" && .venv/bin/python -m bench.run --label "gil-ab-$side-$i" --quick >/dev/null)
    cp "$dir/results/gil-ab-$side-$i/summary.csv" "$OUT/$side-$i.csv"
    cp "$dir/results/gil-ab-$side-$i/env.txt" "$OUT/$side-$i.env.txt"
    rm -rf "$dir/results/gil-ab-$side-$i"
    echo "== $side, round $i" | tee -a "$OUT/tables.md"
    python3 - "$OUT/$side-$i.csv" <<'PY' | tee -a "$OUT/tables.md"
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    s, e = float(r["step_s"]), float(r["engine_s"])
    print(f"{r['config']:18} steps {r['steps']:>4}  ms/step {s / int(r['steps']) * 1e3:6.1f}  outside C++ forward {(s - e) / s:5.1%}")
PY
  done
done
