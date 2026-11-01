#!/usr/bin/env bash
# Downloads SmolLM2-135M (Apache-2.0) from Hugging Face at a pinned revision into weights/.
# ~270 MB. The weights are never committed.
set -euo pipefail
REPO=HuggingFaceTB/SmolLM2-135M
REV=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
SHA256=80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1  # model.safetensors
DIR="$(cd "$(dirname "$0")/.." && pwd)/weights/SmolLM2-135M"
mkdir -p "$DIR"
for f in config.json tokenizer.json model.safetensors; do
  [ -s "$DIR/$f" ] || curl -fL --retry 3 -o "$DIR/$f.part" "https://huggingface.co/$REPO/resolve/$REV/$f"
  [ -f "$DIR/$f.part" ] && mv "$DIR/$f.part" "$DIR/$f"
done
echo "$SHA256  $DIR/model.safetensors" | shasum -a 256 -c -
