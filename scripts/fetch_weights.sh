#!/usr/bin/env bash
# Downloads SmolLM2-135M (Apache-2.0) at a pinned revision into the standard Hugging Face cache
# ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub), in huggingface_hub's own
# layout (blobs/ + snapshots/<rev>/ symlinks), so other tools reuse it. ~270 MB, sha256-checked.
# Prints the snapshot directory.
set -euo pipefail
REPO=HuggingFaceTB/SmolLM2-135M
REV=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
SHA256=80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1  # model.safetensors
HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
MODEL="$HUB/models--${REPO//\//--}"
SNAP="$MODEL/snapshots/$REV"
mkdir -p "$MODEL/blobs" "$SNAP"
for f in config.json tokenizer.json model.safetensors; do
  [ -e "$SNAP/$f" ] && continue
  url="https://huggingface.co/$REPO/resolve/$REV/$f"
  etag=$(curl -fsSI "$url" | tr -d '\r' | sed -n 's/^x-linked-etag: "\(.*\)"$/\1/Ip')
  [ -n "$etag" ] || { echo "no etag for $f" >&2; exit 1; }
  [ -s "$MODEL/blobs/$etag" ] || { curl -fsSL --retry 3 -o "$MODEL/blobs/$etag.part" "$url" && mv "$MODEL/blobs/$etag.part" "$MODEL/blobs/$etag"; }
  ln -s "../../blobs/$etag" "$SNAP/$f"
done
echo "$SHA256  $SNAP/model.safetensors" | shasum -a 256 -c - >&2
echo "$SNAP"
