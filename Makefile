BUILD ?= build
PY := .venv/bin/python
LABEL ?= local

.PHONY: help build weights setup test reference serve bench clean
help: ## list targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

build: ## configure and build libminiserve (fetches kv-cache-bench @ 1be44cd)
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD) -j

weights: ## download SmolLM2-135M (~270 MB, pinned revision, sha256-checked)
	./scripts/fetch_weights.sh

$(PY): requirements.txt
	uv venv -q --python 3.12 .venv
	uv pip install -q --python $(PY) -r requirements.txt
	@touch $(PY)

setup: $(PY) ## Python venv: numpy, tokenizers

test: build weights setup ## scheduler and block pool tests, then logits vs transformers
	$(PY) -m unittest -v tests.test_scheduler tests.test_logits

reference: weights ## regenerate tests/data/reference.npz with transformers on the CPU
	uv venv -q --python 3.12 .venv-ref
	uv pip install -q --python .venv-ref/bin/python -r requirements-ref.txt
	.venv-ref/bin/python scripts/make_reference.py

serve: build weights setup ## OpenAI-compatible server on :8000
	$(PY) -m mini_serve.server

bench: build weights setup ## naive vs continuous on the same load, writes results/$(LABEL)/
	$(PY) -m bench.run --label $(LABEL)

clean: ## remove the build tree
	rm -rf $(BUILD)
