"""Minimal OpenAI-compatible completions server: POST /v1/completions (with "stream": true as
server-sent events) and GET /v1/models.

One engine thread owns the scheduler and the model and runs steps back to back. HTTP handler
threads never touch the scheduler: they hand new requests (and aborts) to the engine thread
through a queue, and read their sequence's sampled tokens from a per-request queue.

    python -m mini_serve.server --port 8000 [--static] [--no-prefix-cache] [--num-blocks N]
"""

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tokenizers import Tokenizer

from .blocks import BlockPool
from .engine import DEFAULT_WEIGHTS, Runner
from .llm import LLM
from .scheduler import Scheduler, Sequence

MODEL = "SmolLM2-135M"


class EngineLoop(threading.Thread):
    def __init__(self, llm):
        super().__init__(daemon=True)
        self.llm = llm
        self.inbox = queue.Queue()   # ("add", seq) or ("abort", seq)
        self.outboxes = {}           # seq.id -> queue of sampled tokens, then None

    def submit(self, seq):
        box = queue.Queue()
        self.outboxes[seq.id] = box
        self.inbox.put(("add", seq))
        return box

    def run(self):
        sched = self.llm.scheduler
        while True:
            try:  # block only when idle
                while True:
                    op, seq = self.inbox.get(block=not sched.has_work())
                    self._handle(op, seq)
            except queue.Empty:
                pass
            out = self.llm.step()
            if out is None:
                continue
            batch, tokens = out
            for (seq, _, _), tok in zip(batch.items, tokens):
                box = self.outboxes.get(seq.id)
                if box is None:
                    continue
                box.put(tok)
                if seq.finished:
                    box.put(None)
                    del self.outboxes[seq.id]

    def _handle(self, op, seq):
        if op == "add":
            try:
                self.llm.scheduler.add(seq)
            except ValueError as e:
                self.outboxes.pop(seq.id).put(e)
        else:
            self.llm.scheduler.abort(seq)
            self.outboxes.pop(seq.id, None)


class Detokenizer:
    """Turns a growing list of token ids into text deltas. Holds text back while it ends in an
    incomplete UTF-8 sequence (decoded as U+FFFD) so multi-byte characters are never split."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0

    def push(self, tid, final=False):
        self.ids.append(tid)
        text = self.tok.decode(self.ids)
        if text.endswith("�") and not final:
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


def make_handler(loop, tok):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, msg):
            self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

        def do_GET(self):
            if self.path == "/stats":  # counters for the benchmark (not part of the OpenAI API)
                sched = loop.llm.scheduler
                return self._json(200, dict(loop.llm.stats, preemptions=sched.preemptions,
                                            prefix_hit_blocks=sched.pool.hits))
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
            self._error(404, "not found")

        def do_POST(self):
            if self.path != "/v1/completions":
                return self._error(404, "not found")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                prompt = req["prompt"]
                ids = prompt if isinstance(prompt, list) else tok.encode(prompt).ids
                if not all(isinstance(t, int) and 0 <= t < tok.get_vocab_size() for t in ids):
                    raise ValueError("prompt token ids out of range")
                max_tokens = int(req.get("max_tokens", 16))
                temperature = float(req.get("temperature", 1.0))
                stream = bool(req.get("stream", False))
                eos = -1 if req.get("ignore_eos", False) else loop.eos_id
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
                return self._error(400, f"bad request: {e}")

            seq = Sequence(list(ids), max_tokens=max_tokens, eos_id=eos, temperature=temperature)
            box = loop.submit(seq)
            rid, created = f"cmpl-{seq.id}", int(time.time())
            detok = Detokenizer(tok)

            def chunk(text, finish):
                return {"id": rid, "object": "text_completion", "created": created, "model": MODEL,
                        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}

            def tokens():
                while (t := box.get()) is not None:
                    if isinstance(t, Exception):
                        raise t
                    yield t

            def finish_reason():
                return "stop" if seq.output and seq.output[-1] == seq.eos_id else "length"

            if not stream:
                try:
                    out = list(tokens())
                except ValueError as e:
                    return self._error(400, str(e))
                text = tok.decode(out)
                body = chunk(text, finish_reason())
                body["usage"] = {"prompt_tokens": len(ids), "completion_tokens": len(out),
                                 "total_tokens": len(ids) + len(out)}
                return self._json(200, body)

            try:
                first = box.get()
                if isinstance(first, Exception):
                    return self._error(400, str(first))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                t = first
                while t is not None:
                    nxt = box.get()
                    text = detok.push(t, final=nxt is None)
                    self._sse(chunk(text, finish_reason() if nxt is None else None))
                    t = nxt
                self._sse("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                loop.inbox.put(("abort", seq))

        def _sse(self, obj):
            data = obj if isinstance(obj, str) else json.dumps(obj)
            payload = f"data: {data}\n\n".encode()
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()

    return Handler


def build(args):
    pool = BlockPool(args.num_blocks, args.page_size, prefix_cache=not (args.static or args.no_prefix_cache))
    sched = Scheduler(pool, max_num_seqs=args.max_num_seqs, max_prefill_tokens=args.max_prefill_tokens,
                      static=args.static)
    runner = Runner(args.num_blocks, args.page_size, weights=args.weights)
    return LLM(runner, sched)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-prefill-tokens", type=int, default=2048)
    ap.add_argument("--static", action="store_true", help="baseline: static batches, no prefix cache")
    ap.add_argument("--no-prefix-cache", action="store_true")
    args = ap.parse_args(argv)

    llm = build(args)
    tok = Tokenizer.from_file(str(args.weights / "tokenizer.json"))
    loop = EngineLoop(llm)
    loop.eos_id = llm.runner.eos_id
    loop.start()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(loop, tok))
    srv.daemon_threads = True
    print(f"mini-serve on http://{args.host}:{args.port} ({'static' if args.static else 'continuous'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
