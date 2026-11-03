"""Minimal OpenAI-compatible completions server: POST /v1/completions (with "stream": true as
server-sent events), GET /v1/models, and GET /metrics (Prometheus text format).

The engine (scheduler + model) runs in its own process and steps back to back; the HTTP server
runs in this one. They talk through two multiprocessing queues: requests and aborts go in,
each step's sampled tokens come out, and a dispatcher thread here routes them to per-request
queues read by the handler threads. With both in one process, the handler threads (decoding
and writing tokens) and the engine loop competed for the GIL, and a third to a half of each
step's time was spent outside the forward pass (README, "What I learned").

    python -m mini_serve.server --port 8000 [--static] [--no-prefix-cache] [--chunked-prefill N] [--num-blocks N]
"""

import argparse
import itertools
import json
import multiprocessing as mp
import queue
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tokenizers import Tokenizer

from .blocks import BlockPool
from .engine import DEFAULT_WEIGHTS, Runner
from .llm import LLM
from .metrics import RequestMetrics, render
from .scheduler import Scheduler, Sequence

MODEL = "SmolLM2-135M"


def engine_main(args, inq, outq):
    """The engine process. In: ("add", id, prompt, max_tokens, eos_id, temperature), ("abort", id),
    ("stats",). Out: ("ready", eos_id), ("tokens", [(id, token, finished), ...]) once per step,
    ("error", id, message) for a rejected request, ("stats", dict)."""
    pool = BlockPool(args.num_blocks, args.page_size, prefix_cache=not (args.static or args.no_prefix_cache))
    sched = Scheduler(pool, max_num_seqs=args.max_num_seqs, max_prefill_tokens=args.max_prefill_tokens,
                      static=args.static, chunk_size=args.chunked_prefill)
    llm = LLM(Runner(args.num_blocks, args.page_size, weights=args.weights), sched)
    outq.put(("ready", llm.runner.eos_id))
    live = {}
    while True:
        try:  # block only when idle, and exit if the server process is gone
            while True:
                try:
                    msg = inq.get(block=not sched.has_work(), timeout=None if sched.has_work() else 1.0)
                except queue.Empty:
                    if not sched.has_work() and not mp.parent_process().is_alive():
                        return
                    raise
                if msg[0] == "add":
                    _, sid, prompt, max_tokens, eos, temperature = msg
                    seq = Sequence(prompt, max_tokens=max_tokens, eos_id=eos, temperature=temperature, id=sid)
                    try:
                        sched.add(seq)
                        live[sid] = seq
                    except ValueError as e:
                        outq.put(("error", sid, str(e)))
                elif msg[0] == "abort":
                    if (seq := live.pop(msg[1], None)) is not None:
                        sched.abort(seq)
                else:
                    outq.put(("stats", dict(llm.stats, preemptions=sched.preemptions, prefix_hit_blocks=pool.hits,
                                            running=len(sched.running), waiting=len(sched.waiting),
                                            free_blocks=pool.num_free(), num_blocks=pool.num_blocks,
                                            step_hist=llm.step_hist.snapshot())))
        except queue.Empty:
            pass
        out = llm.step()
        if out is not None:
            batch, tokens = out
            outq.put(("tokens", [(seq.id, tok, seq.finished) for (seq, _, _), tok, sampled
                                 in zip(batch.items, tokens, batch.sample) if sampled]))
            for seq, _, _ in batch.items:
                if seq.finished:
                    live.pop(seq.id, None)


class EngineClient:
    """This process's handle on the engine process."""

    def __init__(self, args):
        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=engine_main, args=(args, self.inq, self.outq), daemon=True)
        self.proc.start()
        msg = self.outq.get(timeout=300)
        assert msg[0] == "ready", msg
        self.eos_id = msg[1]
        self.boxes = {}              # request id -> queue of tokens, then None (or an error string)
        self.stats_box = queue.Queue()
        self.stats_lock = threading.Lock()  # one stats request in flight, so answers cannot cross
        self.ids = itertools.count()
        threading.Thread(target=self._dispatch, daemon=True).start()

    def submit(self, prompt, max_tokens, eos_id, temperature):
        sid, box = next(self.ids), queue.Queue()
        self.boxes[sid] = box
        self.inq.put(("add", sid, prompt, max_tokens, eos_id, temperature))
        return sid, box

    def abort(self, sid):
        self.boxes.pop(sid, None)
        self.inq.put(("abort", sid))

    def stats(self):
        with self.stats_lock:
            self.inq.put(("stats",))
            return self.stats_box.get(timeout=30)

    def _dispatch(self):
        while True:
            msg = self.outq.get()
            if msg[0] == "tokens":
                for sid, tok, finished in msg[1]:
                    if (box := self.boxes.get(sid)) is not None:
                        box.put(tok)
                        if finished:
                            box.put(None)
                            del self.boxes[sid]
            elif msg[0] == "error":
                if (box := self.boxes.pop(msg[1], None)) is not None:
                    box.put(msg[2])
            else:
                self.stats_box.put(msg[1])


class Detokenizer:
    """Turns sampled token ids into text deltas. Only the ids not yet sent are decoded: with the
    byte-level BPE of this model every token is a fixed byte string, so decoding a suffix gives
    exactly the suffix of the text, as long as it starts on a character boundary. Ids are held
    back while they end in an incomplete UTF-8 sequence (decoded as U+FFFD)."""

    def __init__(self, tok):
        self.tok, self.pending = tok, []

    def push(self, tid, final=False):
        self.pending.append(tid)
        text = self.tok.decode(self.pending)
        if text.endswith("�") and not final:
            return ""
        self.pending = []
        return text


def make_handler(engine, tok):
    vocab = tok.get_vocab_size()  # not per token: each call costs ~3.5 ms with the GIL held
    metrics = RequestMetrics()

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
            if self.path == "/metrics":
                body = render(engine.stats(), metrics.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/stats":  # counters for the benchmark (not part of the OpenAI API)
                return self._json(200, engine.stats())
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
            self._error(404, "not found")

        def do_POST(self):
            if self.path != "/v1/completions":
                return self._error(404, "not found")
            t0 = time.perf_counter()
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                prompt = req["prompt"]
                ids = prompt if isinstance(prompt, list) else tok.encode(prompt).ids
                if not all(isinstance(t, int) and 0 <= t < vocab for t in ids):
                    raise ValueError("prompt token ids out of range")
                max_tokens = int(req.get("max_tokens", 16))
                temperature = float(req.get("temperature", 1.0))
                stream = bool(req.get("stream", False))
                eos = -1 if req.get("ignore_eos", False) else engine.eos_id
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
                metrics.finished("bad_request")
                return self._error(400, f"bad request: {e}")

            sid, box = engine.submit(list(ids), max_tokens, eos, temperature)
            rid, created = f"cmpl-{sid}", int(time.time())

            def chunk(text, finish):
                return {"id": rid, "object": "text_completion", "created": created, "model": MODEL,
                        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}

            def finish_reason(last):
                return "stop" if last == eos else "length"

            first = box.get()
            if isinstance(first, str):  # rejected by the scheduler
                metrics.finished("rejected")
                return self._error(400, first)
            t_first = t_last = time.perf_counter()
            n = 1

            def done(outcome):
                metrics.finished(outcome, len(ids), ttft=t_first - t0, e2e=t_last - t0,
                                 tpot=(t_last - t_first) / (n - 1) if n > 1 else None)

            if not stream:
                out = [first]
                while (t := box.get()) is not None:
                    out.append(t)
                    t_last, n = time.perf_counter(), n + 1
                done("ok")
                body = chunk(tok.decode(out), finish_reason(out[-1]))
                body["usage"] = {"prompt_tokens": len(ids), "completion_tokens": len(out),
                                 "total_tokens": len(ids) + len(out)}
                return self._json(200, body)

            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                detok, t = Detokenizer(tok), first
                while t is not None:
                    nxt = box.get()
                    if nxt is not None:
                        t_last, n = time.perf_counter(), n + 1
                    self._sse(chunk(detok.push(t, final=nxt is None), finish_reason(t) if nxt is None else None))
                    t = nxt
                self._sse("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                done("ok")
            except (BrokenPipeError, ConnectionResetError):
                engine.abort(sid)
                done("aborted")

        def _sse(self, obj):
            data = obj if isinstance(obj, str) else json.dumps(obj)
            payload = f"data: {data}\n\n".encode()
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()

    return Handler


def make_server(argv=None):
    """Parses server flags, starts the engine process; returns the (not yet serving) HTTP server."""
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
    ap.add_argument("--chunked-prefill", type=int, metavar="N",
                    help="mix decode and prefill chunks, at most N tokens per step (off by default)")
    args = ap.parse_args(argv)

    engine = EngineClient(args)
    tok = Tokenizer.from_file(str(args.weights / "tokenizer.json"))
    ThreadingHTTPServer.request_queue_size = 1024  # listen backlog; the default 5 drops bursts of connects
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine, tok))
    srv.daemon_threads = True
    srv.mode = "static" if args.static else "continuous"
    return srv


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so the engine process is stopped too
    srv = make_server()
    host, port = srv.server_address[:2]
    print(f"mini-serve on http://{host}:{port} ({srv.mode})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
