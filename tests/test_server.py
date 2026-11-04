"""HTTP API: /v1/completions with and without streaming, errors, /v1/models. Needs the weights."""

import http.client
import json
import re
import socket
import threading
import time
import unittest

from tokenizers import Tokenizer

from mini_serve.engine import DEFAULT_WEIGHTS
from mini_serve.server import make_server


SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})? (-?[0-9.e+-]+|\+Inf|NaN)$')


def parse_metrics(text):
    """Prometheus text format -> {(name, labels): value}; asserts every sample's family was
    declared with # TYPE first and every line is well formed."""
    types, out = {}, {}
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            types[name] = kind
        elif line.startswith("# HELP ") or not line:
            continue
        else:
            m = SAMPLE.match(line)
            assert m, f"malformed line: {line!r}"
            name, labels, value = m.groups()
            family = re.sub(r"_(bucket|sum|count)$", "", name) if name not in types else name
            assert family in types, f"no # TYPE for {name}"
            out[(name, labels or "")] = float(value)
    return out, types


class ServerTest(unittest.TestCase):
    FLAGS = []

    @classmethod
    def setUpClass(cls):
        cls.srv = make_server(["--port", "0", "--num-blocks", "64"] + cls.FLAGS)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.tok = Tokenizer.from_file(str(DEFAULT_WEIGHTS / "tokenizer.json"))

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def post(self, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json", "Connection": "close"})
        return c.getresponse()

    def complete(self, prompt, n):
        r = self.post({"prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True})
        self.assertEqual(r.status, 200)
        return json.loads(r.read())

    def stream(self, prompt, n):
        r = self.post({"prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True, "stream": True})
        self.assertEqual(r.status, 200)
        self.assertEqual(r.getheader("Content-Type"), "text/event-stream")
        events = [line[6:].strip() for line in r.read().decode().splitlines() if line.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        return [json.loads(e) for e in events[:-1]]

    def test_stream_equals_non_stream(self):
        for prompt in ["The capital of France is", "Привет, меня зовут", "def add(a, b):"]:
            full = self.complete(prompt, 24)
            chunks = self.stream(prompt, 24)
            self.assertEqual(len(chunks), 24)  # one event per token
            self.assertEqual("".join(c["choices"][0]["text"] for c in chunks), full["choices"][0]["text"])
            self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
            self.assertTrue(all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1]))
            self.assertEqual(full["usage"]["completion_tokens"], 24)
            self.assertEqual(full["usage"]["prompt_tokens"], len(self.tok.encode(prompt).ids))

    def test_token_id_prompt(self):
        ids = self.tok.encode("The capital of France is").ids
        self.assertEqual(self.complete(ids, 8)["choices"][0]["text"],
                         self.complete("The capital of France is", 8)["choices"][0]["text"])

    def test_concurrent_requests_match_sequential(self):
        prompts = [f"Item {i}: the best tool for" for i in range(6)]
        alone = [self.complete(p, 16)["choices"][0]["text"] for p in prompts]
        together = [None] * len(prompts)

        def go(i):
            together[i] = self.complete(prompts[i], 16)["choices"][0]["text"]
        threads = [threading.Thread(target=go, args=(i,)) for i in range(len(prompts))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(together, alone)

    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request("GET", path, headers={"Connection": "close"})
        r = c.getresponse()
        return r, r.read().decode()

    def test_metrics_match_load_and_stats(self):
        r, text = self.get("/metrics")
        self.assertEqual(r.status, 200)
        self.assertTrue(r.getheader("Content-Type").startswith("text/plain; version=0.0.4"))
        before, _ = parse_metrics(text)

        self.complete("The capital of France is", 8)
        self.complete("def add(a, b):", 6)
        self.stream("Привет, меня зовут", 5)
        self.assertEqual(self.post({"max_tokens": 3}).status, 400)
        prompt_tokens = sum(len(self.tok.encode(p).ids) for p in
                            ["The capital of France is", "def add(a, b):", "Привет, меня зовут"])

        after, types = parse_metrics(self.get("/metrics")[1])
        stats = json.loads(self.get("/stats")[1])
        d = lambda name, labels="": after[(name, labels)] - before[(name, labels)]
        self.assertEqual(d("mini_serve_requests_total", '{outcome="ok"}'), 3)
        self.assertEqual(d("mini_serve_requests_total", '{outcome="bad_request"}'), 1)
        self.assertEqual(d("mini_serve_prompt_tokens_total"), prompt_tokens)
        self.assertEqual(d("mini_serve_generation_tokens_total"), 8 + 6 + 5)
        for h in ("time_to_first_token_seconds", "request_seconds", "time_per_output_token_seconds"):
            self.assertEqual(d(f"mini_serve_{h}_count"), 3)
        # Engine counters are the same numbers /stats reports (nothing ran in between).
        self.assertEqual(after[("mini_serve_generation_tokens_total", "")], stats["sampled_tokens"])
        self.assertEqual(after[("mini_serve_preemptions_total", "")], stats["preemptions"])
        self.assertEqual(after[("mini_serve_prefix_cache_hit_blocks_total", "")], stats["prefix_hit_blocks"])
        for kind in ("prefill", "decode"):
            self.assertEqual(after[("mini_serve_steps_total", f'{{kind="{kind}"}}')], stats[f"{kind}_steps"])
            self.assertEqual(after[("mini_serve_step_rows_total", f'{{kind="{kind}"}}')], stats[f"{kind}_rows"])
        self.assertEqual(after[("mini_serve_step_seconds_count", "")], stats["prefill_steps"] + stats["decode_steps"])
        # Idle server: nothing running or waiting, every block free.
        self.assertEqual(after[("mini_serve_running_sequences", "")], 0)
        self.assertEqual(after[("mini_serve_waiting_sequences", "")], 0)
        self.assertEqual(after[("mini_serve_kv_blocks_free", "")], after[("mini_serve_kv_blocks_total", "")])
        # Histogram buckets are cumulative and end at _count.
        for name, kind in types.items():
            if kind != "histogram":
                continue
            buckets = [v for (n, _), v in after.items() if n == name + "_bucket"]
            self.assertEqual(buckets, sorted(buckets), name)
            self.assertEqual(after[(name + "_bucket", '{le="+Inf"}')], after[(name + "_count", "")], name)

    def open_request(self, body):
        """Sends a completion request on a raw socket and returns the socket, unread."""
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        data = json.dumps(body).encode()
        sock.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: %d\r\n\r\n" % len(data) + data)
        return sock

    def wait_stats(self, cond, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            stats = json.loads(self.get("/stats")[1])
            if cond(stats):
                return stats
            time.sleep(0.05)
        self.fail(f"engine state not reached: {stats}")

    def test_client_disconnect_frees_blocks(self):
        """A client that goes away mid-stream, mid non-streaming request, or while its request is
        still queued: the engine drops the request and the pool is whole again, long before the
        900 tokens it asked for would have been generated."""
        aborted = lambda: parse_metrics(self.get("/metrics")[1])[0][("mini_serve_requests_total", '{outcome="aborted"}')]
        before = aborted()
        body = {"prompt": "Once upon a time", "max_tokens": 900, "temperature": 0, "ignore_eos": True}
        a = self.open_request(dict(body, stream=True))
        self.assertTrue(a.recv(4096).startswith(b"HTTP/1.1 200"))  # sent after its first token
        b = self.open_request({"prompt": [100] * 1010, "max_tokens": 1})  # all 64 blocks: waits behind a
        self.wait_stats(lambda s: s["running"] == 1 and s["waiting"] == 1)
        b.close()                                            # gone while queued
        self.wait_stats(lambda s: s["running"] == 1 and s["waiting"] == 0)
        a.close()                                            # gone mid-stream
        self.wait_stats(lambda s: s["running"] == 0 and s["free_blocks"] == s["num_blocks"])
        c = self.open_request(body)                          # non-streaming: nothing is written until the end
        self.wait_stats(lambda s: s["running"] == 1)
        c.close()
        self.wait_stats(lambda s: s["running"] == 0 and s["free_blocks"] == s["num_blocks"])
        deadline = time.monotonic() + 5
        while aborted() - before < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(aborted() - before, 3)

    def test_errors(self):
        self.assertEqual(self.post({"max_tokens": 3}).status, 400)                       # no prompt
        self.assertEqual(self.post({"prompt": [10 ** 9], "max_tokens": 3}).status, 400)  # bad token id
        self.assertEqual(self.post({"prompt": "x", "max_tokens": 10 ** 6}).status, 400)  # larger than the pool
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/v1/models", headers={"Connection": "close"})
        self.assertEqual(json.loads(c.getresponse().read())["data"][0]["id"], "SmolLM2-135M")


class ChunkedPrefillServerTest(ServerTest):
    """The same API tests with prompts split into 4-token chunks: partial chunks must send nothing."""
    FLAGS = ["--chunked-prefill", "4"]


if __name__ == "__main__":
    unittest.main()
