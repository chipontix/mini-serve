"""HTTP API: /v1/completions with and without streaming, errors, /v1/models. Needs the weights."""

import http.client
import json
import threading
import unittest

from tokenizers import Tokenizer

from mini_serve.engine import DEFAULT_WEIGHTS
from mini_serve.server import make_server


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = make_server(["--port", "0", "--num-blocks", "64"])
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

    def test_errors(self):
        self.assertEqual(self.post({"max_tokens": 3}).status, 400)                       # no prompt
        self.assertEqual(self.post({"prompt": [10 ** 9], "max_tokens": 3}).status, 400)  # bad token id
        self.assertEqual(self.post({"prompt": "x", "max_tokens": 10 ** 6}).status, 400)  # larger than the pool
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/v1/models", headers={"Connection": "close"})
        self.assertEqual(json.loads(c.getresponse().read())["data"][0]["id"], "SmolLM2-135M")


if __name__ == "__main__":
    unittest.main()
