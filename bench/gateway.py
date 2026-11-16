"""A local OpenAI-compatible endpoint for ``bench.compare``.

    python -m bench.gateway --port 8765 --log usage.jsonl

Other memory systems call a model to write. To compare them fairly this
puts one small server between them and the model provider:

* ``POST .../embeddings`` for a ``minishlab/...`` model is answered here
  with ``myelin.embedders.StaticEmbedder``, the embedder Myelin-AI's own
  runs use, so every system ranks with the same vectors.
* Everything else (``chat/completions``, ``models``, embeddings for any
  other model) is forwarded unchanged to ``MYELIN_BENCH_BASE_URL``, and
  the tokens and cost the provider reports are added up.

A client marks its calls by putting a tag in the path,
``http://127.0.0.1:8765/t/<tag>/v1``; ``GET /t/<tag>/usage`` returns the
totals for that tag. Standard library only; the embedder is loaded on
first use. Credentials are handled as in ``bench.llm``: the key in
``MYELIN_BENCH_API_KEY`` if there is one, otherwise none, and a key sent
by the client is dropped.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOCAL_PREFIX = "minishlab/"
# ``calls`` and the token counts are chat calls only. Embeddings computed
# here are counted in ``embedded_texts``; embeddings bought from the
# provider in ``embedding_tokens``. ``cost`` is everything the provider billed.
FIELDS = (
    "calls", "prompt_tokens", "completion_tokens", "cost", "embedded_texts", "embedding_tokens"
)


def split_tag(path: str) -> tuple[str, str]:
    """``/t/<tag>/v1/chat/completions`` -> (``<tag>``, ``chat/completions``)."""
    parts = path.split("?", 1)[0].strip("/").split("/")
    tag = ""
    if len(parts) >= 2 and parts[0] == "t":
        tag, parts = parts[1], parts[2:]
    if parts and parts[0] == "v1":
        parts = parts[1:]
    return tag, "/".join(parts)


def usage_of(body: bytes) -> dict[str, float]:
    """Tokens and cost from a provider's reply, plain JSON or a stream."""
    found: dict[str, Any] = {}
    try:
        found = json.loads(body).get("usage") or {}
    except (ValueError, AttributeError):
        for line in body.decode("utf-8", "replace").splitlines():
            if line.startswith("data: {") and '"usage"' in line:
                try:
                    found = json.loads(line[6:]).get("usage") or found
                except ValueError:
                    continue
    return {
        "prompt_tokens": float(found.get("prompt_tokens") or found.get("input_tokens") or 0),
        "completion_tokens": float(
            found.get("completion_tokens") or found.get("output_tokens") or 0
        ),
        "cost": float(found.get("cost") or 0.0),
    }


class Gateway:
    """State shared by the request threads."""

    def __init__(self, upstream: str, log_path: str = "", api_key: str = "") -> None:
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key
        self.totals: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()
        self._embedders: dict[str, Any] = {}
        self._log = open(log_path, "a", encoding="utf-8") if log_path else None  # noqa: SIM115

    def record(self, tag: str, **row: Any) -> None:
        with self._lock:
            total = self.totals.setdefault(tag, dict.fromkeys(FIELDS, 0.0))
            for key in FIELDS:
                total[key] += float(row.get(key) or 0)
            if self._log is not None:
                self._log.write(json.dumps({"time": time.time(), "tag": tag, **row}) + "\n")
                self._log.flush()

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if model not in self._embedders:
                from myelin.embedders import StaticEmbedder

                self._embedders[model] = StaticEmbedder(model)
            embedder = self._embedders[model]
            return embedder.encode_batch(texts)

    def embeddings_reply(self, request: dict[str, Any]) -> dict[str, Any]:
        """An OpenAI embeddings response computed locally."""
        given = request.get("input")
        texts = [given] if isinstance(given, str) else list(given or [])
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("input must be a string or a list of strings")
        vectors = self.embed(request["model"], texts)
        if request.get("encoding_format") == "base64":
            data: list[Any] = [
                base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode("ascii") for v in vectors
            ]
        else:
            data = vectors
        return {
            "object": "list",
            "model": request["model"],
            "data": [
                {"object": "embedding", "index": n, "embedding": v} for n, v in enumerate(data)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    def forward(self, method: str, rest: str, body: bytes | None) -> tuple[int, str, bytes]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.upstream}/{rest}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return (
                    response.status,
                    response.headers.get("Content-Type", "application/json"),
                    response.read(),
                )
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type", "application/json"), exc.read()
        except OSError as exc:
            return 502, "application/json", json.dumps({"error": {"message": str(exc)}}).encode()


def handler_for(gateway: Gateway) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # quiet
            pass

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, "application/json", json.dumps(payload).encode())

        def do_GET(self) -> None:
            tag, rest = split_tag(self.path)
            if rest == "usage":
                with gateway._lock:
                    totals = dict(gateway.totals.get(tag) or dict.fromkeys(FIELDS, 0.0))
                self._json(200, totals)
                return
            self._send(*gateway.forward("GET", rest, None))

        def do_POST(self) -> None:
            tag, rest = split_tag(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                request = json.loads(body) if body else {}
            except ValueError:
                request = {}
            model = str(request.get("model") or "")
            started = time.monotonic()
            if rest == "embeddings" and model.startswith(LOCAL_PREFIX):
                try:
                    reply = gateway.embeddings_reply(request)
                except (ValueError, KeyError, ImportError) as exc:
                    self._json(400, {"error": {"message": str(exc)}})
                    return
                gateway.record(tag, path=rest, model=model, embedded_texts=len(reply["data"]))
                self._json(200, reply)
                return
            status, content_type, reply_body = gateway.forward("POST", rest, body)
            used = usage_of(reply_body) if status == 200 else {}
            if rest == "embeddings":
                used = {"embedding_tokens": used.get("prompt_tokens"), "cost": used.get("cost")}
            elif status == 200:
                used["calls"] = 1
            gateway.record(
                tag, path=rest, model=model, status=status,
                seconds=round(time.monotonic() - started, 3), **used,
            )
            self._send(status, content_type, reply_body)

    return Handler


def serve(port: int, log_path: str = "", upstream: str = "") -> ThreadingHTTPServer:
    """A server ready for ``serve_forever``; port 0 picks a free one."""
    gateway = Gateway(
        upstream or os.environ.get("MYELIN_BENCH_BASE_URL") or "https://api.openai.com/v1",
        log_path,
        os.environ.get("MYELIN_BENCH_API_KEY") or "",
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(gateway))
    server.daemon_threads = True
    server.gateway = gateway  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.gateway", description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log", default="", help="append one JSON line per call here")
    parser.add_argument("--upstream", default="", help="default: MYELIN_BENCH_BASE_URL")
    args = parser.parse_args(argv)
    server = serve(args.port, args.log, args.upstream)
    print(f"gateway on http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
