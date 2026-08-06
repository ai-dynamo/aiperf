#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal HTTP server that captures raw prompt content from benchmark clients.

Accepts OpenAI-compatible /v1/chat/completions requests, writes each raw payload
to a JSONL file, and returns a synthetic streaming response so the client doesn't
error out.

Usage::

    uv run python tools/capture_server.py --out /tmp/prompts.jsonl --port 18000

Then run your benchmark client against http://localhost:18000.
Each JSONL line contains the full request payload with a ``_seq`` counter added.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

_out_file = None
_count = 0

_MAX_BODY = 64 * 1024 * 1024  # 64 MiB

MODELS_RESPONSE = json.dumps(
    {
        "object": "list",
        "data": [{"id": "capture-model", "object": "model"}],
    }
).encode()


def _sse_chunk(completion_id: str, content: str) -> bytes:
    data = json.dumps(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ],
            "usage": None,
        }
    )
    return f"data: {data}\n\n".encode()


def _sse_done(completion_id: str, prompt_tokens: int) -> bytes:
    data = json.dumps(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 4,
                "total_tokens": prompt_tokens + 4,
            },
        }
    )
    return f"data: {data}\n\ndata: [DONE]\n\n".encode()


class CaptureHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence access logs
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(MODELS_RESPONSE)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global _count

        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_response(411)
            self.end_headers()
            return
        try:
            length = int(raw_length)
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return
        if length < 0:
            self.send_response(400)
            self.end_headers()
            return
        if length > _MAX_BODY:
            self.send_response(413)
            self.end_headers()
            return

        body = self.rfile.read(length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        if not isinstance(payload, dict):
            self.send_response(400)
            self.end_headers()
            return

        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            self.send_response(400)
            self.end_headers()
            return

        stream = payload.get("stream", False)

        if _out_file is not None:
            record = {**payload, "_seq": _count}
            _out_file.write(json.dumps(record) + "\n")
            _out_file.flush()

        _count += 1
        if _count % 1000 == 0:
            print(f"  captured {_count} requests", file=sys.stderr)

        content = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )
        prompt_tokens = max(1, len(content) // 4)

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(_sse_chunk(completion_id, "ok"))
            self.wfile.write(_sse_done(completion_id, prompt_tokens))
        else:
            resp = json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 1,
                        "total_tokens": prompt_tokens + 1,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        required=True,
        metavar="FILE",
        help="Output JSONL file — one captured request per line",
    )
    ap.add_argument("--port", type=int, default=18000, metavar="PORT")
    ap.add_argument("--host", default="127.0.0.1", metavar="HOST")
    args = ap.parse_args()

    global _out_file
    server = HTTPServer((args.host, args.port), CaptureHandler)
    print(f"Capture server listening on {args.host}:{args.port}", file=sys.stderr)
    print(f"Writing to {args.out}", file=sys.stderr)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(
        fd, 0o600
    )  # enforce on existing files too — O_CREAT mode is ignored when file exists
    with open(fd, "w") as f:
        _out_file = f
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    print(f"\nCaptured {_count} requests → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
