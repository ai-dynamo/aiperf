#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal HTTP server that captures raw prompt content from benchmark clients.

Accepts OpenAI-compatible /v1/chat/completions requests, writes each prompt
to a JSONL file byte-for-byte, and returns a synthetic streaming response so
the client doesn't error out.

Usage::

    python tools/capture_server.py --out /tmp/prompts.jsonl --port 18000

Then run your benchmark client against http://localhost:18000.
Each JSONL line contains the full messages array plus metadata.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

_out_file = None
_count = 0

MODELS_RESPONSE = json.dumps(
    {
        "object": "list",
        "data": [{"id": "capture-model", "object": "model"}],
    }
).encode()


def _sse_chunk(content: str) -> bytes:
    data = json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ],
            "usage": None,
        }
    )
    return f"data: {data}\n\n".encode()


def _sse_done(prompt_tokens: int) -> bytes:
    data = json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
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
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        messages = payload.get("messages", [])
        stream = payload.get("stream", False)

        # Write captured data
        if _out_file is not None:
            record = {
                "i": _count,
                "messages": messages,
                "model": payload.get("model"),
                "max_completion_tokens": payload.get("max_completion_tokens"),
                "stream": stream,
                "ignore_eos": payload.get("ignore_eos"),
            }
            _out_file.write(json.dumps(record) + "\n")
            _out_file.flush()

        _count += 1
        if _count % 1000 == 0:
            print(f"  captured {_count} requests", file=sys.stderr)

        # Count prompt tokens (rough estimate for response)
        content = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )
        prompt_tokens = max(1, len(content) // 4)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(_sse_chunk("ok"))
            self.wfile.write(_sse_done(prompt_tokens))
        else:
            resp = json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
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
    with open(args.out, "w") as f:
        _out_file = f
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    print(f"\nCaptured {_count} requests → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
