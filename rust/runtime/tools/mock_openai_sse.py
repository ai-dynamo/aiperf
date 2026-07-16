#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone mock OpenAI-compatible streaming chat server for aiperf proofs.

Streams: one role-only chunk, `max_tokens` content chunks, a finish chunk, then
[DONE] -- exercising the client's content-delta filtering over real TCP.
"""

import contextlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        try:
            req = json.loads(body or b"{}")
        except Exception:
            req = {}
        max_tokens = int(req.get("max_tokens", 8))
        model = req.get("model", "mock")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(delta, finish):
            obj = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())

        emit({"role": "assistant"}, None)
        for _ in range(max_tokens):
            emit({"content": "tok"}, None)
        emit({}, "stop")
        self.wfile.write(b"data: [DONE]\n\n")
        with contextlib.suppress(Exception):
            self.wfile.flush()

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
