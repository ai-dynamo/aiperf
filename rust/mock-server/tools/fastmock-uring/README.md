<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# fastmock-uring — io_uring twin of `fastmock`, for A/B microbenchmarking

`fastmock-uring` is a thread-per-core [monoio](https://crates.io/crates/monoio)
(io_uring) server that speaks the exact same fixed chat/models responses as the
std-only `../fastmock.rs`. It exists to answer one question: **is io_uring worth
it for the lowest-overhead loopback mock target, versus the blocking
thread-per-connection model?**

It is a standalone cargo project (its own empty `[workspace]`) so the `monoio` /
`io-uring` dependency tree stays out of the main aiperf workspace build.

```bash
cargo build --release            # in this directory
./target/release/fastmock-uring 8131 --cores 8   # 8 io_uring runtimes, SO_REUSEPORT on :8131
```

## The A/B result (TL;DR)

**The winner flips on the client's load pattern** — which is the whole point.

1. **Realistic load — 1 in-flight request per connection** (how every real
   HTTP/1.1 client behaves: browsers disabled pipelining; reqwest/curl/wrk/oha
   don't do it): **io_uring wins +27% to +54%**, biggest at low core counts. It
   amortizes per-request syscall cost and dodges thread-per-connection overhead.
2. **Deep client-side pipelining — 32 in-flight per connection: blocking wins
   ~29%**, because the app already gets 32 requests per `read()` and sends 32 per
   `write()`, so io_uring's submit/complete machinery is pure overhead. **This
   regime is unrepresentative** — treat pipelined numbers as a server "retirement
   ceiling", never as client-achievable RPS.
3. A weak client hides everything: the `reqwest`-based `examples/loadgen` (no
   pipelining) caps at ~650k rps — itself the bottleneck, ~3.7× below the servers'
   true ceiling. Use `../fastclient.rs`, which does >1.5M rps.

### Numbers — realistic (pipeline=1), 512 connections, server pinned to N cores

| server cores | `fastmock --procs N` (blocking) | `fastmock-uring --cores N` (io_uring) | io_uring advantage |
|-------------:|--------------------------------:|--------------------------------------:|:------------------:|
| 1 |   199,716 rps |   307,059 rps | **+54%** |
| 2 |   308,969 rps |   459,416 rps | **+49%** |
| 4 |   643,301 rps |   913,645 rps | **+42%** |
| 8 | 1,228,966 rps | 1,556,854 rps | **+27%** |

### Mechanism — 8 cores, pipeline × connections (why it flips)

| pipeline | conns | blocking | io_uring | winner |
|---------:|------:|---------:|---------:|:------:|
| 1  |   64 | 1,368,815 | 1,599,706 | io_uring +17% |
| 1  |  512 | 1,227,857 | 1,573,448 | io_uring +28% |
| 32 |   64 | 2,197,006 | 1,701,633 | blocking +29% |
| 32 | 1000 | 1,944,064 | 1,519,287 | blocking +28% |

Full write-up (setup, caveats, version pins) is saved as a durable finding at
`~/.claude/benchmark-findings/rust-io_uring-monoio-vs-blocking-threadperconn-http.md`.

## Reproducing

```bash
# 1. Build server A (blocking), server B (io_uring), and the monster client.
rustc -C opt-level=3 -C lto=fat -C codegen-units=1 ../fastmock.rs   -o /tmp/fastmock
rustc -C opt-level=3 -C lto=fat -C codegen-units=1 ../fastclient.rs -o /tmp/fastclient
cargo build --release                                              # fastmock-uring (here)

# 2. Server on cores 0..N-1, client on the rest (so the client is never the limiter).
taskset -c 0-3 /tmp/fastmock --procs 4 9001 &                      # or: fastmock-uring --cores 4 9001
taskset -c 4-31 /tmp/fastclient http://127.0.0.1:9001/v1/chat/completions \
  --connections 512 --duration 6 --pipeline 1                      # pipeline 1 = realistic
```

> **Gotcha:** run the benchmark with the sandbox **disabled**. A sandbox that
> namespaces the network makes `SO_REUSEPORT` binds intermittently fail with
> `EADDRINUSE`, which shows up as a crashed `--procs`/`--cores` server and bogus
> `0 rps` rows. Native (unsandboxed) execution binds cleanly.

## What's identical vs different

Identical: the fixed chat SSE + models JSON response bytes, the byte-buffer
request framing (find `\r\n\r\n`, read Content-Length body), keep-alive, per-port
SO_REUSEPORT. The **only** variable is the I/O engine — blocking `read`/`write`
on OS threads vs monoio's owned-buffer (`Rent`) io_uring submissions.
