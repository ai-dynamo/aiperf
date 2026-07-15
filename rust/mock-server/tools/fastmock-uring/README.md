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

Two findings, both actionable:

1. **Per core, io_uring is ~1.7–1.9× the blocking model** — but only where the
   *server* is the bottleneck (≤2 cores here).
2. **Above ~4 cores the loopback client (the load generator) saturates first**
   (~650–700k rps on this box), so neither server is the limiter. For fastmock's
   actual job — being a non-bottleneck target — the std `fastmock --procs/--threads`
   already exceeds what a client can drive. **Reach for io_uring only if you must
   serve very high throughput from few cores.**

### Numbers (32-core box, server pinned to N cores, load generator pinned to the rest)

| server cores | `fastmock --procs N` (blocking, thread-per-conn) | `fastmock-uring --cores N` (io_uring) | io_uring speedup |
|-------------:|-------------------------------------------------:|--------------------------------------:|:----------------:|
| 1 | 135,671 rps | 263,403 rps | **1.94×** |
| 2 | 254,321 rps | 436,038 rps | **1.71×** |
| 4 | 531,262 rps | 698,413 rps | 1.31× (io_uring nearing client cap) |
| 8 | 583,033 rps | 613,048 rps | 1.05× (both client-capped) |

The plateau/regression from 4→8 cores (and io_uring's 698k@4 > 613k@8) is the
tell that the **load generator** — not the server — sets the ceiling above ~4
server cores: with more cores handed to the server, the generator has fewer, so
its cap drops. Interpret only the 1–2 core rows as a pure engine comparison.

## Reproducing

```bash
# 1. Build the three pieces.
cargo build --release                                             # fastmock-uring (here)
rustc -C opt-level=3 -C lto=fat -C codegen-units=1 ../fastmock.rs -o /tmp/fastmock
cargo build --release --example loadgen -p aiperf-mock-server     # from the workspace

# 2. Server on cores 0..N-1, load generator on the rest (so it isn't the bottleneck).
taskset -c 0-1 /tmp/fastmock --procs 2 9001 &                     # or: fastmock-uring --cores 2 9001
taskset -c 2-31 <workspace>/target/release/examples/loadgen \
  --url http://127.0.0.1:9001/v1/chat/completions --concurrency 600 --total 600000
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
