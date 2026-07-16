<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Microbenchmark tools — `fastmock`, `fastclient`, `fastmock-uring`

These are loose, single-file `rustc`-compiled helpers under `rust/mock-server/tools/`.
They are **not** registered cargo bins — you compile each directly with `rustc`.
Use them only for lowest-overhead loopback transport micro-benchmarks; for anything
realistic (multiple endpoints, latency modeling, telemetry, token counting), use the
full `aiperf-mock-server` (see the SKILL and the other references).

## `fastmock` — the ultra-minimal server

`rust/mock-server/tools/fastmock.rs` is a std-only TCP server that returns one fixed
streaming chat response. No routes, latency, or token behavior — just the absolute
lowest-overhead loopback target.

```bash
rustc -O rust/mock-server/tools/fastmock.rs -o /tmp/fastmock
/tmp/fastmock 8131                       # 127.0.0.1:8131, one accept thread (baseline)
/tmp/fastmock 8131 --threads 4           # 4 accept threads share one listener, same process
/tmp/fastmock 8131 --procs 4             # 4 SO_REUSEPORT processes on :8131 (kernel-balanced)
/tmp/fastmock 8131 --procs 4 --threads 2 # compose: 4 processes × 2 accept threads
```

`fastmock` scales two ways without any proxy hop (both `0` = auto = CPU count):

- `--threads M` — M concurrent `accept()` threads over one shared listener, lifting the
  single-accept-loop ceiling in one process with zero added latency.
- `--procs N` — N independent server processes bound to the same port via `SO_REUSEPORT`;
  the kernel spreads new connections across them (true multi-process sharing, no L4 proxy
  in the data path). The leader supervises the children and, on Linux, sets
  `PR_SET_PDEATHSIG` so they never orphan. Linux-only; elsewhere `--procs` degrades to a
  plain bind.

Prefer these over fronting `fastmock` with the `aiperf-mock-server --processes N` balancer:
that balancer re-execs `aiperf-mock-server` (not `fastmock`) and adds a byte-splicing hop
that undercuts `fastmock`'s whole reason to exist.

## `fastclient` — a monster load generator for the fast targets

`rust/mock-server/tools/fastclient.rs` is a std-only (`rustc`-compiled) HTTP/1.1 load
generator that blazes past the reqwest `examples/loadgen` (which caps ~650k rps and is
itself the bottleneck). Persistent keep-alive connections, response framing by probed byte
length (no per-request parsing) — >2M rps on loopback.

```bash
rustc -O rust/mock-server/tools/fastclient.rs -o /tmp/fastclient
/tmp/fastclient http://127.0.0.1:8131/v1/chat/completions --connections 512 --duration 6
```

`--pipeline` defaults to **1** (one in-flight per connection) — the honest setting, since
real HTTP/1.1 clients don't pipeline; express concurrency via `--connections`.
`--pipeline P>1` measures the server's raw *retirement ceiling* (batches syscalls in a way
real traffic won't) — label it as such, never quote it as client RPS.

## `fastmock-uring` — io_uring engine A/B

`rust/mock-server/tools/fastmock-uring/` is a monoio (io_uring) thread-per-core twin of
`fastmock` for A/B-ing the I/O engine. Result: at realistic `pipeline=1`, io_uring beats
blocking thread-per-connection **+27–54%**; under deep pipelining blocking wins
(unrepresentative). See its `README.md` and the durable finding at
`~/.claude/benchmark-findings/rust-io_uring-monoio-vs-blocking-threadperconn-http.md`.
