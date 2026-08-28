# Native-plugin performance-parity artifacts

Result documents from the native-plugin performance-parity gate land here. One
file is one experiment: the `parity` binary
(`cargo run -p aiperf-plugin-perf --bin parity`) writes exactly one JSON
document per run, and never appends to or rewrites an existing one.

## Naming convention

```
<metric>-<experiment-id-prefix>-<timestamp>.json
```

- `<metric>` is the metric compared, such as `ttft_p50` or `e2e_p50`.
- `<experiment-id-prefix>` is the first 16 hex characters of the
  `identity.experiment_id` field inside the document.
- `<timestamp>` is the run's UTC instant, matching `identity.timestamp_utc`.

Example: `ttft_p50-3f9a21c07be45d18-epoch-seconds-1787000000Z.json`

The experiment id is BLAKE3 over the newline-joined static binary digest,
dynamic binary digest, `Cargo.lock` digest, harness digest, CPU model, memory
topology, toolchain version, timestamp, and metric. Because the binaries are
part of the address, rebuilding either side produces a new experiment id rather
than silently overwriting a previous verdict. A filename collision therefore
means the same experiment was published twice, which is a bug worth
investigating rather than a file worth replacing.

## What a document contains

Each document is self-contained, so a reader can recompute the verdict without
rerunning anything:

- `identity` — the frozen, content-addressed experiment identity.
- `warmup_pairs` — the discarded warmup pairs, retained for auditability.
- `retained_pairs` — the pairs the bound was computed from, in execution order.
- `bootstrap_lower_bound`, `point_estimate`, `bootstrap_resamples`,
  `bootstrap_seed` — the retention bound and everything needed to reproduce it.
- `cv_static`, `cv_dynamic` — sample stability, which must sit at or below 2%.
- `allocation_delta` — dynamic allocations minus static allocations, which must
  not be positive.
- `is_zero_loss` — the verdict, and the binary's exit status.

## Gate conditions

A run passes when all three hold:

1. `bootstrap_lower_bound >= 0.99` — the one-sided 95% lower bound on the
   `static / dynamic` retention ratio.
2. `cv_static <= 0.02` and `cv_dynamic <= 0.02` — the rig was stable enough to
   have measured anything.
3. `allocation_delta <= 0` — the dynamic build did not allocate more.

A failing run is published, not deleted. Rerunning a confirmed regression until
it passes is the laundering this gate exists to prevent.
