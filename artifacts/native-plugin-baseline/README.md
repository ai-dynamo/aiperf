# Native plugin baseline

This directory records the immutable pre-plugin baseline captured from commit
`caa3ff6fcf20ffe36a7704abe16274bedadbb9fb` on paper-rig. The canonical
comparison inventory is `rust/benchmarks/plugin-parity.yaml`.

The capture used Rust/Cargo 1.98.0, Python 3.11.2, and tini 0.19.0 without
changing the rig-provided `CARGO_BUILD_JOBS=144`, `CARGO_INCREMENTAL=1`, or
`CARGO_TARGET_DIR=/cargo-target`. The host exposed CPUs `0-143` and NUMA nodes
`0-2`; mock servers were pinned to CPUs `0-3` and clients to disjoint CPU sets.

`package-topology.json` is the measured Cargo-metadata/tree census consumed by
the package-separation tasks. `allocation-probe.json` records TLS-scoped
mimalloc allocation counts and bytes for all five required request paths.
The gitignored `evidence-manifest.json` authenticates the complete local
`review1i` evidence tree. The inventory, topology, allocation receipt, and
manifest are committed into the working tree as one crash-durable local
generation; no external release or remote read-back is part of Task 1.
The pre-measurement `review1f` attempt is also rejected: Cargo correctly
refused its incomplete measurement projection under `--locked`. The corrected
`review1g` projection passed that lock check, but its engine clean build then
encountered ENOSPC when the shared target filesystem reached 100%. Exact cache
cleanup receipts were retained, the filesystem was expanded to 1 TiB, and the
`review1h` capture completed all build matrices before exposing that background
owned commands had received `/dev/null` instead of their heredoc producers.
Its exact failure ledger is retained, and the next fresh replacement suffix is
`review1i-final`.

The experiment source is one materialized tree: the complete `git archive` of
`caa3ff6f` overlaid once with the exact allowlist recorded in
`identity/measurement-source-projection.txt`. Every build, runtime, metadata,
allocation, and digest command uses that same tree and its effective overlaid
Cargo.lock. The base archive and base lock are separately labeled provenance;
the inventory, report, and generated evidence are excluded to avoid a
self-reference. The checked-in capture harness enforces singleton admission,
checks that the effective tree and lock remain unchanged through teardown,
and extracts and verifies the completed local bundle against its embedded
manifest.

The inventory digest is calculated after replacing every
`canonical_inventory_digest` and the outer `experiment_identity_digest` value
with `blake3:` followed by 64 zeroes. The resulting digest is embedded in the
identity JSON before that identity receives its own digest. This avoids a cycle
while retaining one byte-exact canonicalization rule.
