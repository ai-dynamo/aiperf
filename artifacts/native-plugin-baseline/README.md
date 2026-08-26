# Native plugin baseline

This directory records the immutable pre-plugin baseline captured from commit
`caa3ff6fcf20ffe36a7704abe16274bedadbb9fb` on paper-rig. The canonical
comparison inventory is `rust/benchmarks/plugin-parity.yaml`.

The capture used Rust/Cargo 1.98.0, Python 3.11.2, and tini 0.19.0 without
changing the rig-provided `CARGO_BUILD_JOBS=144`, `CARGO_INCREMENTAL=1`, or
`CARGO_TARGET_DIR=/cargo-target`. The host exposed CPUs `0-143` and NUMA nodes
`0-2`; mock servers were pinned to CPUs `0-3` and clients to disjoint CPU sets.

`package-topology.json` is the measured Cargo-metadata/tree census consumed by
the package-separation tasks. `allocation-probe.json` records the 100,000-record
exporter run and its mimalloc process statistics. Raw build logs, release
artifacts, environment identity, scenario configurations, and run summaries live
under the ignored `artifacts/native-plugin-baseline/raw/` evidence directory on
paper-rig. Their relative paths and BLAKE3 identities are frozen in the YAML.

The inventory digest is calculated after replacing every
`canonical_inventory_digest` value with `blake3:` followed by 64 zeroes. This
avoids a self-referential digest while retaining one byte-exact canonicalization
rule.
