# Runtime Clippy Baseline Remediation Design

## Goal

Make `cargo clippy -p aiperf-runtime --all-targets --features engine -- -D warnings` pass without weakening lint policy or changing Harbor behavior.

## Scope

The strict command currently reports existing warnings in several independent runtime subsystems, including graph recording, agent replay, AgentX, datasets, engine tests, and execution support. This work corrects those findings in small module-local batches. Harbor behavior and its public contracts are regression-protected by the existing Harbor runtime, CLI, and real-Docker E2E suites.

## Approach

1. Capture the strict clippy inventory and group errors by independently buildable module area.
2. For each batch, write or adjust the smallest behavior test where a lint correction can affect behavior; apply the minimal idiomatic Rust change.
3. Run the affected tests and the strict clippy command. Do not add broad `allow` attributes or relax workspace lint configuration.
4. Obtain an independent Graham inspection for every committed batch before starting the next one. A rejected batch is repaired and re-reviewed before work proceeds.
5. After the final batch, rerun the complete strict command, Harbor runtime/engine/CLI/P0 suites, serial ignored real-Docker Harbor matrix, formatting/diff checks, documentation guards, and a full Graham branch review.

## Constraints

- Preserve public interfaces and serialized artifacts unless a lint correction demonstrably requires a compatible change.
- Keep each commit confined to one coherent runtime subsystem.
- Use existing project error types and `Clock` timing; do not introduce `unwrap` or `expect` in production code.
- No lint suppressions as a substitute for correcting behavior or code structure.
- Every batch must pass Graham review before the next batch is committed as accepted.

## Verification

The remediation is complete only when the strict clippy command exits zero, all existing Harbor acceptance commands remain green, and final Graham inspection reports no Critical or Important findings.
