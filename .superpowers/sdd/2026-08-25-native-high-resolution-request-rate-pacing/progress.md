# SDD ledger — plan: docs/superpowers/plans/2026-08-25-native-high-resolution-request-rate-pacing.md

## Controller preflight

Plan base: `c03a786e0d` (design artifact); implementation base: exact merge `86a93aaec1`.

| Task | Files | Interface fit | Status |
|---|---|---|---|
| 1 | `timing/arrival.rs`, `request_rate.rs` | Pure target policy and parser tests are independent of runtime I/O. | completed (`ce88e7631f`) |
| 2 | `timing/arrival.rs`, `request_rate.rs` | Construction-time environment policy feeds allocation-free local target arithmetic; global gate untouched. | completed (`edaf58681a`) |
| 3 | `request_rate_real.rs`, specs/agent docs/`llms.txt` | Real `RequestRateWorkload` + `RealClock` evidence closes product semantics and documentation. | completed (`a70ac2df39`) |
| 4 | finding/campaign/review artifacts | Full-range review and verification consume completed implementation and exact ancestry. | completed (closure commit) |

Self-consistency: Task 1 defines the two interfaces Task 2 consumes; Task 2 is
required before Task 3's real-clock characterization; Task 4 depends on all
implementation and documentation commits. No task requires a file created by a
later task. One implementer at a time owns the worktree.

## Decisions and evidence

- The full-flow assignment is advance approval for the architectural design;
  no interactive design pause is required.
- `AIPERF_TIMING_HIGH_RES_TIMER` is Python-pacer-specific and does not bypass the
  native injected `Clock`.
- Global dense slots remain unchanged; only local/sharded renewal pacing gets
  bounded re-anchoring.
- Every Cargo command uses `/usr/bin/sccache` and
  `/mnt/4tb/aiperf-origin-port-045-target`.

## Task receipts

- Task 1 RED: `cargo test -p aiperf-runtime --features engine --lib bounded_reanchor`
  failed with 12 `E0425` errors for the intentionally absent helper/parser.
- Task 2 GREEN: pure boundary/default/range tests passed; the deterministic
  issuer-level oversleep characterization passed with bounded admissions
  `[250, 250, 450, 450]` versus zero-window `[250, 250, 500, 500]`.
- Task 3 debug: exact 5,000/5,000 completions in 1,052,579,898 ns at
  4,750.233 req/s.
- Task 3 release: exact 5,000/5,000 completions in 1,008,803,639 ns at
  4,956.366 req/s.
- Documentation: `check_agent_files_sync.py` and `check_docs_current.py` passed.
- Final focused matrix: request-rate library 12 passed; simulation integration
  7 passed; debug and release real-clock tests each completed exactly 5,000
  requests; runtime-package formatting and range whitespace passed.
- Wide matrix: no-engine library 1,785 passed / 1 pre-existing failure / 7
  ignored; engine library 2,342 passed / 5 pre-existing failures / 7 ignored.
  All-target engine Clippy and workspace formatting stop on verified unchanged
  baseline files, with no #45-path diagnostic.
- Independent two-pass Graham review of `f423b618da..a70ac2df39`: no Critical
  or Important findings; `GRAHAM APPROVED`.
