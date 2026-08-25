# Task 3 Independent Review

## Verdict

**APPROVED**

No Important, Minor, or blocking findings remain in Task 3 commit
`89dbbb91acc76fa786f342e662af779b40a2add5` relative to parent
`0e9998eabe`.

## Scope reviewed

This review was limited to Task 3: native console rendering, the GenAI-Perf v1
JSON histogram, and processed JSONL speculative-decode fields. The production
diff is confined to:

- `rust/runtime/src/export/console_txt.rs`
- `rust/runtime/src/export/genai_perf.rs`
- `rust/runtime/src/engine/records.rs`

No Task 4 exporter behavior is present in the commit. CSV production code and
the SGLang server-metrics implementation are unchanged.

## Evidence

### Console contract

- `rust/runtime/src/export/console_txt.rs:492-502` places `SpecDecode` after
  `Reasoning` and before `Default`.
- `rust/runtime/src/export/console_txt.rs:625-634` appends the histogram directly
  beneath the speculative-decode scalar table.
- `rust/runtime/src/export/console_txt.rs:842-879` suppresses empty/zero-total
  histograms, fills missing buckets, folds every key `>= 8`, treats even an
  authored zero-count high key as activating the cap, and emits the exact label
  and three-space bucket separators.
- `rust/runtime/src/export/console_txt.rs:898-909` emits the exact title
  `NVIDIA AIPerf: Spec Decode`.
- Percentage formatting follows the upstream Python `:.0f` behavior, including
  ties-to-even. The focused tests cover the worked example, gaps, `>=8` folding,
  a zero-count high key, and absence suppression.
- The existing SGLang speculative-decoding table is separate and unchanged;
  its rate-only, model/rank filtering, and distinct-series regression tests all
  pass in the adjacent exporter suite.

### GenAI-Perf v1 JSON and CSV

- `rust/runtime/src/export/genai_perf.rs:635-670` adds the full, uncapped
  `pooled_spec_decode_acceptance_histogram` only when nonempty.
- The typed serializer writes `BTreeMap<u64, u128>` directly rather than routing
  counts through `serde_json::Value`, preserving exact counts above
  `u64::MAX` at the JSON boundary.
- Focused tests prove full keys including `0`, `8`, and `12`, exact
  `18446744073709551616` serialization, absent suppression, and no histogram
  column or content in CSV. The broader JSON/CSV oracle tests remain green.

### Processed JSONL

- `rust/runtime/src/engine/records.rs:101-109` adds the optional canonical
  `spec_decode_acceptance` object with absence suppression.
- `rust/runtime/src/engine/records.rs:992-1012` copies the canonical typed value
  from terminal `RecordIngest`; the pre-existing native record-metric projection
  supplies all six per-record metrics.
- `rust/runtime/src/engine/records.rs:1305-1370` proves the canonical object and
  all six exact metric values, and proves both the object and every
  `spec_decode_*` metric are omitted when absent.

### Safety, compatibility, and cost

- The widened histogram never narrows through an intermediate JSON value.
- Console summation uses checked `u128` addition and suppresses an impossible
  overflow rather than wrapping or panicking.
- No non-finite number is newly admitted at a serialization boundary.
- JSONL clones the optional canonical DTO once per processed record; no new
  synchronization, allocation per token, logging, or async work was added to a
  transport hot path.
- `git diff --check 0e9998eabe 89dbbb91acc76fa786f342e662af779b40a2add5`
  passed. `cargo fmt --all --check` passed.

## Fresh verification

From `/mnt/4tb/aiperf-origin-port-013/rust` with
`RUSTC_WRAPPER=sccache` and
`CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013`:

- `cargo test -p aiperf-runtime spec_decode --lib --features engine -- --nocapture`
  — **30 passed, 0 failed**.
- `cargo test -p aiperf-runtime export:: --lib --features engine -- --nocapture`
  — **121 passed, 0 failed**.
- `cargo test -p aiperf-runtime engine::records::tests --lib --features engine -- --nocapture`
  — **10 passed, 0 failed**.

The runs emitted four unrelated existing warnings (`unused_mut`, `dead_code`);
none originates in the Task 3 diff.
