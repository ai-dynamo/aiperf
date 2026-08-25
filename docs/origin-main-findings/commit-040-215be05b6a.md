# Commit 040 — `215be05b6a`

## Upstream intent

Upstream keeps `duration_e2e_ms`, `duration_ttft_ms`, and
`cached_tokens_reference` in every Baseten source projection. These values are
recorded outcomes, not replay controls: later fidelity analysis needs them even
when replay is open-loop or KV request hints are disabled.

The exact upstream diff changes one Python loader method and one Python unit
test. It adds no component-integration or end-to-end test. The unit test checks
the default, omit-KV-hints, and closed-loop projections, then proves all three
outcomes remain on loaded rows while unrelated large columns do not.

## Native applicability

The native Baseten loader at the campaign base already retains
`duration_e2e_ms` through its private loader/composer intermediate because
closed-loop delay derivation consumes it. It drops `duration_ttft_ms` and
`cached_tokens_reference` at `parse_row`, so neither value survives the first
native boundary and no later fidelity comparison is possible.

The native port must retain the two missing fields beside the existing E2E
duration, round-trip all three through `RawRow`, and attach them to the composed
turn as non-dispatching recorded-outcome metadata. The fields must never enter
the request body or affect scheduling. A missing source column remains absent.

## Upstream-to-native test map

| Upstream behavior | Native evidence |
| --- | --- |
| Default projection retains all recorded outcomes | Baseten loader unit coverage builds a real Parquet fixture and asserts exact E2E, TTFT, and cached-token values on the composed turn. |
| `omit_kv_hints` does not discard outcomes | The same unit path disables KV hints and asserts the request body has no hashes while recorded outcomes remain unchanged. |
| Closed-loop replay does not discard outcomes | The unit path disables open-loop replay, asserts continuation delay derivation, and asserts both turns retain their own outcomes. |
| Missing optional TTFT and cached-token values stay absent | A unit fixture with null optional values retains E2E while keeping TTFT and cached-token outcomes absent. |
| Values survive the public registry path | `rust/runtime/tests/baseten_outcome_fidelity.rs` loads a real Parquet file through `LoaderRegistry::with_builtin_formats`, freezes a `Dataset`, and inspects the public turn metadata. |

## Ancestry constraint

The campaign base does not contain tracker #39's native work. Because exact
upstream commit `215be05b6a534fb19b84bf83f711db2d20f5bea1` has upstream #39 as
its parent, a normal content merge would import #39's unrelated Python
columnar-loader changes. The closure merge therefore uses Git's `ours` tree
strategy: it has the exact upstream commit as second parent while its tree is
byte-identical to the completed native first parent. This records real
two-parent ancestry without cherry-picking or importing #39's pending work.

## Closure evidence

The native implementation landed in `ccb8c27c14`; the isolated-base compiler
prerequisite is `de4eccf95d`. Merge `1d20f63c51e7f0e12732d54d61996dc4dc577f71`
has parents `b2dbbb0da77755c3837843f5f3db1808057f1a0c` and exact upstream
`215be05b6a534fb19b84bf83f711db2d20f5bea1`. Its tree
`a1e14c7c5250a9bc78259e7e32ff94c9bb6a1830` exactly equals its first-parent
tree, and the base-to-merge diff for upstream's Python loader and unit test is
empty; tracker #39 content was not imported.

TDD first failed with `E0609` because `Turn::recorded_outcome` did not exist.
After implementation and review fixes, all 13 Baseten loader unit tests and the
one real-Parquet public-registry integration test passed. Runtime all-target
Clippy exited successfully with pre-existing warnings. The complete runtime
library run passed 1,777 tests and failed one unrelated existing version
snapshot (`0.12.0` actual versus `0.0.0` expected) in
`metrics_core::report::tests::v2_uses_type_specific_series_and_null_for_non_finite_tail`;
the port does not change that module. Scoped Rust formatting, docs-current, and
exact-range whitespace checks passed; workspace formatting reports one
unrelated existing wrap difference in `rust/cli/src/yaml.rs`.

The first Graham pass found and fixed an overstated evidence claim and an
unsafe test-fixture `Default`. Re-review of `106019c5a1..964c3bc32a` approved
the corrected production and test surface with no remaining finding. Review
receipts live under
`.superpowers/sdd/2026-08-25-native-baseten-outcome-fidelity/`.
