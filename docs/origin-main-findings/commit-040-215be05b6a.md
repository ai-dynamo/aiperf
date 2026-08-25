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
