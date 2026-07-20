# Round 3 Fix E — Node resolved-bounds parity

## Outcome

The Node verifier now consumes the canonical `resolveScene` snapshot emitted by
`compile-decks.ts`. IR node checks join authored nodes to snapshot
`resolvedBounds`, connector checks consume snapshot path/direction data, and
resolver diagnostics retain source locations. The raw `geomOf` /
`resolveEndpoint` reconstruction path was removed from `geometry.mjs`; its
remaining geometry implementation is the explicitly synthetic, deck-independent
curved-router matrix.

The parity test now passes an actual serialized resolver snapshot to
`verifyPackageIr` while leaving authored stack/circle dimensions unresolved. It
therefore guards that intrinsic stack reflow and circle sizing reach the Node
consumer through the snapshot bridge rather than through pre-materialized roots.

## Residual gaps

- Snapshot serialization does not currently emit connectors for every legacy
  `sdk.Edge` mode. The full IR verifier reports 139
  `resolved-connector-missing` errors instead of reconstructing those routes in
  JavaScript.
- Forwarding all canonical resolver diagnostics exposes existing deck debt. The
  full run currently ends with 441 errors and 321 warnings; most errors beyond
  the 139 connector omissions are existing resolver ownership diagnostics.
- Static snapshots do not encode timeline playhead state, so cue-window sampling
  remains local in `geometry.mjs`.

## Verification

- `npm --prefix apps/explainers test -- src/flow/dev-tools/verify-geometry.test.ts`
  — pass, 5 tests.
- `npm --prefix apps/explainers test -- scripts/flow-verifier/ir.test.mjs`
  — pass, 2 tests.
- `npm --prefix apps/explainers run flow-verifier -- --ir-only`
  — fails with the residual 441 errors / 321 warnings described above.
