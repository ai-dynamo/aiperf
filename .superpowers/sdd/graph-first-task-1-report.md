status: DONE_WITH_CONCERNS

files changed:
- `apps/architecture-atlas/src/domain/architecture.ts`
- `apps/architecture-atlas/src/domain/integrity.ts`
- `apps/architecture-atlas/src/content/index.ts`
- `apps/architecture-atlas/src/content/scenes/graph-catalog.ts`
- `apps/architecture-atlas/src/domain/graph-catalog.test.ts`

tests written:
- `apps/architecture-atlas/src/domain/graph-catalog.test.ts`
  - validates required tiers/channels/flavors
  - validates nine approved Rust scenes are present
  - validates complete Tier-0 Python->result journey nodes
  - validates Dynamo-online representation (library seam built, runner pair planned)
  - validates new integrity failures (scene refs, port refs, evidence status rules, runner planned enforcement, hierarchy cycles)

RED command and observed expected failure:
- command:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run test -- src/domain/graph-catalog.test.ts`
- observed failure:
  - failed 10/10 tests
  - missing new schema exports (`tierSchema` undefined)
  - missing catalog fields (`graphNodes`, `graphEdges`, `graphScenes`)
  - schema rejected unrecognized graph keys

GREEN/full verification commands and exact results:
- focused:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run test -- src/domain/graph-catalog.test.ts`
  - result: `1 passed, 10 tests passed`
- full:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run validate:content && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run typecheck && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas test`
  - result:
    - `validate:content`: `Architecture Atlas content is valid: 25 components, 20 edges, 23 crates.`
    - `typecheck`: success
    - `npm test`: `12 passed, 99 tests passed`

commit hash(es):
- `6f40d5327`
- `fa30ee23c`
- `c5ad057fa`
- `0ab1c12a8`
- `dcc387cce`
- `789041ff0`

self-review findings and remaining concerns:
- implemented hierarchical graph catalog primitives (tiers, parents/children, seam ports, edge ports, flow channels, scenes, audience depth/visibility, execution flavors, explicit built/planned status model).
- implemented integrity checks for all required graph-first failure classes, including planned/built evidence policy and Dynamo-online runner planned-only enforcement.
- concern: graph catalog content currently focuses on canonical validation coverage and scene grounding rather than full UX-oriented node granularity; follow-up Task 2+ can refine graph density and layout semantics.
- concern resolved by the review fix: public graph catalog facts now use compile-time `satisfies` checks and Zod-parsed exports without unsafe casts.

## Review Fix — 2026-07-12

status: DONE

files changed:
- `apps/architecture-atlas/src/content/scenes/graph-catalog.ts`
- `apps/architecture-atlas/src/domain/integrity.ts`
- `apps/architecture-atlas/src/domain/graph-catalog.test.ts`
- `.superpowers/sdd/graph-first-task-1-report.md`

fixes:
- removed `dynamo_online` from every built Tier-0 runner journey node and edge;
- retained only `run_scheduled_backend_online` as a built Dynamo-online library seam and kept the runner backend/pair plus its edge planned with explicit design evidence;
- required explicit graph evidence roles, line ranges for source evidence, and rejected spec/design paths used as source for both nodes and edges;
- enforced reciprocal parent/child declarations and populated Tier-0 child IDs;
- added feature-gated built Dynamo-offline runner backend, SimClock, SteppableReplay, and report-gate nodes and edges;
- replaced all `as unknown as` catalog casts with compile-time `satisfies` checks and Zod-parsed public exports;
- required graph edge channels to match both endpoint port channels.

tests added:
- built Tier-0 journey excludes Dynamo online;
- complete Dynamo-offline feature-gated path exists;
- reciprocal hierarchy rejection and canonical Tier-0 child coverage;
- missing source line ranges rejected for nodes and edges;
- missing design role and design-as-source rejected for nodes and edges;
- incompatible edge/port channels rejected.

RED:
- command:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run test -- src/domain/graph-catalog.test.ts`
- observed:
  - `1 failed test file`
  - `11 failed, 10 passed`
  - failures exactly covered Dynamo-online built leakage, absent Dynamo-offline entities, one-way hierarchy, missing source/design evidence enforcement, and missing channel compatibility.

GREEN:
- focused command:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run test -- src/domain/graph-catalog.test.ts`
- focused result:
  - `1 passed test file`
  - `21 passed`
- full commands:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run validate:content`
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run typecheck`
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas test`
- full results:
  - `validate:content`: `Architecture Atlas content is valid: 25 components, 20 edges, 23 crates.`
  - `typecheck`: success
  - `npm test`: `12 passed test files, 110 passed tests`

fix commit hash(es):
- `c5ad057fa`

self-review:
- every built graph node and edge now has at least one explicit, line-ranged, non-design source reference;
- every planned graph node and edge has explicit design evidence;
- the catalog module is compile-time checked and runtime parsed without unsafe casts;
- no remaining Task 1 review concern identified.

## Source Drift Re-review Fix — 2026-07-12

status: DONE

source-grounded correction:
- commit `2e1aa0782` was preserved unchanged;
- the first-class `dynamo_online` visual flavor now selects the built, feature-gated `dynamo_offline` runner backend/pair with `replay_mode=online`;
- the built path is grounded in `DynamoReplayModeSpec::Online` and `DynamoOfflineExecutor::execute_scheduled`;
- the distinct `dynamo_online` backend ID / registered pair remains a separate planned node and edge with design evidence.

integrity fixes:
- scenes now reject edges whose source or target node is absent from the scene;
- every canonical scene is closed over all selected edge endpoints;
- dedicated Dynamo-online runner-pair entities with `delivery: runner_pair` must remain planned for both nodes and edges.

tests added:
- built online replay maps through `node.dynamo-offline-runner-backend`;
- online replay has feature-gated built status and exact line-ranged runner evidence;
- the dedicated runner backend/pair remains planned;
- scene endpoint closure rejects incomplete scenes and validates all canonical scenes;
- parameterized node/edge tests reject built dedicated Dynamo-online runner-pair facts.

RED:
- command:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run test -- src/domain/graph-catalog.test.ts`
- observed:
  - `1 failed test file`
  - `5 failed, 20 passed`
  - failures covered missing built online replay facts, absent scene endpoint closure, and missing edge-level planned-only enforcement.

GREEN:
- focused command:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run test -- src/domain/graph-catalog.test.ts`
- focused result:
  - `1 passed test file`
  - `25 passed`
- full commands:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run validate:content`
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run typecheck`
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas test`
- full results:
  - `validate:content`: `Architecture Atlas content is valid: 25 components, 20 edges, 23 crates.`
  - `typecheck`: success
  - `npm test`: `12 passed test files, 114 passed tests`

source-drift fix commit hash(es):
- `dcc387cce`

concerns:
- none identified.

## Final Dynamo Flavor Fix — 2026-07-12

status: DONE

fixes:
- added `dynamo_online` to every built shared Tier-0 Python-to-result journey node and edge because the visual flavor now represents the runner-reachable `replay_mode=online` path;
- retained the distinct `delivery: runner_pair` node and edge as planned-only facts for a future dedicated `dynamo_online` backend ID;
- corrected the library-helper copy to state that the existing feature-gated `dynamo_offline` runner pair invokes `run_scheduled_backend_online`.

tests:
- replaced the stale Tier-0 exclusion assertion with complete shared-journey inclusion checks;
- added explicit copy/status checks distinguishing the invoked library helper from the planned dedicated backend/pair.

RED:
- command:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run test -- src/domain/graph-catalog.test.ts`
- observed:
  - `1 failed test file`
  - `2 failed, 24 passed`
  - failures showed missing `dynamo_online` Tier-0 coverage and stale library-only copy.

GREEN:
- focused command:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run test -- src/domain/graph-catalog.test.ts`
- focused result:
  - `1 passed test file`
  - `26 passed`
- full commands:
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run validate:content`
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas run typecheck`
  - `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && npm --prefix /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/architecture-atlas test`
- full results:
  - `validate:content`: `Architecture Atlas content is valid: 25 components, 20 edges, 23 crates.`
  - `typecheck`: success
  - `npm test`: `12 passed test files, 115 passed tests`

final flavor fix commit hash(es):
- pending

concerns:
- none identified.
