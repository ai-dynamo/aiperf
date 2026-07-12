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
- pending

self-review findings and remaining concerns:
- implemented hierarchical graph catalog primitives (tiers, parents/children, seam ports, edge ports, flow channels, scenes, audience depth/visibility, execution flavors, explicit built/planned status model).
- implemented integrity checks for all required graph-first failure classes, including planned/built evidence policy and Dynamo-online runner planned-only enforcement.
- concern: graph catalog content currently focuses on canonical validation coverage and scene grounding rather than full UX-oriented node granularity; follow-up Task 2+ can refine graph density and layout semantics.
- concern: the graph catalog data module uses explicit casting to preserve compatibility with existing UI types while introducing the new canonical model; this keeps behavior stable for Task 1 but could be tightened in later tasks.
