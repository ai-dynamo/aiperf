# YAML Config — Future Goals

This document captures **scoped-out** design ideas from prior YAML-config feature brainstorms. Each entry has motivation, a sketch, design questions still open, and prior-art references. None of these are committed to a release; they are durable idea-storage so we don't redo the design conversation when a user request resurrects one.

When a future-goal entry graduates to a real feature, it gets its own spec under `.claude/specs/YYYY-MM-DD-<topic>-design.md`, the entry below shrinks to a one-line pointer, and the spec assumes responsibility for the design.

---

## Inline-dataset extensions (deferred from `.claude/specs/2026-05-10-inline-dataset-design.md`)

The v1 inline-dataset feature shipped just two YAML changes: a new `records:` field on `FileDataset` (alternative to `path:`), with multi-pool dict-of-lists support for `random_pool`. The brainstorm raised several adjacent ideas that we deliberately punted to keep v1 tight. They are recorded here in roughly the order a user is likely to ask for them.

### `generate:` directive — programmatic record building

**Motivation.** Inline `records:` is great for hand-curated prompt sets but gets unwieldy when you want N parameterized prompts. Use case: "32 prompts of length-N built from a template," or "100 trace records with computed timestamps." Today this is achievable only by pre-rendering YAML out-of-band.

**Sketch.**

```yaml
benchmark:
  dataset:
    type: file
    format: single_turn
    generate:
      count: 32
      record:
        text: "Sample {{ index }} on {{ topic }}"
        output_length: "{{ 50 + (index % 10) * 50 }}"
      vars:
        topic: physics
```

- **Loop variable `index`** (0-indexed). Chosen over `i` after prior-art review — Ansible's `item` collision tax and Helm's `$index`+meaningful-name best-practice both argue for a longer, namespaced loop var.
- **Per-block `vars:`** override top-level `variables:`. Precedence (highest wins): loop var > `vars:` > top-level `variables:`. Collision between `vars:`/`variables:` and `index` raises a config error at load (don't silently shadow).
- **`count:`** accepts an int or a Jinja-evaluating string. Bounds `1 <= count <= 1_000_000`.
- **Probe-time validation**: at config load (`aiperf config validate` and `expand`), render `record:` once with `{index: 0, **vars, **variables}`, surface Jinja `UndefinedError` / `TemplateSyntaxError` immediately, and run the rendered probe through the format's per-record validator. Helm's #1 regret per published pitfalls writeups is debug-at-render — AIPerf already has Jinja in-process, so this is cheap.
- **Mutual exclusion**: `generate:` is a third source alongside `path:` and `records:`, also XOR. Mixing static `records:` with `generate:` was considered (additive concat) but rejected as confusing — if you want both, hand-write the static records as the leading entries of `vars:` and use a conditional in the template.

**Open questions.**

- Should the schema be `generate: {count, record}` (today) vs. `generate: oneOf({count, record}, {for_each, record})` (Argo-style `withSequence` vs `withItems`)? Argo's split is clean and avoids Terraform's count-only regret. v1 ships `count:` only, but the schema must not lock us out — name the field `generate:` (not `generate_count:`) and the `for_each:` variant slots in later as a sibling key inside the same block.
- Should `generate:` accept a *list* of generate blocks (multiple loops concatenated)? Reject for v1; users can rerender via two configs if they need it.
- Should `generate:` work for multi-pool `random_pool`? Defer until asked. v1 would constrain `generate:` to flat-list formats only.

**Prior-art references.**

- [Argo Workflows `withSequence` / `withItems`](https://argo-workflows.readthedocs.io/en/latest/walk-through/loops/) — clean separation of count- vs list-driven loops.
- [Helm `range $i, $v := list`](https://helm.sh/docs/chart_template_guide/variables/) — both index and value bound in one breath.
- [Terraform `for_each` + `each.key`/`each.value`](https://developer.hashicorp.com/terraform/language/expressions/dynamic-blocks); [Terraform issue #23288](https://github.com/hashicorp/terraform/issues/23288) — five-year-old regret about no index attribute on `dynamic` blocks.
- [Ansible `loop_control.loop_var`](https://docs.ansible.com/projects/ansible/latest/playbook_guide/playbooks_loops.html) — bare `item` collision is a famous pain point; argues for namespaced/long loop-var names.
- [CUE comprehensions](https://cuelang.org/docs/concept/how-cue-works-with-yaml/) and [Jsonnet comprehensions](https://jsonnet.org/learning/tutorial.html) — pure-functional list-builders; nice but require a real language.

### Jinja random helpers (uniform, deterministic)

**Motivation.** Inside `generate:` (and anywhere else `{{ }}` evaluates), users want to inject randomness — pick a random topic per iteration, draw a random output length, etc. The use case the brainstorm landed on: `"Tell me about {{ random_choice(topics) }}"`.

**Sketch.**

```yaml
schemaVersion: "2.0"
random_seed: 42

variables:
  topics: [physics, chemistry, biology]

benchmark:
  dataset:
    type: file
    format: single_turn
    generate:
      count: 100
      record:
        text: "Tell me about {{ random_choice(topics) }}"
        output_length: "{{ random_int(50, 500) }}"
```

**Helpers** (mirror Python's `random` module — least-surprise):

| Helper | Shape |
|---|---|
| `random_choice(items)` | uniform pick from a list |
| `random_int(min, max)` | inclusive integer |
| `random_float(min, max)` | uniform float |

The `random_choice` headline is sufficient for v1-of-this-deferred-feature; `random_int` and `random_float` are natural companions but can ship alone.

**Open questions.**

- **Deterministic seed derivation strategy** — two options:
  - **A. Per-render-site, indexed inside `generate:`.** `rng_seed = stable_hash((random_seed, yaml_path_of_field, index_or_None))`. A `random_choice` inside `phases.profiling.duration` always renders with the same RNG → same value across runs. Inside `generate.record.text` with `count: 100`, each iteration mixes `index` in → 100 distinct deterministic draws. Different fields get different streams, so a `random_int` in `record.output_length` doesn't perturb `record.text`. Re-ordering an unrelated field doesn't shuffle the dataset.
  - **B. Single global stream.** One `random.Random(random_seed)` advances across the whole config render in document order. Simpler; but reordering or adding any field shifts every subsequent draw → fragile reproducibility.
  - **Decision (when this graduates):** strategy A. Same lesson Helm/Terraform learned the hard way — seed-by-path keeps reproducibility stable across config edits.
- **Weighted variants?** `random_choice(items, weights=[...])` mirrors `random.choices`. The brainstorm dropped weighted from v1 of this deferred feature to keep API surface minimal; can re-add as a single keyword argument when asked.
- **Object-bundled weighted lists?** A sugar helper `pick(weighted_list)` for `[{item, weight}, ...]` shapes was considered. Defer; verbose Jinja `map(attribute=...)` works in the meantime.
- **`random_normal(mean, stddev)`?** Matches synthetic-prompt distribution shape. Defer until a user asks.

### Runtime weighted sampling

**Motivation.** Different from Jinja-time random choice. Use case: "60% of requests draw prompt A, 30% B, 10% C." This belongs to the *sampler*, not the *record builder* — the loader produces a flat record list; the sampler picks one per request.

**Sketch.**

- Optional `weight: float >= 0` field on every record schema (`single_turn`, `multi_turn`, `random_pool`, traces). Default `1.0`. Loader normalizes; absolute scale is irrelevant.
- New sampling mode `sampling: weighted_random` (sibling of `sequential` / `random` / `shuffle`).
- Validation: if `sampling: weighted_random` and all weights are zero or absent, error. If `sampling: random` is set but any record has an explicit non-default weight, warn (probable user mistake).
- Composes with `path:` (same `weight:` field in JSONL records on disk), `records:`, and (when it ships) `generate:` (template can compute weight per iteration).
- Trace formats keep their own ordering semantics; `weight:` is a config-error when combined with sequential trace replay.

**Open questions.**

- Per-pool weights for multi-pool `random_pool`? Wrap each pool: `{pool_a: {weight: 3, items: [...]}}`. Layers cleanly on top of per-record weights without breaking the bare-list shape.
- Should weights live on the record (per-entry) or in a parallel list (`weights: [6, 3, 1]`)? Per-entry — index-fragile parallel lists don't compose with `generate:` and are error-prone to maintain.

### Mixing `path:` with `records:` (additive concat)

**Motivation.** "Start from a known on-disk corpus, add a few hand-crafted extras inline." A v1 brainstorm option, deferred because the corner cases multiply (sampling/limit semantics across mixed sources, trace timestamps when concat'ing two trace streams).

**Sketch.** When both are present: file records first, then inline records. Sampling/`entries:` count cap apply across the merged list. Trace formats reject this combination (timestamp re-basing would be implicit and surprising).

**Open questions.**

- Should the order be configurable (e.g. `inline_position: prepend|append`)? Probably not — one fixed order is enough; users who want the other order can swap to `path:`-only or `records:`-only.
- Multi-pool `random_pool` mixing? File directory + inline-dict-of-lists would have to merge by pool name — clean enough, but doubles the surface area. Defer.

### Optional wrapper schema for `records:`

**Motivation.** Argo's `ArtifactLocation.raw = {data: "..."}` lesson: wrapping inline data in a typed object even when it has a single field today lets you add `encoding:`, `schema_version:`, `validate:` later without a breaking change.

**Sketch.** Allow `records:` to accept either:

- Bare list (today): `records: [...]`
- Wrapped form: `records: {items: [...], schema_version: 1, encoding: utf-8}`

The wrapped form is reserved future-additive. v1 of inline-datasets ships bare-list only; this entry tracks the option and the prior-art that motivates keeping it open.

**Open questions.**

- Is the wrapper worth introducing before any of those reserved fields has a real use case? Probably no — premature surface area. Wait for the first concrete need (e.g. base64-encoded image records, or a schema-version stamp) and design the wrapper around that need instead of speculatively.

### Reserve `from:` as the future remote-source key

**Motivation.** `valuesFrom` (Helm/Flux/Argo CD) is a familiar k8s-flavored verb for "load from another source." If we ever want to pull dataset records from a URL or a remote artifact store, `from:` is the natural sibling of `path:` (local) and `records:` (inline).

**Sketch (when it ships).**

```yaml
dataset:
  type: file
  format: single_turn
  from:
    url: https://datasets.example.com/prompts.jsonl
    sha256: 4e2f...
    cache: ~/.aiperf/cache/datasets
```

**Open questions.**

- This is a different-enough feature (network I/O, caching, integrity) that it deserves its own spec when it graduates. The future-goal entry just reserves the field name.

---

## Prior-art notes (referenced from the inline-dataset spec)

The v1 brainstorm did a structured prior-art survey of dual-source YAML patterns and generate-loop DSLs. Persisting the lessons here so the next config-surface design can reuse them without re-running the survey.

### Dual-source patterns (file XOR inline)

- **OpenAPI Example Object (`value` XOR `externalValue`)** — two sibling fields, plain mutual exclusion, enforced by linters. Naming is asymmetric on purpose (`value` is the noun, `externalValue` is the modifier). Closest precedent for the `path:` ↔ `records:` split.
- **Tekton Workspace bindings (`configMap` / `secret` / `emptyDir` / ...)** — N sibling fields, "exactly one" validation webhook. Field name *is* the discriminator; no `type: configMap` redundancy.
- **Argo Workflows `ArtifactLocation`** — same shape as Tekton; `raw: {data: ...}` is the inline variant. Wrapping inline data in an object lets you add fields later non-breaking.

**Lesson:** with ≤6 sibling sources and a "exactly one" rule, sibling fields beat a `type:` discriminator. Adopted in v1.

### Generate-loop patterns

- **Helm `range $i, $v := list`** — both index and value bound in one breath, `$`-prefix marks user variables clearly. `until N` builtin handles count-based loops.
- **Terraform `for_each` + `each.key`/`each.value`** — stable keys (no churn on insert), namespaced `each.*`. Notable regret: no index attribute on dynamic-block iterator (open issue #23288 for 5+ years).
- **CUE comprehensions `for k, v in list`** — both index and value, optional `if` filter, comprehension *is* the value (no separate "loop directive" keyword). Statically validated.
- **Ansible `loop:` with `loop_control.loop_var`** — default loop var is `item`; collides constantly under nesting. **Lesson: short generic names like `item` or `i` produce real collisions; require/encourage renaming.** This is the single biggest argument for `index` over `i`.
- **Argo `withSequence: {count: N}` + `withItems: [...]`** — separate keys for count- vs list-driven loops; same loop variable. Future-additive `for_each:` for AIPerf would mirror this.

### Validation strategy that wins

Three-stage validation, not single late-binding crash:

1. **Parse-time** — structural mutual exclusion, type checks, bounds.
2. **Lint-time** — template renders with mock vars, Jinja syntax/undefined errors surface before run.
3. **Run-time** — only the actual data substitution.

Helm/Argo postmortems converge on this. v1 inline-datasets does parse-time and run-time; lint-time only matters once `generate:` ships.

### Sources

- [Argo Workflows Artifacts](https://argo-workflows.readthedocs.io/en/latest/walk-through/artifacts/)
- [Argo Workflows Loops](https://argo-workflows.readthedocs.io/en/latest/walk-through/loops/)
- [Tekton Workspaces](https://tekton.dev/docs/pipelines/workspaces/)
- [OpenAPI 3.1 — Example Object](https://spec.openapis.org/oas/v3.1.0)
- [Helm Variables](https://helm.sh/docs/chart_template_guide/variables/)
- [Helm Templating Pitfalls](https://medium.com/@mathumathiv247/helm-templating-pitfalls-i-wish-someone-warned-me-about-28414b587cd7)
- [Kustomize configMapGenerator](https://kubectl.docs.kubernetes.io/references/kustomize/kustomization/configmapgenerator/)
- [Ansible loop docs](https://docs.ansible.com/projects/ansible/latest/playbook_guide/playbooks_loops.html)
- [Terraform Dynamic Blocks](https://developer.hashicorp.com/terraform/language/expressions/dynamic-blocks)
- [Terraform issue #23288](https://github.com/hashicorp/terraform/issues/23288)
- [CUE / YAML](https://cuelang.org/docs/concept/how-cue-works-with-yaml/)
- [Jsonnet comprehensions](https://jsonnet.org/learning/tutorial.html)
- [Pkl for-generator amends limitation](https://github.com/apple/pkl/discussions/718)
- [GitHub Actions matrix](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs)
- [JSON Schema $ref vs inline](https://json-schema.org/understanding-json-schema/structuring)
