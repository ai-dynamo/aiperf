# AIPerf Flow CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current `FLOW_CLI_VERSION` stub into a deterministic `aiperf-flow` binary exposing `format`, `check`, `build`, `inspect`, and `capabilities` over the language, compiler, and schema packages' ready APIs.

**Architecture:** Keep Commander argument handling thin and put command behavior in testable service functions. Commands read `.flow` text, call `parseDocument`, `formatDocument`, `compileSource`, or `packFlow` directly, and select an exported schema capability manifest. `build` emits only the deterministic `PackedFlow` files; it does not package, invoke, or depend on a renderer.

**Tech Stack:** TypeScript 7, Node.js ESM, Commander 15, Vitest 4, Node `fs/promises`, `@aiperf/flow-language`, `@aiperf/flow-compiler`, and `@aiperf/flow-schema`.

## Global Constraints

- Scope is `apps/aiperf-flow/packages/cli/**` plus this plan.
- `apps/aiperf-flow/preview/**` is forbidden: do not read, modify, import, test, or use it as a fixture.
- The CLI is independent of renderer architecture. Do not import `@aiperf/flow-runtime`, emit HTML/runtime assets, or encode assumptions about Canvas, SVG, React, semantic twins, playback, or site layout.
- Use the current public APIs exactly:
  - `parseDocument(source: string, sourceName: string): Result<DocumentAst>`;
  - `formatDocument(document: DocumentAst): string`;
  - `compileSource(request: CompileRequest): Result<FlowIr>`;
  - `packFlow(ir: FlowIr, sourceName: string): PackedFlow`.
- `CompileRequest` requires `source`, `sourceName`, `capabilities`, and `strict`.
- Capability choices come from the ready schema exports `FOUNDATION_CAPABILITIES` and `P0_CAPABILITIES`; do not duplicate descriptor data in the CLI.
- Preserve `Result<T>` diagnostics. Human diagnostics go to stderr; `--json` produces stable JSON suitable for CI.
- Exit `0` on success, `1` for source/compiler/check failures, and `2` for command usage or filesystem failures. Set `process.exitCode`; do not call `process.exit()`.
- `format --write` must replace a source atomically. `format --check` must never modify files.
- `build` writes only `PackedFlow.files`, rejects absolute/traversing packed paths and symlink escapes, requires an absent or empty output directory unless `--clean` is set, and leaves no partial output after failure.
- Two builds from identical source, source name, strictness, and capability manifest must produce byte-identical output trees.
- Add the NVIDIA SPDX copyright and Apache-2.0 license header to every new source file.
- Do not add examples or generated scripts.
- Do not create a git commit unless explicitly requested.
- Activate the project environment before every repository command:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.

---

## Current Package Baseline

- `packages/cli/src/index.ts` currently exports only `FLOW_CLI_VERSION = 1`.
- `packages/cli/package.json` already declares Commander, compiler, and runtime dependencies and maps `aiperf-flow` to `dist/main.js`.
- Add direct dependencies on `@aiperf/flow-language` and `@aiperf/flow-schema`; remove the unused direct runtime dependency.
- The CLI TypeScript config currently includes only `src`, so tests remain under `test/` and run through a CLI-local Vitest config.
- The compiler already exports `compileSource`, `packFlow`, `canonicalJson`, `CompileRequest`, `PackedFlow`, and related pack types.
- The schema already exports sorted `FOUNDATION_CAPABILITIES` and `P0_CAPABILITIES` manifests.
- Do not depend on an example `.flow` file being present. Define a compact valid source fixture under `packages/cli/test/fixtures.ts`.

## File Map

- Modify `apps/aiperf-flow/packages/cli/package.json`: dependencies and test configuration.
- Modify `apps/aiperf-flow/packages/cli/src/index.ts`: public service/type exports; retain `FLOW_CLI_VERSION`.
- Create `apps/aiperf-flow/packages/cli/src/main.ts`: Commander program and exit-code adaptation.
- Create `apps/aiperf-flow/packages/cli/src/types.ts`: command requests/results, capability profile, inspect view, and exit codes.
- Create `apps/aiperf-flow/packages/cli/src/io.ts`: source reads, atomic writes, safe output-directory publication.
- Create `apps/aiperf-flow/packages/cli/src/diagnostics.ts`: stable human and JSON rendering.
- Create `apps/aiperf-flow/packages/cli/src/manifests.ts`: resolve `foundation` and `p0` to schema manifests.
- Create `apps/aiperf-flow/packages/cli/src/format.ts`: format service.
- Create `apps/aiperf-flow/packages/cli/src/check.ts`: compile-only validation service.
- Create `apps/aiperf-flow/packages/cli/src/build.ts`: compile, pack, and artifact publication service.
- Create `apps/aiperf-flow/packages/cli/src/inspect.ts`: AST, IR, pack manifest, and packed-file views.
- Create `apps/aiperf-flow/packages/cli/src/capabilities.ts`: capability manifest output.
- Create `apps/aiperf-flow/packages/cli/vitest.config.ts`: Node test environment and package aliases.
- Create `apps/aiperf-flow/packages/cli/test/fixtures.ts`: valid, invalid, warning-producing, and noncanonical source fixtures.
- Create `apps/aiperf-flow/packages/cli/test/*.test.ts`: service and binary tests.

---

### Task 1: Shared CLI foundation

**Interfaces**

- `CapabilityProfile = "foundation" | "p0"`.
- `CommandResult = { exitCode: 0 | 1 | 2; stdout: string; stderr: string }`.
- `resolveCapabilityManifest(profile): CapabilityRegistryManifest`.
- Stable diagnostic ordering by source, start offset, severity, code, then message.

- [ ] Add failing tests for manifest selection, unknown-profile rejection, human diagnostic ranges/repairs, stable JSON diagnostics, UTF-8 source loading, atomic replacement, and packed-path traversal rejection.
- [ ] Add the direct language and schema dependencies, remove the runtime dependency, and add a CLI Vitest config using the existing workspace package alias pattern.
- [ ] Implement shared types, manifest resolution, diagnostics, and filesystem helpers without importing runtime or preview code.
- [ ] Re-export public command types and services from `src/index.ts` while preserving `FLOW_CLI_VERSION`.
- [ ] Run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && npm test -w @aiperf/flow-cli -- diagnostics.test.ts io.test.ts manifests.test.ts`
  and expect all selected tests to pass.

---

### Task 2: `format`

**Command:** `aiperf-flow format <sources...> [--check | --write] [--json]`

**Interface:** `formatCommand(request: FormatRequest): Promise<CommandResult>`.

- [ ] Write failing tests proving default mode prints canonical source, `--check` returns `1` for noncanonical text and `0` for canonical text, `--write` atomically rewrites every valid input, parse errors preserve `parseDocument` diagnostics, and `--check` plus `--write` is usage error `2`.
- [ ] Implement one parse per file with `parseDocument(source, sourceName)` followed by `formatDocument(result.value)` only when parsing succeeds.
- [ ] Ensure multi-file default output is deterministic and unambiguous, `--check` and `--write` emit no formatted payload, and no file is changed if any input fails to parse.
- [ ] Run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && npm test -w @aiperf/flow-cli -- format.test.ts`
  and expect the format tests to pass.

---

### Task 3: `check`

**Command:** `aiperf-flow check <sources...> [--strict] [--capabilities <foundation|p0>] [--json]`

**Interface:** `checkCommand(request: CheckRequest): Promise<CommandResult>`.

- [ ] Write failing tests for valid foundation and P0 sources, parse/link/type/capability errors, strict warning promotion, non-strict warning success, multiple-file diagnostic aggregation, stable JSON, and unknown capability profile as usage error `2`.
- [ ] Implement exactly one `compileSource({ source, sourceName, capabilities, strict })` call per input using `resolveCapabilityManifest`.
- [ ] Aggregate every returned diagnostic deterministically; return `1` when any compile result is not `ok`, otherwise `0` while retaining successful warnings.
- [ ] Run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && npm test -w @aiperf/flow-cli -- check.test.ts`
  and expect the check tests to pass.

---

### Task 4: `build`

**Command:** `aiperf-flow build <source> --out <directory> [--strict] [--clean] [--capabilities <foundation|p0>] [--json]`

**Interface:** `buildCommand(request: BuildRequest): Promise<CommandResult>`.

- [ ] Write failing tests proving `compileSource` feeds `packFlow`, every `PackedFlow.files` entry is written byte-for-byte at its relative path, compile failure writes nothing, nonempty output is rejected without `--clean`, clean replacement is atomic, unsafe paths/symlinks are rejected, and repeated builds are byte-identical.
- [ ] Implement source read → `compileSource` → `packFlow(result.value, sourceName)` → safe staged directory publication.
- [ ] Emit only `PackedFlow.files` (`flow.manifest.json`, scene chunks, and transcript for current packs); do not synthesize HTML, JavaScript, CSS, renderer configuration, or other site files.
- [ ] Make human success output report output path, pack id, content hash, and file count; make `--json` report the same fields in stable JSON.
- [ ] Run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && npm test -w @aiperf/flow-cli -- build.test.ts`
  and expect the build tests to pass.

---

### Task 5: `inspect`

**Command:** `aiperf-flow inspect <source> --view <ast|ir|manifest|files> [--strict] [--capabilities <foundation|p0>]`

**Interface:** `inspectCommand(request: InspectRequest): Promise<CommandResult>`.

- [ ] Write failing tests for all four views, invalid source diagnostics, strict compilation behavior, unknown view/profile usage errors, and deterministic output.
- [ ] Implement `ast` with `parseDocument`; serialize its `DocumentAst` deterministically without compiling.
- [ ] Implement `ir` with `compileSource`; serialize the validated `FlowIr` using compiler `canonicalJson`.
- [ ] Implement `manifest` and `files` with `compileSource` then `packFlow`; `manifest` prints `packed.manifest`, while `files` prints sorted `{ path, mediaType, hash, byteLength }` metadata and never dumps binary file contents.
- [ ] Keep stdout payload-only and stderr diagnostic-only so every view can be piped safely.
- [ ] Run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && npm test -w @aiperf/flow-cli -- inspect.test.ts`
  and expect the inspect tests to pass.

---

### Task 6: `capabilities`

**Command:** `aiperf-flow capabilities [--profile <foundation|p0>] [--json]`

**Interface:** `capabilitiesCommand(request: CapabilitiesRequest): CommandResult`.

- [ ] Write failing tests proving the default profile is `foundation`, P0 selection returns `P0_CAPABILITIES`, human rows are sorted by id and include id/version/kind/description, and JSON equals the selected exported manifest without reshaping.
- [ ] Implement the command solely through `resolveCapabilityManifest`; do not discover capabilities from runtime registries or renderer modules.
- [ ] Return usage error `2` for an unknown profile and `0` for either exported manifest.
- [ ] Run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && npm test -w @aiperf/flow-cli -- capabilities.test.ts`
  and expect the capability tests to pass.

---

### Task 7: Binary wiring and verification

- [ ] Write failing binary tests for help, version, all five subcommands, option conflicts, stdout/stderr separation, and exit codes; assert no `preview` command appears.
- [ ] Implement `src/main.ts` with a shebang and Commander registrations that delegate to the five services and assign `process.exitCode`.
- [ ] Confirm package build emits `dist/main.js`, declarations, and sourcemaps, and that the existing `bin` mapping needs no change.
- [ ] Run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && npm test -w @aiperf/flow-cli`
  and expect all CLI tests to pass.
- [ ] Run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && npm run build -w @aiperf/flow-cli`
  and expect TypeScript compilation to pass.
- [ ] Run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && node packages/cli/dist/main.js --help`
  and verify exactly `format`, `check`, `build`, `inspect`, and `capabilities` are listed.
- [ ] Run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && npm run flow:check`
  and expect the full workspace gate to pass.
- [ ] Search CLI source and tests for forbidden coupling:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow && rg '@aiperf/flow-runtime|preview/|Canvas|SVG|React|site\\.js|index\\.html' packages/cli`
  and expect no matches.

## Dependency Order

```text
Task 1
  ├── Task 2: format
  ├── Task 3: check
  ├── Task 4: build
  ├── Task 5: inspect
  └── Task 6: capabilities
Tasks 2–6 → Task 7
```

Tasks 2–6 are independently reviewable after the shared foundation. None depends on renderer work, runtime bundle availability, browser preview fixtures, or renderer architecture decisions.

## Out of Scope

- Any file under `apps/aiperf-flow/preview/**`.
- Runtime/static-site packaging, browser serving, renderer selection, screenshots, video, or frame export.
- A `preview`, `serve`, `watch`, `migrate`, or `schema` command.
- Changes to language, compiler, schema, runtime, examples, or renderer packages.
- New `.flow` examples or usage scripts.
