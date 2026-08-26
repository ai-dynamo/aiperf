# Native Verbatim System Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one exact, startup-acquired system prompt to native profile configuration, dataset identity/composition, and OpenAI/Anthropic production requests with full Rust parity evidence.

**Architecture:** Resolve inline/file authoring into one owned string before typed Dataset construction. Carry a distinct verbatim prompt through strict engine specs, inject it after format-specific system hoisting while rebasing prefix-dependent handles, and combine it with unhoisted system data only in fresh endpoint payload values. Preserve the no-feature path and all existing turn/ISL semantics.

**Tech Stack:** Rust 2024, clap, serde/serde_yaml, Tokio current-thread tests, BLAKE3 segment store, Hyper-based endpoint builders, in-repo Rust mock server, `/usr/bin/sccache`.

**Spec:** `docs/specs/2026-08-25-native-verbatim-system-prompts.md`

**Author:** Sol

## Global Constraints

- Work only in `/mnt/4tb/aiperf-origin-port-049` on `ajc/port-origin-049`; do not touch the shared checkout or #56 worktree.
- Preserve target-only ancestry and use an actual two-parent merge whose exact second parent is `88242293b552db96b90b2e3999bbfa93488c994f`; never cherry-pick.
- Use `/usr/bin/sccache` and `CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-049-target` for every Cargo command.
- Keep raw file paths out of resolved Config-v2, composition, cellular transport, worker state, and request hot paths.
- Preserve exact prompt bytes after blank validation; use exactly two newlines only when joining custom and authored system text.
- Do not add synchronization, background work, a new cache framework, a tokenizer bypass, or a dependency.
- No production `unwrap()`/`expect()`; file descriptors and all error exits remain RAII-owned.
- Preserve absent-feature serialization and request bytes, synthetic user ISL, authored turn metrics, and raw Anthropic blocks.
- Keep unrelated campaign churn and the untracked `.venv` symlink unstaged.

---

### Task 1: Commit semantic finding, design, and executable plan

**Files:**
- Create: `docs/origin-main-findings/commit-049-88242293b5.md`
- Create: `docs/specs/2026-08-25-native-verbatim-system-prompts.md`
- Create: `docs/superpowers/plans/2026-08-25-native-verbatim-system-prompts.md`

**Interfaces:**
- Produces: exact upstream/native disposition and all acceptance semantics.
- Produces: one Sol-authored implementation sequence with no unresolved design choice.

- [ ] **Step 1: Verify the upstream delta and native anchors**

Run `git show --stat 88242293b552db96b90b2e3999bbfa93488c994f`, inspect all changed upstream source/tests, and compare the config, composition, segment, and endpoint anchors named in the spec.

- [ ] **Step 2: Write the individual finding and spec**

Record exact source/native dispositions, applicable/non-applicable test mappings, precedence, acquisition safety, composition ordering, identity, endpoint payloads, allocation constraints, and scope.

- [ ] **Step 3: Self-review the plan**

Map every spec requirement to Tasks 2-7, scan for placeholders, and verify every field/function name is introduced before consumption.

- [ ] **Step 4: Commit the design slice**

```bash
git add docs/origin-main-findings/commit-049-88242293b5.md \
  docs/specs/2026-08-25-native-verbatim-system-prompts.md \
  docs/superpowers/plans/2026-08-25-native-verbatim-system-prompts.md
git commit -m "docs: design native verbatim system prompts"
```

### Task 2: Create the exact target-only upstream merge

**Files:**
- Import exactly the 20 paths in upstream commit `88242293b552db96b90b2e3999bbfa93488c994f`.

**Interfaces:**
- Consumes: Task 1 target branch tip.
- Produces: an actual merge commit with exact upstream second parent and exact upstream first-parent source delta.

- [ ] **Step 1: Start the two-parent merge without content resolution**

```bash
git merge --strategy=ours --no-ff --no-commit 88242293b552db96b90b2e3999bbfa93488c994f
```

- [ ] **Step 2: Apply only the target commit's exact binary-safe delta**

```bash
git diff --binary --output=/tmp/origin-049.patch \
  88242293b552db96b90b2e3999bbfa93488c994f^ \
  88242293b552db96b90b2e3999bbfa93488c994f
git apply --index /tmp/origin-049.patch
```

- [ ] **Step 3: Verify scope before commit**

Run `git diff --cached --stat` and compare the path list plus numstat to `git diff-tree --no-commit-id --numstat -r 88242293b5`. Expected: exactly the upstream 20 paths and no cumulative source.

- [ ] **Step 4: Commit and prove ancestry**

```bash
git commit -m "Merge origin/main 88242293b5 for verbatim system prompts"
git rev-parse HEAD^1 HEAD^2
```

Expected: second parent is exactly `88242293b552db96b90b2e3999bbfa93488c994f`.

### Task 3: Resolve inline/file sources and project typed configuration

**Files:**
- Modify: `rust/cli/src/flags.rs`
- Modify: `rust/cli/src/load.rs`
- Modify: `rust/cli/src/yaml.rs`
- Modify: `rust/runtime/src/config/resolve.rs`
- Create: `rust/runtime/src/config/system_prompt.rs`
- Modify: `rust/runtime/src/config/mod.rs`
- Modify: `rust/runtime/src/config/model/dataset.rs`
- Modify: `rust/runtime/src/engine/dataset_input.rs`
- Modify: `rust/runtime/src/engine/execute/dataset_build.rs`

**Interfaces:**
- Produces: `ProfileFlags.system_prompt: Option<String>` and `system_prompt_file: Option<PathBuf>`.
- Produces: unresolved `Inputs.system_prompt`/`system_prompt_file`, then crate-private `config::system_prompt::resolve_system_prompt(Option<&str>, Option<&Path>) -> Result<Option<String>, SystemPromptError>`.
- Produces: resolved `system_prompt: Option<String>` on all typed Dataset variants and engine specs.
- Produces: CLI source-unit overlay that clears the alternate YAML source.

- [ ] **Step 1: Write failing resolution and precedence tests**

In the existing `load.rs`, `yaml.rs`, and `config::resolve` test modules, add focused cases asserting: exact inline whitespace survives; a UTF-8 file resolves; mutation after resolution does not change the owned value; YAML inline/file project; CLI inline replaces YAML file; CLI file replaces YAML inline; same-layer dual sources fail; absent source stays `None`; all three typed Dataset variants carry the resolved string.

- [ ] **Step 2: Write failing file-safety tests**

On Unix, use temporary regular files plus leaf/parent symlinks. Add a consolidated error table for blank inline, blank file, missing path, directory, invalid UTF-8, symlink leaf, and symlinked parent. Assert errors name `--system-prompt` or `--system-prompt-file`, the authored path where applicable, and the expected regular UTF-8/no-symlink contract.

- [ ] **Step 3: Run focused tests and retain RED**

```bash
env RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-049-target \
  cargo test -p aiperf-cli system_prompt --lib
env RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-049-target \
  cargo test -p aiperf-runtime --features engine config:: --lib system_prompt
```

Expected: compilation/assertion failure because the profile fields and resolver do not exist.

- [ ] **Step 4: Implement minimal authoring and one-time acquisition**

Add clap/YAML fields and source-unit overlay. In the focused
`config/system_prompt.rs` module, reject both and validate inline without
trimming storage. On Unix, open the startup CWD/root and walk each component
with `openat`, `O_NOFOLLOW`, `O_CLOEXEC`, and directory-only parent flags; read
UTF-8 from the RAII-owned regular-file descriptor into one retained `String`.
Use a symlink-rejecting canonical fallback on non-Unix. Add
`#[serde(default, skip_serializing_if = "Option::is_none")]` resolved fields to
typed models/specs and project them through file/public/synthetic builders.

- [ ] **Step 5: Run focused tests to GREEN and commit**

Run the Step 3 commands plus `cargo test -p aiperf-cli --lib yaml` and commit the config slice separately.

### Task 4: Validate compatibility and compose prefix-dependent identity

**Files:**
- Modify: `rust/runtime/src/engine/dataset_input.rs`
- Modify: `rust/runtime/src/engine/execute/dataset_build.rs`
- Modify: `rust/runtime/src/dataset/compose.rs`
- Modify: `rust/runtime/src/dataset/loader/synthetic.rs`
- Modify: relevant dataset loader/composition unit tests

**Interfaces:**
- Consumes: resolved `system_prompt` from Task 3 and `EndpointDescriptor::consumes_system_message()`.
- Produces: `ComposeConfig.verbatim_system_prompt: Option<String>`.
- Produces: post-hoist exact system combination and `rebase_conversation_handles` identity propagation.

- [ ] **Step 1: Write failing validation tests**

Add one supported-endpoint table (chat, responses, messages, chat_embeddings), one unsupported table (completions, embeddings, rankings), conflict assertions for generated shared system and prefix pool, an allowance for user-context only, and paired `warmup_isolation_system` assertions with/without a resolved prompt.

- [ ] **Step 2: Write failing composition and identity tests**

Use synthetic, inline-record/file, and public-adapter fixtures. Assert exact assignment, `custom\n\nauthored` prepend, absent-feature neutrality, unhoisted leading-system retention, additive user ISL, and identical inline/file roots. Capture root and descendant `Handle`s: equal text must match; one-byte-changed prompt must change both; existing descendants must materialize to the same turn bytes after rebasing.

- [ ] **Step 3: Run dataset-focused RED**

```bash
env RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-049-target \
  cargo test -p aiperf-runtime --features engine system_prompt --lib
```

Expected: validation and composition assertions fail because no prompt reaches `ComposeConfig`.

- [ ] **Step 4: Implement minimal validation and composition**

Set the distinct verbatim field in every dataset builder. Validate system capability and prefix policy before loader work. Extend `apply_common_contexts` to materialize an existing system, allocate the exact two-newline join, tokenize/intern the combined root, and rebase descendants. Extend synthetic construction to install the verbatim root separately from generated user ISL. Reuse interned handles for repeated identical roots.

- [ ] **Step 5: Run focused suites to GREEN and commit**

Run the Step 3 command plus the existing simple/synthetic/public dataset suites. Commit validation, composition, and identity together.

### Task 5: Construct correct OpenAI and Anthropic production payloads

**Files:**
- Modify: `rust/runtime/src/endpoints/implementation.rs`
- Modify: `rust/runtime/src/endpoints/anthropic.rs`
- Modify: `rust/runtime/tests/endpoints_endpoints.rs`
- Modify: `rust/runtime/tests/endpoints_anthropic_messages.rs`

**Interfaces:**
- Consumes: `PreparedRequest.system_message()` produced from composed conversation roots.
- Produces: non-mutating Chat merge in warmup/profiling, Responses instructions, Chat Embeddings inheritance, and Anthropic prompt-plus-raw-block arrays.

- [ ] **Step 1: Write failing OpenAI construction tests**

Add literal payload assertions for a leading system string, a leading content-parts system, no leading system, Responses `instructions`, and Chat Embeddings. Format the same request three times in both warmup and profiling; assert one system message, exact `custom\n\nauthored`, custom occurrence count one, and unchanged stored turn values.

- [ ] **Step 2: Write failing Anthropic construction tests**

Assert prompt+raw blocks yields one leading text block followed by byte-equivalent authored JSON blocks with `cache_control`; raw-only remains the original array; prompt-only remains a string; latest authored raw-system selection is unchanged.

- [ ] **Step 3: Run focused endpoint RED**

```bash
env RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-049-target \
  cargo test -p aiperf-runtime --test endpoints_endpoints system
env RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-049-target \
  cargo test -p aiperf-runtime --test endpoints_anthropic_messages system
```

Expected: Chat profiling drops or misjoins the custom prompt and Anthropic drops it when raw blocks exist.

- [ ] **Step 4: Implement minimal non-mutating merges**

Only when `system_message()` is nonempty, request a mutable first rendered message for all credit phases. Prepend to a fresh string with `\n\n` or insert a leading text part, otherwise insert a standalone system message. For Anthropic, allocate a combined vector only when both sources exist and extend it with cloned raw blocks. Leave source turns and lowered wires untouched.

- [ ] **Step 5: Run endpoint suites to GREEN and commit**

Run both Step 3 targets without test filters, then commit the endpoint slice.

### Task 6: Add real-binary OpenAI and Anthropic integration evidence

**Files:**
- Modify/Create: focused tests under `rust/e2e-tests/tests/`
- Modify only if required for observation: `rust/mock-server/` request-recording test seam

**Interfaces:**
- Consumes: the native `aiperf` binary, `aiperf-mock-server`, and existing recorded-request inspection seam.
- Produces: product-level proof for inline and file configuration through actual HTTP payloads.

- [ ] **Step 1: Write the OpenAI E2E test first**

Launch the mock server with request recording. Run `aiperf profile` once with YAML file source overridden by CLI inline and once with CLI file source. Assert success, exact request count, exact preserved whitespace, one system role, correct two-newline merge with an authored system, and no duplicate prefix.

- [ ] **Step 2: Write the Anthropic E2E test first**

Run the native profile path against the mock Messages endpoint using a dataset turn with raw system blocks. Assert the recorded top-level system array begins with the resolved text block and preserves the authored block, order, and `cache_control`.

- [ ] **Step 3: Run E2E RED, implement only missing observation glue, then GREEN**

```bash
env RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-049-target \
  cargo test -p aiperf-e2e-tests --test system_prompt -- --nocapture
```

Expected RED is missing profile projection/payload behavior, not server timing. After implementation, both exact recorded-payload tests pass.

- [ ] **Step 4: Commit E2E evidence**

Commit the product test and any narrowly required mock recording fixture separately from documentation.

### Task 7: Synchronize docs, verify, Graham-review, repair, and close #49

**Files:**
- Modify: `docs/specs/2026-08-25-native-verbatim-system-prompts.md`
- Modify: `docs/specs/dataset.md`
- Modify: `docs/specs/endpoints.md`
- Modify: `docs/specs/README.md`
- Modify: user-facing CLI/config documentation generated or owned by native source
- Modify: `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc`, and `llms.txt` only where current-truth synchronization requires it
- Modify: `docs/origin-main-findings/commit-049-88242293b5.md`
- Modify: `docs/porting-origin-main-campaign.md`
- Create: `.superpowers/sdd/2026-08-25-native-verbatim-system-prompts/graham-review.md`
- Create as needed: `.superpowers/sdd/2026-08-25-native-verbatim-system-prompts/graham-rereview.md`

**Interfaces:**
- Consumes: complete native feature, exact merge ancestry, and all focused receipts.
- Produces: current-truth docs, zero unresolved Critical/Important findings, explicit `GRAHAM APPROVED`, and clean closure evidence.

- [ ] **Step 1: Update current-truth documentation**

Document flags/YAML precedence, safe one-time file read, supported endpoints, additive ISL, two-newline composition, and Anthropic block behavior. Update generated docs through their repository generators rather than hand-editing generated output.

- [ ] **Step 2: Run fresh focused and package gates**

Run Cargo format; CLI config tests; runtime dataset/endpoint tests; runtime library with and without `engine`; mock-server/E2E tests; scoped Clippy with `--no-deps`; docs guards; and `git diff --check`. Use the mandated sccache and target directory for every Cargo command and record pass counts plus any proven unrelated baseline failure.

- [ ] **Step 3: Run independent full-range Graham review**

Review from base `1d1978c22e00786ccf8739a599fd5b70d0d1b191` through tip, including exact ancestry and unstaged intended files. Inspect errors, path/descriptors, allocation, clones, repeated formatting, hot-path work, cache identity, strict serde, endpoint wire semantics, tests, and scope. Record exact severity/file:line findings.

- [ ] **Step 4: Repair blockers through fresh RED/GREEN cycles**

For each Critical/Important finding, add the smallest reproducing Rust test, verify RED, implement the minimal correction, verify GREEN, commit it, and request an independent re-review of the updated full range. Repeat until the verdict is exactly `GRAHAM APPROVED`.

- [ ] **Step 5: Prove ancestry, scope, and clean state**

Run `git rev-list --parents -n 1 <merge>`, compare its exact second parent, compare first-parent changed paths to the upstream target delta, inspect `git diff --stat 1d1978c22e..HEAD`, run `git diff --check 1d1978c22e..HEAD`, and confirm `git status --short` contains only the intentional untracked `.venv` link before closure artifacts are committed.

- [ ] **Step 6: Close the finding and tracker**

Record exact commits, test counts, integration payload evidence, ancestry, scope, review/repair receipts, and final verdict in the individual finding; mark campaign row #49 complete; commit closure artifacts separately.
