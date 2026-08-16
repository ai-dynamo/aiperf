# Final fix report: standard-task identity and verifier-owned paths

## Scope

Implemented only the final whole-branch findings for normalized standard-task
identity and verifier-owned path collisions. Source snapshot timing, Docker
snapshot architecture, cleanup mechanics, CLI output, and unrelated lifecycle
behavior were not changed.

## Changes

- Added a versioned canonical execution-plan encoder in
  `rust/runtime/src/eval/execution/plan.rs`. Every tag and value is
  length-framed; maps use normalized `BTreeMap` order, lists preserve authored
  order, durations retain seconds plus nanoseconds, and secret bindings encode
  only the normalized reference name.
- Standard task identity now includes task id, image kind/digest, implicit vs
  explicit layout, reward strategy, all environment/phase/verifier fields,
  every effective artifact declaration, every resolved step, and all regular
  files in each selected verifier tree ordered by normalized relative path.
- Import rejects artifact sources that equal, contain, or are contained by
  verifier-owned `/tests` or `/logs/verifier`, using component-aware paths so
  neighbors such as `/tests-output` remain valid. It also rejects known
  separate-verifier staging targets under those namespaces.
- Multi-step Docker execution validates CLI/authored workdirs before verifier
  creation and image workdirs immediately after inspection. Both checks run
  before artifact transfer, verifier reset, test copy, or verifier execution.

## TDD evidence

### RED

Focused importer command:

```bash
cd rust
env -u RUSTC_WRAPPER cargo test -p aiperf-runtime --test harbor_import
```

Result before implementation: 14 passed and 4 expected failures.

- All 21 policy/artifact mutations retained the same task digest.
- Changing a selected root or step verifier helper retained the same digest.
- Reserved artifact sources imported successfully.
- Known authored-workdir staging collisions imported successfully.

Focused runtime command:

```bash
cd rust
env -u RUSTC_WRAPPER cargo test -p aiperf-runtime --test harbor_docker_runtime
```

Result before implementation: 19 passed, 2 expected failures, and 3 ignored.
Both `/tests` image-WORKDIR and `/logs/verifier` CLI-workdir cases completed
artifact transfer and verification instead of returning `InvalidWorkspace`.

### GREEN

```bash
cd rust
cargo fmt --all --check
env -u RUSTC_WRAPPER cargo test -p aiperf-runtime \
  --test harbor_import --test harbor_docker_runtime
cd ..
git diff --check
```

Result: `harbor_import` 19 passed; `harbor_docker_runtime` 21 passed with 3
expected ignored real-Docker tests; formatting and diff checks exited zero.

The identity suite additionally proves normalized allowlist casing/order and
environment table order do not change identity, while secret reference names,
implicit/explicit layout, reward strategy, plan policy, artifacts, and helper
bytes do.

## Broader verification

The first sandboxed full-unit run reached 1,449 passes but 35 existing tests
failed to bind loopback sockets or spawn local helpers with `Operation not
permitted`. The required elevated retry was:

```bash
cd rust
cargo test -p aiperf-runtime --lib
```

Result: 1,484 passed, 0 failed, and 7 ignored.

Targeted Clippy:

```bash
cd rust
env -u RUSTC_WRAPPER cargo clippy -p aiperf-runtime \
  --test harbor_import --test harbor_docker_runtime --no-deps
```

Result: exited zero. Output contained only the repository's existing warning
baseline; no warning pointed to the new identity/path implementation or tests.

## Commit

Scoped commit subject: `fix(eval): bind task identity and verifier paths`.
