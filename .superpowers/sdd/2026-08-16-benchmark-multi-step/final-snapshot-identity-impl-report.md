# Final implementation report: source snapshot and package identity

## Status

DONE. The final corrective design is implemented as one coherent cutover. A
native eval import now owns one immutable canonical source snapshot, derives a
versioned normalized package identity from the resolved plan plus the exact
executable-source projection, and gives execution no authority to read the
caller's origin. Docker context, selected verifier trees, entry kinds,
canonical modes, empty directories, and normalized artifact exclusions are
bound by the appropriate identities. Shared verifier workdirs reserve only
the required directional namespaces.

The final Docker sweep also exposed and fixed one adjacent runtime defect: a
non-root image user could not create the executor-owned verifier reward
directory. The executor now creates that namespace as root only after agent
artifact capture and removes it after each shared multi-step verifier.

Final review found and corrected two additional boundary defects. Implicit
single-step separate verifiers now receive declared artifacts at the effective
image workdir without mounting the staging tree over the verifier filesystem.
Source acquisition now traverses directories by descriptor and opens every
entry relative to its already-open parent with no-follow semantics, so a
symlink substitution race cannot escape the acquired source root.

## Delivered contracts

### Owned canonical source

- `SourceTreeSnapshot` owns sorted directory and regular-file entries.
- Directory modes canonicalize to `0755`; regular files canonicalize to `0755`
  when any executable bit was present and `0644` otherwise.
- File bytes, empty directories, entry kind, path, and canonical mode
  participate in tree identity.
- Links, special files, non-UTF-8 names, escaping paths, and entry replacement
  races fail closed.
- Directory enumeration and child opens are descriptor-relative. Opened
  file/directory type and inode are validated against enumeration, and file
  and directory metadata are revalidated across acquisition.
- Reading, executable projection, and private materialization operate only on
  retained snapshot data after acquisition.
- Local directories are captured once. Standalone and pinned-Git file sources
  retain their exact owned bytes through the same acquisition boundary.

### Versioned plan and executable-source identity

- `CanonicalPackagePlan` encodes the complete resolved plan with framed fields
  and deterministic map/set ordering.
- Artifact exclusion patterns have one canonical form: relative normalized
  patterns are sorted and deduplicated; absolute and non-normal patterns are
  rejected.
- Package identity uses the `aiperf-eval-package-v2` domain and length-frames
  the canonical plan digest with the executable-source projection digest.
- Standard task projection binds the complete `environment/` Docker build
  context and every resolved verifier test tree.
- Directory-backed JSON packages bind the complete acquired tree. Standalone
  JSON packages bind the primary acquired file.
- The complete source digest remains provenance and intentionally includes
  unselected source entries that do not change normalized executable identity.
- `package.identity_digest()`, task digest, and import-report normalized digest
  are the same central value.

### Execution cutover and verifier reservation

- Import normalization reads only `AcquiredSource`; the package no longer
  carries a caller-owned `source_root` execution authority.
- Local and Docker execution materialize only the retained snapshot.
- Docker materializes once, builds from that private context, copies tests from
  it, retains it through the entire multi-step session and cleanup, then
  releases it.
- A plan containing any shared verifier rejects a manifest, CLI, or image
  workdir equal to or below `/tests` or `/logs/verifier`.
- Ancestors `/` and `/logs`, component neighbors such as `/tests-output`, and
  separate-only workdirs remain valid.
- Static authored conflicts are `InvalidPackage`. CLI/image conflicts are
  `InvalidWorkspace`; CLI conflicts fail before build, while image conflicts
  fail immediately after start/workdir inspection and before healthcheck,
  agent execution, reset, or copy.
- A single-step separate verifier with declared artifacts follows the same
  staging contract: an explicit workdir is validated before provisioning, an
  implicit workdir is inspected from the verifier image, and verified
  artifacts are copied there without mounting or resetting that destination.

## Approved baseline correction

Before feature work, the focused baseline contained one stale test. Duplicate
artifact targets were already rejected correctly during import, but the test
still expected collection-time rejection. Per explicit approval, commit
`514c489f1c` changed only that expectation to import-time `InvalidPackage`; no
validation was weakened.

Focused corrected baseline: 66 passed, 0 failed, and 3 expected ignored
real-Docker tests.

## TDD evidence

### Slice 1: immutable source snapshot

RED was recorded before implementation:

- snapshot types and owned tree acquisition APIs did not compile;
- the native acquirer could expose only primary file bytes/path provenance;
- no test could read, project, or materialize a directory after its origin was
  mutated and removed.

GREEN coverage:

- `capture_orders_entries_and_normalizes_modes_independent_of_creation_order`
- `empty_directories_and_executable_bits_independently_change_the_tree_digest`
- `snapshot_reads_projects_and_materializes_after_origin_mutation_and_deletion`
- `snapshot_rejects_links_special_entries_non_utf8_and_escaping_paths`
- `native_acquirer_owns_directory_tree_identity_but_preserves_file_byte_identity`

Result: 4 snapshot unit tests and the acquisition integration test passed.
Commit: `638b53c7c7` (`feat(eval): capture immutable source artifacts`).

### Slice 2: canonical plan inputs

RED was recorded with tests proving that reordered/duplicated exclusion lists
produced different identities and execution values, while malformed absolute
or non-normal patterns imported successfully.

GREEN: exclusions are normalized once, used by both execution and identity,
and malformed patterns fail import. The focused result was 22 importer tests
plus 4 snapshot unit tests passing.

Commit: `934864b529` (`fix(eval): canonicalize package plan inputs`).

### Slice 3: atomic identity and source-authority cutover

RED was recorded in five focused contracts:

- new identity tests initially failed to compile because the package had no
  central `identity_digest`;
- local directory JSON execution after origin removal failed with `ENOENT`;
- Docker execution after origin removal failed before build because it looked
  for the caller's `environment/Dockerfile`;
- Docker context/selected test changes, modes, and empty directories were not
  all bound by the normalized package identity;
- unselected provenance and executable identity were not separated.

GREEN focused result: all five contracts passed. Full slice regression:

- `harbor_import`: 25 passed;
- `eval_execution`: 8 passed;
- `harbor_execution_plan`: 18 passed;
- `harbor_docker_runtime`: 22 passed and 3 expected ignored.

Commit: `907469b366` (`feat(eval): cut over to snapshot-bound package identity`).

### Slice 4: directional shared-workdir reservation

RED was recorded in importer and Docker tests:

- authored shared `/tests` imported successfully;
- CLI `/tests` proceeded through evaluation instead of failing before build;
- image workdirs below reserved namespaces were not rejected at the required
  post-start/pre-healthcheck boundary.

GREEN coverage proves authored, CLI, implicit/explicit image, mixed
shared/separate, separate-only, ancestor, and component-neighbor cases. Full
slice result:

- `harbor_import`: 27 passed;
- `eval_execution`: 8 passed;
- `harbor_execution_plan`: 18 passed;
- `harbor_docker_runtime`: 26 passed and 3 expected ignored.

Commit: `b7eb0987b5` (`fix(eval): reserve shared verifier workdirs directionally`).

### Slice 5: real-Docker product proof and current-truth docs

The ignored product test
`imported_multi_step_snapshot_survives_origin_mutation_and_removal` imports a
two-step task and then mutates/removes the origin before Docker execution. It
proves shared then separate verification still uses the retained Docker
context and selected test trees, including auxiliary context files, empty
directories, and executable helper modes.

This product proof was added after the atomic behavior was already GREEN; it
is not represented as a new behavioral RED. Slice 3 supplied the preceding
unit/fake-runtime RED for the same contract.

Exact elevated command:

```bash
cd rust
RUSTC_WRAPPER= cargo test -p aiperf-e2e-tests \
  --test test_harbor_benchmark_execution \
  imported_multi_step_snapshot_survives_origin_mutation_and_removal \
  -- --ignored --exact --nocapture
```

Final result after all fixes: 1 passed, 0 failed, 24 filtered; Docker execution
completed in 3.32 seconds.

Commit: `02fef1dba8` (`test(eval): prove snapshot-bound Docker execution`).

### Final Docker-sweep defect: non-root reward namespace

The complete ignored CLI Docker sweep initially produced 5 passes and 2
failures. Diagnosis separated a stale fixture assumption from a real product
gap:

- two old fixtures assumed implicit `/work`, but the product correctly
  preserves an image `WORKDIR`; both fixtures now author
  `[environment] workdir = "/work"` explicitly;
- `openclaw-sandbox:bookworm-slim` uses `WORKDIR /home/sandbox` and
  `USER sandbox`, so the verifier could not create `/logs/verifier`.

After correcting only the fixture workdir, the pinned real-Docker test was RED
with `mkdir: cannot create directory '/logs': Permission denied`. A recording
lifecycle test was independently RED because no root preparation event existed
between agent completion and test copy.

GREEN behavior creates a fresh `/logs/verifier` as root after artifact capture,
sets it writable for the effective verifier user, copies/runs the verifier,
and removes the namespace after every shared multi-step verifier so it is not
left writable for the next agent.

Focused recording result: 1 passed. Full fake-runtime result: 26 passed and 3
expected ignored. All ignored CLI Docker tests then passed: 7 passed, 0 failed
in 14.14 seconds.

Commit: `015932128e` (`fix(eval): prepare nonroot verifier rewards`).

### Slice 6: single-step separate-verifier transfer parity

Three focused lifecycle tests were added before the implementation. The RED
run selected all three and failed all three:

- an implicit verifier image workdir was never inspected or populated;
- an explicit verifier workdir was incorrectly passed as a host mount instead
  of receiving a verified copy;
- explicit `/tests` was accepted and execution completed instead of failing
  before verifier provisioning.

```bash
cd rust
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine \
  --test harbor_docker_runtime single_step_separate_verifier_ -- --nocapture
```

RED: 0 passed, 3 failed. GREEN: 3 passed, 0 failed.

GREEN removes the single-step staging mount, validates explicit destinations,
inspects the image workdir when implicit, and calls the same verified transfer
boundary used by multi-step execution. A preservation run caught one
intermediate regression for packages with no declared artifacts; the final
implementation intentionally performs workdir validation/inspection/transfer
only when an artifact transfer is required.

Focused result: 3 passed. Full fake-Docker lifecycle result: 29 passed and 3
expected ignored.

Commit: `d959678e19` (`fix(eval): stage single-step verifier artifacts safely`).

### Slice 7: descriptor-relative source acquisition

The deterministic adversarial test installs a test-only pre-open boundary,
replaces an enumerated regular file with a symlink to bytes outside the source
root, and requires import-time `InvalidPackage`. RED was a compile failure
because the secure capture boundary did not exist. Under the prior path-based
`fs::read` behavior, the substitution could follow the outside-root link
rather than fail closed.

```bash
cd rust
# RED: the capture boundary intentionally does not exist yet.
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --lib \
  snapshot_rejects_file_swapped_to_an_outside_symlink_before_open \
  -- --exact --nocapture
# GREEN: fully qualified exact filter.
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --lib \
  eval::import::source_snapshot::tests::snapshot_rejects_file_swapped_to_an_outside_symlink_before_open \
  -- --exact --nocapture
```

RED: compilation failed because `capture_with_before_open` was absent. GREEN:
1 passed, 0 failed, and 1,908 filtered.

GREEN opens the root with `O_DIRECTORY | O_NOFOLLOW`, enumerates each already
open directory descriptor, and opens children with `openat` and
`O_NOFOLLOW | O_NONBLOCK`. It rejects known link/special entry types before
open, validates enumerated type/inode against `fstat`, reads files only from
the validated descriptor, recursively traverses only validated directory
descriptors, and compares metadata fingerprints across each read/traversal.

The correctly qualified exact GREEN command selected 1 test and passed. The
complete source snapshot module passed 5 tests, and `harbor_import` passed all
27 tests. An earlier exact-filter spelling selected zero tests; it was not
counted as evidence and was immediately replaced by the qualified command.

Commit: `505831ccc2` (`fix(eval): capture source trees without link races`).

## Final verification

### Focused runtime and CLI

```bash
cd rust
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine \
  --test harbor_import --test eval_execution \
  --test harbor_execution_plan --test harbor_docker_runtime
RUSTC_WRAPPER= cargo test -p aiperf-cli --test eval_command
```

Results:

- `harbor_import`: 27 passed;
- `eval_execution`: 8 passed;
- `harbor_execution_plan`: 18 passed;
- `harbor_docker_runtime`: 29 passed and 3 expected ignored;
- CLI: 8 passed and 7 expected ignored.

### Complete real-Docker runtime and CLI fixtures

```bash
cd rust
RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine \
  --test harbor_docker_runtime -- --ignored --nocapture
RUSTC_WRAPPER= cargo test -p aiperf-cli --test eval_command \
  -- --ignored --nocapture
```

Elevated Docker results:

- runtime: all 3 ignored fixtures passed;
- CLI: all 7 ignored fixtures passed.

The snapshot-bound product proof was also rerun after the descriptor-relative
change: 1 passed, 0 failed, and 24 filtered.

### Broad runtime safety

```bash
cd rust
RUSTC_WRAPPER= cargo test -q -p aiperf-runtime --features engine --lib \
  -- --test-threads=1
```

The engine-enabled serial suite passed: 1,902 passed, 0 failed, and 7 ignored
in 85.48 seconds. A preceding default-parallel run had 1,900 passes and two
worker-characterization timing/data-shape failures; both exact tests passed
immediately in isolation, and the complete serial rerun was clean. Those tests
do not execute eval or source acquisition code.

### Static and documentation guards

```bash
cd rust
cargo fmt --all -- --check
RUSTC_WRAPPER= cargo clippy -p aiperf-runtime --features engine \
  --lib \
  --test harbor_import --test harbor_docker_runtime \
  --test harbor_execution_plan --test eval_execution --no-deps
RUSTC_WRAPPER= cargo clippy -p aiperf-cli --test eval_command \
  -p aiperf-e2e-tests --test test_harbor_benchmark_execution --no-deps
cd ..
/usr/bin/python3 tools/check_agent_files_sync.py
/usr/bin/python3 tools/check_docs_current.py
git diff --check
```

Every command exited zero. Clippy reported only the repository's existing
warning baseline; no new production or test warning was introduced by this
implementation.

## Commits

- `514c489f1c` — `test(eval): expect duplicate artifact rejection at import`
- `638b53c7c7` — `feat(eval): capture immutable source artifacts`
- `934864b529` — `fix(eval): canonicalize package plan inputs`
- `907469b366` — `feat(eval): cut over to snapshot-bound package identity`
- `b7eb0987b5` — `fix(eval): reserve shared verifier workdirs directionally`
- `02fef1dba8` — `test(eval): prove snapshot-bound Docker execution`
- `015932128e` — `fix(eval): prepare nonroot verifier rewards`
- `21191957bd` — `docs(eval): record snapshot identity implementation`
- `d959678e19` — `fix(eval): stage single-step verifier artifacts safely`
- `505831ccc2` — `fix(eval): capture source trees without link races`

The linked worktree has no `.venv`, so commits skipped only the license and
agent-sync hooks that require that environment. All touched source files retain
the required SPDX header, and the agent-sync/docs checks were run directly and
passed.

## Concerns and remaining scope

- No known correctness concern remains for the corrective design.
- The full engine library suite's two worker-characterization tests showed
  parallel-run sensitivity; both exact reproductions and the serial full suite
  passed. This is an existing non-eval test stability concern, not evidence of
  a source snapshot or verifier lifecycle regression.
- Existing Clippy/dead-code warnings remain outside this change.
- Persistent CAS, registry fetch, Docker Compose, task sidecars, artifact wire
  format changes, reward aggregation changes, and CLI redesign remain out of
  scope exactly as specified.
