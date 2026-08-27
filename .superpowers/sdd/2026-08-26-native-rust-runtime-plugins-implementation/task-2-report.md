# Task 2 plugin crate-boundary report

## Cumulative commits

- `b421eb91c2f6b642842592c3ab0856fb712b62c1` — initial crate shells.
- `65a102bc4bef13c27ce5173009a3c025d559acb9` — exact workspace policy.
- `fd3aea9ec0ba5dc37424e118dccbb0c64b6f1725` — strict candidate and lock policy.
- `20809f9e48f4a2aa7e719adb1263256c96fee8ca` — captured topology checks.
- `2dabbceacb761ecfd74b57756728e0fc72a0a4a6` — exact lock/topology identity.
- `7dd5c9ed4f55a7ad96bcd91816cde36ca689ac1a` — complete dependency projection.

## Graham round-5 remediation

Status: implementation complete except for importing the parallel Task-1-owned
authenticated Cargo projection. Task 2 remains intentionally RED on that one
missing ancestor artifact and does not claim GO before independent re-review.

### Task-1 authenticated Cargo projection contract

Task 1 owns the exact projection inside the existing authenticated artifact
`artifacts/native-plugin-baseline/package-topology.json`. Task 2 does not own a
second digest or authority file. The required top-level field is
`cargo_projection`, an ordered array of strict package objects:

```text
{
  name,
  version,
  edition,
  dependencies,
  features
}
```

Each `dependencies` entry has this complete strict shape:

```text
{
  package,
  local_name,
  kind,
  source,
  requirement,
  registry,
  path,
  target,
  is_optional,
  uses_default_features,
  features,
  is_workspace
}
```

Paths are normalized relative to workspace root `rust`. Package and dependency
arrays and every feature vector are deterministically sorted while retaining
multiplicity. Task 2 strictly deserializes these package/dependency rows and
compares its current pre-Task-2 package projection, after removing only the
three named Task-2 e2e feature additions, directly to the ancestor-owned values.
The removed `rust/plugin-api/task1-cargo-authority.toml` is not retained as a
secondary circular authority.

Test-first consumer RED:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cargo test --locked --manifest-path rust/plugin-api/Cargo.toml \
  --test dependency_policy workspace_and_template_policy -- --nocapture
```

Output before the Task-1 artifact import:

```text
Task 1 package topology must be JSON: Error("missing field `cargo_projection`", line: 1202, column: 1)
test workspace_and_template_policy ... FAILED
```

### Complete Task-2 and reviewed Task-3 shell features

Both exact shell projections now include each package's complete sorted
`BTreeMap<String, Vec<String>>` feature map, including `default` and dependency
feature forwarding vectors with multiplicity retained. The independently
calculated exact hashes are:

```text
task2_neutral  blake3:abf887c2a5947da0854f0e04c90d17240c978839063e64bcf82f093f3ab783d7
task3_reviewed blake3:a194e94568ae4e0dec27d957ee75e8a47f4e94f47e97ca8c9932e0902fdec6f5
```

The test-data-first RED changed those two expected hashes before changing the
projection code and ran:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cargo test --locked --manifest-path rust/plugin-api/Cargo.toml \
  --test dependency_policy workspace_and_template_policy -- --nocapture
```

It failed exactly:

```text
complete task2_neutral shell dependency matrix drift
left:  "blake3:d7baa308220d3b295bb1c3bb8aaf7c5d59ddc985baed16c2eb05474af4bcf2be"
right: "blake3:abf887c2a5947da0854f0e04c90d17240c978839063e64bcf82f093f3ab783d7"
```

After implementation the same focused command passed. Two restored independent
manifest mutations then proved the feature coverage:

- Adding `[features] unauthorized = []` to `aiperf-plugin-host` failed with
  actual digest
  `blake3:d5f803aca161f67d2d239b926023b42f2d7e5114e0a8eb24627f5536f9b23628`.
- Adding `default = ["forwarded"]` and
  `forwarded = ["aiperf-plugin-test-support/forwarded"]` to
  `aiperf-export-sdk`, with the target empty feature on test support, failed
  with actual digest
  `blake3:bc980b7ec4ca109964f47879c31f534612c2fceda662a4e1096f7245abd2cd20`.

### Executable distribution and host-universe boundaries

`rust/scripts/verify-plugin-test-support-boundaries.py` replaces selected-file
string scans with these executable checks:

- executes `make --no-print-directory -n install-native` and requires the exact
  complete six-command vector: one `aiperf-cli` full-feature Cargo build, one
  output-directory creation, one single-binary copy, and the three exact
  informational messages;
- builds the real project wheel without isolation, runs the actual wheel
  repacker with a real ELF, and verifies the complete ZIP/RECORD identity, exact
  selected native executable, and absence of test-support source;
- parses Docker's logical stage graph and resolves context/stage COPY ancestry
  into the actual default runtime stage, including JSON-form COPY, chown/chmod,
  stage inheritance, directory copies, and aliases. It additionally hashes the
  ordered complete instruction vectors of every stage reachable from the
  default target and admits only
  `sha256:85179fec37130992b2fe6037ea1d3063a641ebc9f053566dadd054570f4efbf5`.
  Thus the Docker wheel/runtime projection is independently exact rather than
  inferred from the separately built host wheel. Reachable ADD, ONBUILD, RUN
  mounts, final-stage RUN mutation, and every unreviewed reachable instruction
  fail closed;
- executes the operator's typed envelope-to-JobSet projection using the exact
  reachable Docker instruction SHA-256 as its digest-qualified boundary token,
  requires every role to use that same proved runtime-image boundary, rejects
  alternate init/ephemeral images and untracked filesystem volumes, and
  validates the exact image-capabilities schema; and
- retains Cargo-metadata traversal of the complete non-dev host universe rooted
  at `aiperf-cli`, `aiperf-runtime`, and `aiperf-plugin-host`.

The test-first RED pointed `distribution_exclusion_policy` at the not-yet-created
verifier and failed with:

```text
python: can't open file '.../rust/scripts/verify-plugin-test-support-boundaries.py': [Errno 2] No such file or directory
test distribution_exclusion_policy ... FAILED
```

After implementation, each mutation below was applied alone, exercised, and
restored:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cargo test --locked --manifest-path rust/plugin-api/Cargo.toml \
  --test dependency_policy distribution_exclusion_policy -- --nocapture
```

- The reviewer's exact final-stage
  `COPY --from=wheel-builder /workspace/rust /opt/aiperf/rust-source` failed
  `reachable Docker instruction projection drift` with
  `sha256:988a369cdd7961dc7ec63e001b3fb35f8e6458f0621a417d9b8a7c6e9285a851`.
- `cp -R rust dist/native-bin/rust-source` in `install-native` failed with the
  complete two-command `native install payload drift` projection.
- Hatch `force-include` of `rust/plugin-test-support` failed after the real
  wheel build with both shipped Cargo/source paths named.
- Replacing the operator JobSet container image with an alternate tag failed
  `Kubernetes roles do not share the exact digest-qualified image`.
- A normal host-to-test-support dependency plus coherent lock failed
  `host-universe dependency closure includes test support`.

An additional native mutation appended a Python file-creation command to
`install-native`. It first reproduced a false GREEN under the selected-command
filter. After the exact six-command projection replaced that filter, the same
mutation failed with `native install command/payload projection drift` and
printed the unauthorized Python command in the actual vector.

The pre-hardening sanitizer accepted this appended file-creation command as a
false GREEN because its fragment checks still matched:

```Dockerfile
&& python -c "import shutil; shutil.copytree('/workspace/rust','/dist/rust-source')"
```

The test-first replacement set the admitted projection digest to zero; the
unmodified Dockerfile then failed RED with the independently calculated
`sha256:85179fec37130992b2fe6037ea1d3063a641ebc9f053566dadd054570f4efbf5`.
After pinning that exact projection, the following independent Docker mutations
were applied and restored. Each was exercised with:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
python -c "import importlib.util,pathlib; \
p=pathlib.Path('rust/scripts/verify-plugin-test-support-boundaries.py'); \
s=importlib.util.spec_from_file_location('boundary',p); \
m=importlib.util.module_from_spec(s); s.loader.exec_module(m); \
m.verify_final_container(pathlib.Path('.').resolve())"
```

- The appended Python `copytree` failed with
  `sha256:dbcc52cb9d06e3f775843ce2e561918e3e6a77746c13f26e40dbf0ac7c83ef1b`.
- An extra `RUN` before the wheel build failed with
  `sha256:f387f603513f0fed880e7330320762e14ccc5124298449233605f0147c4b5f31`;
  one after it failed with
  `sha256:d5381640afae559d2c6d947da16f9528e12f1e5b903c275ed4981879b1768931`.
- JSON-form directory COPY with `--from=wheel-builder`, `--chown`, and
  `--chmod` failed with
  `sha256:44d39f2914df1a14bd06128561f23e72d33c1de4a00f644315f60a989c1742be`.
- Final-stage `ADD rust ...` failed with
  `sha256:48e87dacfc24fea0e33909b5b273dd9be281aa589910b9d9ca3d0d1d98030557`.
- Final-stage `ONBUILD COPY rust ...` first reproduced the missing guard as a
  false GREEN; the exact projection then rejected it with
  `sha256:3030c042b0d1fe9d4c1ecc9cc3ef1e2071fc1b432470d0f98679d88aab9684d9`.
- A final alias inheriting `runtime` plus whole-directory COPY from
  `wheel-builder:/workspace/rust/` failed with
  `sha256:c412f1d6662f9a69ffdd1cd6c99c54417fed847f11a4b22aca0979797892c11b`.

After restoration, the complete native, exact Docker projection, real wheel,
typed Kubernetes, and host-universe gate passed:

```text
test distribution_exclusion_policy ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out
```

### Minor findings

The boolean predicate is now `is_git_object_present`; the pre-fix RED was the
exact `rg` result naming `git_object_exists` at its definition and use. The
candidate generator and generated inventory no longer retain the false
`provisional_against` field or provisional history comment. The test-first
top-level-shape assertion failed before that removal with:

```text
left:  {"base_commit", "provisional_against", "source"}
right: {"base_commit", "source"}
```

## Final verification

The controller ruled that this distinct remediation commit must remain on its
current prerequisite until Task 1's separate correction receives independent
Graham C0/I0/M0. Task 2 therefore does not copy, synthesize, cherry-pick, or
merge those bytes in this commit. The exact Task-1 ref delivered for later
integration is `d3a81465e7351a364b859238fd41151a727ad6a3` (prerequisite
`76b19b237d2e5a775f8a48c9d744496f01dc721e`), but is intentionally not yet an
ancestor here.

Commands and results on the restored Task-2 worktree:

```text
cargo test --locked -p aiperf-plugin-api --all-targets -- --nocapture
  3 passed; 1 failed
  sole expected RED: workspace_and_template_policy — missing cargo_projection

cargo test --locked -p aiperf-plugin-api --test dependency_policy \
  distribution_exclusion_policy -- --exact --nocapture
  1 passed; 0 failed

cargo test --locked -p aiperf-plugin-api --test dependency_policy \
  candidate_inventory_policy -- --exact --nocapture
  1 passed; 0 failed

cargo clippy --locked -p aiperf-plugin-api --all-targets -- -D warnings
  PASS

cargo fmt --manifest-path plugin-api/Cargo.toml -- --check
  PASS

cargo metadata --locked --format-version 1 --no-deps
  PASS

cargo metadata --locked --format-version 1 --no-deps \
  --manifest-path tests/plugin-third-party/Cargo.toml
  PASS

python rust/scripts/generate-plugin-candidate-inventory.py . --check
  PASS

ruff check rust/scripts/verify-plugin-test-support-boundaries.py \
  rust/scripts/generate-plugin-candidate-inventory.py
  PASS

ruff format --check rust/scripts/verify-plugin-test-support-boundaries.py
  PASS

git diff --check
  PASS
```

The repository-wide `cargo fmt --all --check` was also probed and found only
pre-existing unrelated formatting drift in CLI/runtime files outside this Task
2 range. The required package-scoped formatting gate above passes; no unrelated
files were reformatted.
