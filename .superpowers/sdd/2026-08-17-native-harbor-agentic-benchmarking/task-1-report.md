# Task 1 report — Schema 1.1 NativeGraph package

## Status

Independent Graham approved the fourth correction round. The Task-1
implementation was committed after final host verification; this report records
that exact implementation commit.

## Commits

- Base: `1ed94ca837` (`docs(eval): specify native graph Harbor benchmarking`).
- Task implementation: `55ef618a50` (`feat(eval): resolve native graph packages`).

## RED evidence

The initial focused test was added before production implementation. After
correcting a test-helper return type, the host command below failed for the
intended reason:

```text
cargo test -p aiperf-runtime --test native_graph_package --test harbor_import -- native_graph --nocapture
error[E0599]: no method named `native_graph` found for struct `HarborTaskPackage`
  --> runtime/tests/native_graph_package.rs:15:37
```

The independent review then required a correction round. New tests were added
before the corresponding production changes and produced these intended RED
results:

```text
cargo test -p aiperf-runtime --test native_graph_package -- --nocapture
error[E0599]: no method named `program_source` found for
`&NativeGraphPackagePlan`

cargo test -p aiperf-cli --test eval_command \
  native_eval_refuses_schema_1_1_native_graph_before_provisioning \
  -- --exact --nocapture
unexpected NativeGraph refusal: sandbox command failed: "docker run ..."
```

The CLI RED reaches Docker because the test demonstrates the missing
pre-provision rejection; its agent marker was not created. The project had
pre-existing `GraphWorkload` dead-code warnings during Rust compilation; they
are unrelated to this task.

## GREEN evidence

Every final host run preserved the inherited `RUSTC_WRAPPER`/sccache
configuration and passed:

```text
cargo test -p aiperf-runtime --test native_graph_package -- --nocapture
native_graph_package: 7 passed

cargo test -p aiperf-runtime --test harbor_import -- --nocapture
harbor_import: 32 passed

cargo test -p aiperf-cli --test eval_command -- --nocapture
eval_command: 12 passed, 7 Docker-image-dependent tests ignored

cargo fmt --check
git diff --check
```

The correction tests retain the acquired graph program path/bytes/digest;
reject unknown fields, invalid profiles and identifiers, explicit steps,
noncanonical paths, credential-bearing model and server-tokenizer URLs,
query/fragment URLs, nonfinite generation values, invalid headers, and adapter
argv that does not start at its declared executable. A compact identity-mutation
table compares resolved `ModelBindingSpec` values, rather than the composite
task digest, so changes to `models.toml` cannot be masked by the independent
executable-source digest. It covers binding ID, endpoint profile/factory,
transport factory, model, ordered URLs, streaming, local-tokenizer
name/revision/template, the server-tokenizer variant/URL/template, both logical
header/secret references, every configured generation setting, retries, timeout,
and capture. The strict-boundary table includes duplicate model/adapter IDs and
headers, empty URLs/argv, both invalid tokenizer shapes, zero timeout, and both
native and externally driven profile-role violations. The canonical-path case
binds argv to the malformed executable so its rejection cannot be caused by an
unrelated argv mismatch; the external-driver baseline is explicitly imported
before its invalid role pairing is asserted. The schema-1.0 digest golden
remains unchanged.

## Verification commands

```text
cargo test -p aiperf-runtime --test native_graph_package --test harbor_import -- native_graph --nocapture
cargo test -p aiperf-runtime --test native_graph_package -- --nocapture
cargo test -p aiperf-runtime --test harbor_import -- --nocapture
cargo test -p aiperf-cli --test eval_command native_eval_refuses_schema_1_1_native_graph_before_provisioning -- --exact --nocapture
cargo test -p aiperf-cli --test eval_command -- --nocapture
cargo fmt --check
cargo fmt
cargo fmt --check
git diff --check
```

The brief's unfiltered Cargo example places `--nocapture` before the
test-harness separator; the recorded GREEN commands use `-- --nocapture`.

## Graham review

- Initial independent Graham verdict: not approved; it identified the
  fail-closed CLI, URL credential/query, ID grammar, retained-program-source,
  and adapter-argv identity gaps corrected in this round.
- Second independent Graham verdict: not approved solely for missing identity
  and strict-boundary test contracts. This round adds those tests without a
  production behavior change because the existing parser and identity encoder
  already satisfy the requested behavior.
- Third independent Graham verdict: not approved solely for the final omitted
  binding/tokenizer identity rows and causal strict-test assertions. This round
  adds those test-only contracts; production behavior remains unchanged.
- Fourth independent Graham verdict: not approved because the model-manifest
  source projection made task-digest mutation insufficient to prove normalized
  binding retention. The test now compares resolved bindings directly and
  explicitly verifies the server-tokenizer template flag.
- Final independent Graham verdict: **approved**. The implementation commit
  above was then created with all applicable pre-commit hooks passing; only the
  `add-license` hook was skipped because its required `.venv` is unavailable.

## Concerns

- The focused Cargo output retains two existing `GraphWorkload` dead-code
  warnings.
- NativeGraph CLI execution deliberately fails closed until its native runner
  is implemented; this prevents legacy Docker/agent provisioning from running
  a schema-1.1 package.
- Model and server-tokenizer URLs allow only absolute HTTP(S) endpoints with
  no userinfo, query, or fragment. Authentication is represented exclusively
  by the logical secret identifiers in `authentication`.
- No Docker, Compose ledger, or `docker_process.rs` changes were made by this
  task. Those files remain pre-existing worktree edits and are excluded from
  the eventual Task-1 commit.
- The implementation commit's pre-commit run skipped only `add-license` after
  its first attempt failed solely because `.venv/bin/activate` is absent. All
  other applicable hooks passed.
