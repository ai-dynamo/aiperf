# Native Verbatim System Prompts

Status: Designed

Author: Sol

## Purpose

Port the user-visible semantics of origin/main commit
`88242293b552db96b90b2e3999bbfa93488c994f` into the native Rust profile path.
A benchmark may author one exact system prompt inline or in a UTF-8 file; AIPerf
acquires that text at startup, applies it consistently to every supported
dataset kind, includes it in native prefix/cache identity, and emits it through
system-aware production endpoint payloads.

## Public authoring and precedence

`aiperf profile` adds:

- `--system-prompt <TEXT>`
- `--system-prompt-file <PATH>`

Config-v2 YAML accepts the corresponding `dataset.system_prompt` and
`dataset.system_prompt_file` fields. These are authoring sources, not two
independent values. At one layer they are mutually exclusive. A CLI source is
one overlay unit: choosing either CLI flag replaces and clears both YAML source
fields. Thus CLI inline overrides YAML file and CLI file overrides YAML inline
without manufacturing a cross-layer conflict. If neither CLI flag is present,
the YAML pair is retained and validated.

`Inputs` carries the unresolved `Option<String>`/`Option<PathBuf>` pair through
overlay. Resolution returns `Option<String>` and stores that owned text on the
typed `Dataset::{Synthetic, File, Public}` variant. The strict protocol-v2
engine specs carry the resolved text only. No file path is sent into dataset
composition, worker setup, cellular planning, or a request hot path.

## One-time safe file acquisition

The startup resolver enforces these steps in order:

1. Reject simultaneous inline and file sources with both public flag names in
   the error.
2. For inline text, reject `text.trim().is_empty()` and otherwise retain the
   original bytes and whitespace.
3. For a file on Unix, resolve relative paths from the captured startup working
   directory and walk every component with `openat(O_NOFOLLOW|O_CLOEXEC)`, using
   directory-only flags for parents and a read-only leaf descriptor. On other
   platforms, reject symlink metadata before and after canonicalization and
   open the canonical leaf. Require metadata from the opened descriptor to be
   a regular file, then read the complete file as UTF-8 once.
4. Reject unreadable, non-UTF-8, empty, or whitespace-only file content with
   the authored path and expected input shape in the error.

The returned owned `String` is the sole runtime source. Mutating, replacing, or
removing the file after resolution cannot affect the resolved configuration.
The Unix descriptor walk prevents a check/open race from following a replaced
symlinked component; every successful descriptor transfer has one `OwnedFd`
owner. The reader reserves at most the descriptor metadata length and performs
one retained text allocation rather than staging a second byte vector.
The feature adds no per-conversation descriptor, per-request read, watcher, or
cache lock. File and descriptor cleanup is RAII-owned by the single startup
read.

## Validation boundary

A resolved prompt is legal only when the selected endpoint descriptor reports
`consumes_system_message()`. This naturally admits native Chat Completions,
Responses, Anthropic Messages, and Chat Embeddings and refuses endpoints whose
wire contract has no system role. Validation uses the descriptor capability so
new endpoint registrations fail closed unless they explicitly advertise the
seam.

For synthetic prefix policy, a verbatim prompt conflicts with:

- `prefix_prompts.shared_system_length`
- `prefix_prompts.length`
- `prefix_prompts.pool_size`

It does not conflict with `prefix_prompts.user_context_length`. Existing
prefix-policy pair validation still applies. A verbatim prompt is a static
system source and therefore satisfies `cache_bust.target =
warmup_isolation_system`; without either a verbatim prompt or generated shared
system, that target remains an early configuration error.

Blank prompts are rejected rather than normalized to omission. An absent prompt
must serialize exactly as before through serde defaults and
`skip_serializing_if`, preserving old Config-v2 documents and request bodies.

## Composition and prefix identity

`ComposeConfig` gains a distinct `verbatim_system_prompt: Option<String>`.
It is not folded into the existing `shared_system_prompt`, because file/public
formats must first hoist an authored leading system message before the custom
prompt can prepend to it. The order is:

1. A format composer builds conversations and performs its existing leading
   system hoist policy.
2. `apply_common_contexts` resolves any existing system segment, creates
   `custom + "\n\n" + authored`, or installs `custom` when none exists.
3. The combined exact bytes are tokenized once per unique system string and
   interned as a root segment.
4. System-dependent user context, turns, replies, and auxiliary handles are
   rebased through the existing prefix-dependent `SegmentPool` mechanism.

Synthetic composition uses the same distinct field when it constructs the
system root. It never subtracts system tokens from generated input length: the
authored ISL continues to size the user prompt only.

Native does not have upstream's mmap cache. The corresponding native identity
contract is stronger and closer to request construction: BLAKE3 segment IDs
and derived BodyPlan materialization must include the system prefix. Inline and
file sources resolving to identical text produce identical system and
descendant handles; different text changes those handles; omission produces
the pre-feature graph byte-for-byte. The implementation must not add a second
cache-key structure or hash the original file path.

Fixed formats that intentionally keep a leading system turn, such as the
native SpeedBench-equivalent/raw-message path, retain that turn. Their custom
conversation system is merged at the endpoint boundary, because changing the
format's composition/turn metrics would be a semantic regression.

## OpenAI-shaped request construction

Chat request formatting takes the zero-extra-work path when no conversation
system exists. When one exists, it renders the first message to a fresh mutable
`Value` in both warmup and profiling:

- leading system string: `custom + "\n\n" + authored`;
- leading system content parts: insert one leading text part containing
  `custom`, retaining every authored part and field;
- no leading system: insert one standalone system message before user context
  and turns.

Repeated construction may run against shared lowered turn data, so only the
fresh rendered value is mutated. The custom prefix appears exactly once on
every call. Chat Embeddings delegates to the Chat construction and inherits
the same rules.

Responses continues to place the conversation system in top-level
`instructions`; authored input turns stay in `input`. The port adds focused
coverage that resolved custom text survives this production path and retains
exact whitespace.

## Anthropic request construction

Anthropic Messages uses top-level `system`. If a turn carries `raw_system`
blocks and the conversation has a custom system prompt, the payload is:

```json
[
  {"type": "text", "text": "<custom>"},
  "<every authored raw-system block in original order>"
]
```

Authored blocks are cloned without normalization, including `cache_control` and
unknown extension fields. With raw blocks only, the original array remains
unchanged. With a custom prompt only, the existing string form remains. Latest
turn selection for authored raw-system blocks remains unchanged.

## Allocation, synchronization, and errors

File acquisition and prompt tokenization happen before issuance. A root prompt
without an authored system is interned once and reused. Only conversations with
an authored system require one joined allocation and one tokenize/intern pass;
their descendants must already be rebased today when context changes. The
Chat endpoint allocates a mutable first-message value only on the system-present
merge path. Anthropic allocates one additional block/vector only when it must
combine two sources.

No mutex, task, thread, unbounded channel, watcher, log statement, or new
dependency is introduced. Production errors use the existing config/application
error context and dataset composition's explicit error types; no production
`unwrap()` or `expect()` is added.

## Test and integration contract

Focused Rust tests cover every applicable upstream behavior listed in
`docs/origin-main-findings/commit-049-88242293b5.md`. In addition, native tests
must prove:

- CLI-over-YAML source replacement in both directions;
- symlinked parent rejection and invalid UTF-8 file refusal;
- BLAKE3 root plus descendant identity equivalence/change;
- identical payloads across warmup/profiling when the same conversation is
  formatted repeatedly;
- Responses and Chat Embeddings production construction, not only Chat;
- an actual native-binary OpenAI run and Anthropic run against the in-repo mock
  server, with recorded request JSON asserting exact prompt placement.

Every Cargo command uses `/usr/bin/sccache` and
`CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-049-target`. Closure requires
focused suites, runtime with `engine`, CLI tests, mock-server/E2E tests, scoped
Clippy and format, docs guards, range whitespace, exact ancestry, scope diff,
and a clean status apart from the intentionally untracked `.venv` symlink.

## Source anchors

- `rust/cli/src/flags.rs`
- `rust/cli/src/load.rs`
- `rust/cli/src/yaml.rs`
- `rust/runtime/src/config/resolve.rs`
- `rust/runtime/src/config/system_prompt.rs`
- `rust/runtime/src/config/model/dataset.rs`
- `rust/runtime/src/engine/dataset_input.rs`
- `rust/runtime/src/engine/execute/dataset_build.rs`
- `rust/runtime/src/dataset/compose.rs`
- `rust/runtime/src/dataset/loader/simple.rs`
- `rust/runtime/src/dataset/loader/synthetic.rs`
- `rust/runtime/src/endpoints/implementation.rs`
- `rust/runtime/src/endpoints/anthropic.rs`
- `rust/runtime/tests/endpoints_endpoints.rs`
- `rust/runtime/tests/endpoints_anthropic_messages.rs`
- `rust/e2e-tests/`
