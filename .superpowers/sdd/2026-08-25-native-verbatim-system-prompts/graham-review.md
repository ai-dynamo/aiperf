# Graham review — origin/main #49 verbatim system prompts

## Review boundary

- Base: `1d1978c22e00786ccf8739a599fd5b70d0d1b191`
- Reviewed tip: `b4f2376319cfe5bbac94e9d79dfaf5b6e585242c`
- Exact upstream: `88242293b552db96b90b2e3999bbfa93488c994f`
- Target-only merge: `9eeeac98f944b5fdc425b3b2c1f8f65231447ac8`
- Merge parents: `17c69ce049be7aa5c7532a72efb35c617d2a858f`
  and exact upstream `88242293b552db96b90b2e3999bbfa93488c994f`

The independent reviewer completed two passes over the cumulative target
range. The review covered one-time no-follow file acquisition and descriptor
cleanup, exact-text ownership, config and strict-engine validation,
composition and prefix-dependent identity rebasing, OpenAI and Anthropic
production wire rendering, cache/warmup isolation, async and error behavior,
allocation and synchronization cost, tracing and comments, test rigor, and
scope.

## Findings

No Critical findings.

No Important findings.

The implementation reads the authored file only at startup, carries owned text
thereafter, and adds no request-hot-path file I/O, lock, channel, task, or
logging. Endpoint formatting mutates only fresh rendered values. Absent-feature
paths retain their existing representations, while configured text participates
in native BLAKE3 segment identity and precomputed body semantics.

## Evidence

- Exact upstream inventory: 40/40 test functions mapped; 39 native behavioral
  equivalents and one documented non-applicable tokenizer-bypass assertion.
- CLI library: 259 passed.
- OpenAI endpoint integration: 21 passed.
- Anthropic endpoint integration: 14 passed.
- Real binary plus Rust mock-server E2E: 14 passed.
- No-engine runtime: 1,811 passed, 7 ignored, one unchanged version-golden
  failure.
- Engine/all-target Clippy: blocked only by two unchanged AgentX test
  initializers missing an inherited field.
- Formatting, range whitespace, docs-current, and agent-file synchronization:
  passed.

GRAHAM APPROVED
