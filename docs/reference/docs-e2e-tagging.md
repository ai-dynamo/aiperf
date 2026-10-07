<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tagging a Guide for the Docs End-to-End Suite

The docs end-to-end suite answers one question: **does the command printed in
this guide still work?** It runs each tagged command verbatim, inside the
AIPerf container, against a real inference server on a GPU runner.

It is deliberately *not* run against the in-repo mock server. A command that
passes against a mock proves only that AIPerf parsed its own flags; the
failures this suite exists to catch -- a renamed server flag, a chat template
that stopped tokenizing, an endpoint that moved -- only appear against a real
engine. The unit and integration suites already cover the mock.

Harness source lives in `tests/ci/test_docs_end_to_end/`. Guards that keep
tagged blocks honest live in `tests/unit/ci/`.

## The four tags

Tags are HTML comments, so they are invisible in rendered documentation. Each
one wraps a fenced code block and is closed by a matching `/` tag.

Every tag name ends in `-endpoint-server`. The text between the prefix and that
suffix is the **server group**: all tags sharing a group run against one server
instance, booted once.

| Tag | Block contains | Cardinality |
|---|---|---|
| `setup-<group>-endpoint-server` | bash that starts the server | exactly one per group |
| `health-check-<group>-endpoint-server` | bash that blocks until ready | exactly one per group |
| `aiperf-run-<group>-endpoint-server` | bash running `aiperf` | one or more per group |
| `setup-file-<group>-endpoint-server path=<p>` | file content, any language | zero or more per group |

A group's tags may be spread across several markdown files. The harness scans
the whole repository, not just `docs/` -- `README.md` is tagged this way.

````markdown
<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile --model Qwen/Qwen3-0.6B --url http://localhost:8000 \
  --endpoint-type chat --request-count 20
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->
````

## Attributes

Attributes go on the opening tag as `name=value`, space separated.

### `weight=<seconds>`

Estimated runtime, used only by the matrix sharder to bin-pack commands so one
shard does not inherit every slow test. Default `80`. Set it when a command
runs materially longer than a single-point benchmark -- a sweep, a long
`--request-count`, a model that must download first.

````markdown
<!-- aiperf-run-vllm-longrun-openai-endpoint-server weight=1800 -->
````

### `timeout=<seconds>`

Hard kill deadline for the block. Defaults to `AIPERF_COMMAND_TIMEOUT`
(1200s) in `constants.py`. Sweeps and multi-phase workflows legitimately run
past that default, and capping them there is what keeps those guides
untestable.

`weight` and `timeout` are independent: `weight` is a scheduling hint and may
be wrong without consequence, `timeout` terminates the run.

Both must be positive integers with no unit suffix. A malformed value drops
only its own command and logs an error -- one typo must not silently empty the
suite. `timeout=0` is rejected because it would fall through to the global
default, and `timeout=-1` because it would kill the command on contact.

### `path=<relative-path>` (required on `setup-file-`)

Where to write the block inside the container's working directory. Must be
relative and may not contain `..`; the harness refuses anything that escapes
the working directory.

Use this for guides driven by `--config foo.yaml` or a `.jsonl` trace, where
the file's contents are already printed on the page. Materializing that block
is what makes such a guide testable without rewriting it to point at a path
the reader does not have.

````markdown
<!-- setup-file-vllm-tools-openai-endpoint-server path=tools.json -->
```json
[{"type": "function", "function": {"name": "get_weather"}}]
```
<!-- /setup-file-vllm-tools-openai-endpoint-server -->
````

## How a tagged block executes

- Run blocks are piped to `bash -se` over **stdin**, not interpolated into
  `bash -c '...'`. Guides routinely pass JSON in single quotes
  (`--extra-inputs '{"temperature": 0}'`), which quote-wrapping would strip.
- `-e` means a block of several commands **fails on its first error**. Without
  it bash reports only the last command's status, so a guide whose setup step
  crashed would still pass.
- Health-check blocks run without `-e`. Gate each step with
  `|| { echo "..."; exit 1; }`, never `&& echo "ok"` -- the latter leaves the
  block's status to the *next* command, masking the failure.
- Detached setups (`docker run -d`) have their container logs followed
  automatically, so an engine that dies on boot shows its own traceback rather
  than only a failed health check.
- `--ui-type simple` is injected into run commands that do not specify one.

### Blocks must be self-contained

A block that writes a file must also consume it. Nothing may depend on a file
or variable another block created, because the sharder distributes individual
commands across runners. `test_docs_e2e_block_self_contained` fails the build
if that stops holding.

## Adding a new server group

1. Tag `setup-`, `health-check-`, and at least one `aiperf-run-` block.
2. Add a matrix shard naming the group in
   `.github/workflows/test-docs-end-to-end.yml`, or
   `test-docs-long-guides.yml` if it is slow enough to need the weekly job.
   `test_docs_e2e_shard_coverage` fails if a documented group has no shard, or
   a shard names a group that does not exist -- an unsharded group looks
   covered and never runs.
3. Check the job's `timeout-minutes` still exceeds the sum of its command
   ceilings.

Run `python tests/ci/test_docs_end_to_end/main.py --dry-run` to see what the
parser discovered without executing anything.

## Model sweeps

A sweep borrows an existing group's commands, substitutes the model, and runs
them against a different family -- answering "does AIPerf work on family X"
rather than "is this guide correct". A failure there is a product bug, not a
documentation bug. Targets are declared in
`tests/ci/test_docs_end_to_end/model_sweep.py`; a sweep states its own server
command because documented ones carry family-specific flags.
