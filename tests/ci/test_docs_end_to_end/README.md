# Docs End-to-End Harness

This harness runs documented benchmark commands against real inference
servers. HTML comment tags select the commands. The
`tools/check_docs_e2e_tags.py` gate finds new benchmark examples without tags.

> The suite only ever runs against a real server on the GPU runner. Never point
> it at the in-repo mock server -- a green run would then prove nothing about
> the documented command.

> **This page deliberately never spells a tag out in full.** The parser scans
> every markdown file in the repo line by line and does not care whether a line
> sits inside a code fence, so a complete example here would be collected and
> executed as if it were a real benchmark. Placeholders in `{braces}` below
> stand in for the literal text; `docs/tutorial.md` holds working examples of
> all three kinds.

## Tagging a guide

A tag is an HTML comment wrapping one fenced code block, opened with
`{kind}-{server}-endpoint-server` and closed with the same name prefixed by a
slash.

`{kind}` is one of:

| Kind | Purpose |
|---|---|
| `setup` | Supplies the image, arguments, dependencies, and environment for a native vLLM server. |
| `health-check` | Blocks until the server answers. Gates that group's run commands. |
| `aiperf-run` | A command to execute. A group may have many. |

`{server}` names the server group the block belongs to. Blocks sharing a
`{server}` form one unit no matter which file they live in: the harness
collects them across the whole repo, runs that group's single setup and health
check once, then every run command tagged with that name. The groups are:

| Server group | Endpoint under test |
|---|---|
| `vllm-default-openai` | chat / completions -- the bulk of the tutorials |
| `vllm-vision-openai` | image input |
| `vllm-audio-openai` | audio input |
| `vllm-video-openai` | video input |
| `vllm-embeddings-openai` | embeddings |

**A new tutorial usually needs only an `aiperf-run` tag.** The setup and
health-check blocks for each group already exist -- `docs/tutorial.md` for the
default chat server, and `docs/tutorials/{vision,audio,mmvu,embeddings}.md` for
the others -- and a second setup block for a group that already has one is a
conflict, not an addition. Copy the shape from `docs/tutorial.md`.

### Attributes

An opening `aiperf-run` tag may carry a `weight={seconds}` attribute, after the
tag name and before the closing `-->`. The weight is the command's estimated
runtime, used only by the matrix sharder to bin-pack work so one shard does not
end up owning every slow test. The default (80s) suits a typical
synthetic-input tutorial; set it when a command runs materially longer. Getting
it wrong makes the sharding lopsided, not incorrect.

### Fixtures belong with their consumer

A guide that builds its own input -- a heredoc writing a `.jsonl` trace, say --
and then profiles against it must keep both blocks in the same file. The
sharder splits work across runners a whole file at a time precisely so a
fixture and the command that reads it cannot land on different machines; two
files that depend on each other would still break, and nothing detects it.

## Running it

Use a Linux environment with a GPU, Bash, curl, FFmpeg, and vLLM installed.
Install AIPerf in a separate Python environment with `uv`.
If the environments use different interpreters, set
`AIPERF_DOCS_VLLM_PYTHON` to the absolute path of the vLLM interpreter.
Activate the AIPerf environment before running the harness.
The audio dependency overlay requires a system installation of vLLM, as
provided by the CI container.

```bash
cd tests/ci/test_docs_end_to_end
python3 main.py --dry-run                      # list what would run, start nothing
python3 main.py                                # every server group
python3 main.py --server vllm-default-openai   # one group
python3 main.py --server vllm-default-openai --shard-index 0 --shard-total 4
```

`setup_test.sh` is the CI entry point and forwards `SERVER_NAME`,
`SHARD_INDEX`, and `SHARD_TOTAL` to these flags.

CI uses `ai-dynamo-gpu-1-amd64-v1` in the `ai-dynamo-gpu` organization runner
group. Kubernetes launches a vLLM job container with one GPU.
The workflow installs AIPerf in an isolated Python environment.
The harness starts vLLM directly and runs benchmark commands through Bash.
It requires no Docker daemon or privileged container.

The harness translates supported Docker setup wrappers into native execution.
It preserves server arguments and explicit environment assignments.
Inline Dockerfiles support `FROM` and `RUN pip install` instructions.
Their dependencies use a separate environment that inherits the installed
server packages. Version constraints preserve the compiled GPU stack while
allowing Python dependencies to satisfy the documented extras.
The dependency resolver checks the compiled packages but omits them from
installation so the overlay reuses the image's copies.
Unsupported setup instructions fail the test.
If `AIPERF_DOCS_SERVER_IMAGE` is set, the documented base image must match it.

Each server uses one working directory so fixture files survive between
commands. The harness maps `/fixtures/` references to a link to
`tests/fixtures/`. It stops its own process groups after completion, failure,
timeout, or cancellation. It then removes that server's working directory.

The suite tests native server startup, documented health checks, and benchmark
commands. It does not build Docker images or test Docker runtime flags.

## When CI runs it

`.github/workflows/test-docs-end-to-end.yml` decides per push. It runs the GPU
matrix when the parser-extracted config hash differs between base and head
(`dump_config_hash.py`), and unconditionally when the push touches the harness,
its own workflow, the `Dockerfile`, or the surface a documented command is
written against -- the plugin registry, the config and CLI layers, and the
enums whose members appear verbatim in tutorial flags. A behavior change deeper
in `src/` is left to the nightly run.
