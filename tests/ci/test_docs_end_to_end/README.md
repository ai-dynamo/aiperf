# Docs End-to-End Harness

This harness executes the commands printed in `docs/` against a real inference
server, so a tutorial that no longer works fails CI instead of failing a user.
It is driven by HTML comment tags in the markdown itself: an untagged command
is never run, which is what `tools/check_docs_e2e_tags.py` guards against on
newly added docs.

> The suite only ever runs against a real server on the GPU runner. Never point
> it at the in-repo mock server -- a green run would then prove nothing about
> the documented command.

## Tagging a guide

Every tag wraps one fenced code block:

~~~
<!-- <kind>-<server>-endpoint-server -->
```bash
<the command>
```
<!-- /<kind>-<server>-endpoint-server -->
~~~

`<kind>` is one of:

| Kind | Purpose |
|---|---|
| `setup-` | Starts the inference server. Runs once per server group, before anything else. |
| `health-check-` | Blocks until the server answers. Gates that group's run commands. |
| `aiperf-run-` | An `aiperf profile` invocation to execute. A group may have many. |

`<server>` names the server group the block belongs to. Blocks sharing a
`<server>` form one unit no matter which file they live in: the harness
collects them across all of `docs/`, runs that group's single setup and health
check once, then every run command tagged with that name. The groups are:

| Server group | Endpoint under test |
|---|---|
| `vllm-default-openai` | chat / completions -- the bulk of the tutorials |
| `vllm-vision-openai` | image input |
| `vllm-audio-openai` | audio input |
| `vllm-video-openai` | video input |
| `vllm-embeddings-openai` | embeddings |

**A new tutorial usually needs only an `aiperf-run-` tag.** The setup and
health-check blocks for each group already exist (`docs/tutorial.md` for the
default chat server; `docs/tutorials/{vision,audio,mmvu,embeddings}.md` for the
others), and a second setup block for a group that already has one is a
conflict, not an addition.

So, in practice:

~~~markdown
<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->
~~~

### Attributes

An opening `aiperf-run-` tag may carry a `weight=<seconds>` hint:

```
<!-- aiperf-run-vllm-default-openai-endpoint-server weight=300 -->
```

The weight is the command's estimated runtime, used only by the matrix sharder
to bin-pack commands so one shard does not end up owning every slow test. The
default (80s) suits a typical synthetic-input tutorial; set it when a command
runs materially longer. Getting it wrong makes the sharding lopsided, not
incorrect.

## Running it

```bash
cd tests/ci/test_docs_end_to_end
python3 main.py --dry-run                      # list what would run, start nothing
python3 main.py                                # every server group
python3 main.py --server vllm-default-openai   # one group
python3 main.py --server vllm-default-openai --shard-index 0 --shard-total 4
```

`setup_test.sh` is the CI entry point and forwards `SERVER_NAME`,
`SHARD_INDEX`, and `SHARD_TOTAL` to these flags.

## When CI runs it

`.github/workflows/test-docs-end-to-end.yml` decides per push. It runs the GPU
matrix when the parser-extracted config hash differs between base and head
(`dump_config_hash.py`), and unconditionally when the push touches the harness,
its own workflow, the `Dockerfile`, or the surface a documented command is
written against -- the plugin registry, the config and CLI layers, and the
enums whose members appear verbatim in tutorial flags. A behavior change deeper
in `src/` is left to the nightly run.
