# Docs End-to-End Harness

This harness executes the commands printed in the docs against a real
inference server, so a tutorial that no longer works fails CI instead of
failing a user. It is driven by HTML comment tags in the markdown itself: an
untagged command is never run, which is what `tools/check_docs_e2e_tags.py`
guards against on newly added docs.

> The suite only ever runs against a real server on the GPU runner. Never point
> it at the in-repo mock server -- a green run would then prove nothing about
> the documented command.

> **This page spells tags out only in placeholder form.** The parser skips
> tags inside a fenced code block, so a fenced example here is safe; a tag
> written in running prose is not, and would be collected and executed as a
> real benchmark. Placeholders in `{braces}` below keep that distinction from
> mattering. `docs/tutorial.md` holds working examples of the three command kinds.

## Tagging a guide

A tag is an HTML comment wrapping one fenced code block, opened with
`{kind}-{server}-endpoint-server` and closed with the same name prefixed by a
slash.

`{kind}` is one of:

| Kind | Purpose |
|---|---|
| `setup` | Starts the inference server. Runs once per server group, before anything else. |
| `health-check` | Blocks until the server answers. Gates that group's run commands. |
| `aiperf-run` | A command to execute. A group may have many. |
| `setup-file` | A file to write into the container before any command runs. Needs `path=`; the block is yaml/json/jsonl rather than bash. |

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

A server's containers are stopped however its test ends -- a setup that dies,
a health check that never passes, a fixture that cannot be written. Every
documented group binds 8000, so an engine left holding the port makes the next
server fail to bind and report a docker networking error naming neither the
real failure nor the server that caused it.

Discovering no servers at all fails the run too. Nothing legitimately parses
to zero -- the tags live in this repository -- so an empty result means the
suite would boot a GPU runner, build the image and execute nothing.

A duplicate `setup` or `health-check` tag skips that whole server and fails
the run. It is not recoverable the way a dropped command is: two guides
disagree about how to boot the group, and running either would benchmark
against something no guide documents. Every other server still runs and
reports.

### Attributes

An opening tag may carry `name=value` attributes, after the tag name and before
the closing `-->`.

| Attribute | Valid on | Meaning |
|---|---|---|
| `weight={seconds}` | `aiperf-run` | The command's estimated runtime, used only by the matrix sharder to bin-pack work so one shard does not end up owning every slow test. The default (80s) suits a typical synthetic-input tutorial; set it when a command runs materially longer. Getting it wrong makes the sharding lopsided, not incorrect. |
| `timeout={seconds}` | `aiperf-run` | How long this one command may run before the watchdog kills it, overriding the global default. Set it when a documented command is legitimately slower than the rest. It is a ceiling, not an estimate: raising it does not tell the sharder anything, so a command that really does run longer wants `weight=` raised as well. |
| `path={relative path}` | `setup-file` | Where the block's content is written inside the container, relative to the working directory. Required. An absolute path or one containing `..` is rejected before any server boots. |

`weight` and `timeout` must be positive integers. A non-integer, zero or
negative value drops that one command rather than aborting discovery for every
guide -- but the drop is recorded and fails the run, so a typo cannot quietly
shrink what CI covers. The same applies to an attribute name the harness does
not read: a misspelled `timeout=` is rejected rather than ignored, because
ignoring it would leave the guide looking like it raised the deadline when it
did not. The **Valid on** column is enforced, not advisory -- `weight=` or
`timeout=` on a `setup` or `health-check` tag is rejected for the same reason,
since nothing reads either off a lifecycle command and a correctly spelled
attribute that does nothing misleads exactly as much as a typo. A `setup-file`
tag with no `path=`, or one whose fenced block is missing or never closed,
fails the run the same way.

An attribute must be `name=value` with no spaces around the `=` and no spaces
inside the value. A shape the harness cannot read -- `timeout=` with nothing
after it, `weight = 300`, a hyphen in the name -- is reported and drops the
block, rather than making the whole tag unreadable. Getting that backwards is
the worst available outcome: a tag that fails to parse as a tag is
indistinguishable from prose, so the command leaves the suite with nothing
recorded and the run stays green.

### Fixtures belong with their consumer

A guide that builds its own input -- a heredoc writing a `.jsonl` trace, say --
and then profiles against it must keep both blocks in the same file. The
sharder splits work across runners a whole file at a time precisely so a
fixture and the command that reads it cannot land on different machines; two
files that depend on each other would still break, and nothing detects it.

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

The hash covers everything the harness would act on: the setup and
health-check commands, each run command with its `weight=` and `timeout=`, and
every fixture's path and content. An edit the hash cannot see skips the matrix,
so a new tag attribute must be added to the fingerprint in the same change that
teaches the parser to read it. `dump_config_hash.py` also exits non-zero when
the parser rejected anything -- a dropped command, a duplicate tag, a
conflicting fixture -- which the workflow reads as "no comparable hash" and
runs the matrix. Nothing else would stop a tree whose tags no longer parse from
printing its pre-breakage digest and comparing clean against its own base.
