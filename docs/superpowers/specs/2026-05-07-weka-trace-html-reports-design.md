# `aiperf report weka-trace` — HTML Reports for Real Weka Traces

**Status:** design
**Date:** 2026-05-07
**Author:** acasagrande

## Goal

Add a CLI subcommand `aiperf report weka-trace <path>` that runs the existing
`agentic_code_gen` HTML reporting pipeline (`report.html`,
`cache_explorer.html`, `simulation.html`) against real Weka kv-cache-tester
trace files. Today these reports are only produced for the synthetic
`agentic-code` generator; this gives the team an apples-to-apples view of the
real source the synthesizer is calibrated against.

## Scope

- Real-trace report mode only. No comparison-overlay mode (synth-vs-real
  on the same plots) and no rerouting of the existing synth pipeline through
  `WekaTraceLoader`. Both are explicitly out of scope.
- No new metrics, no new renderers. The reporting pipeline already lives in
  `src/aiperf/dataset/agentic_code_gen/reporting/`; we add an input adapter
  and a CLI entry that feeds it.

## Architecture

### New files

```
src/aiperf/dataset/agentic_code_gen/reporting/
  weka_input.py           # NEW: weka JSON file/dir -> dict[session_id, list[ParsedTurn]]

src/aiperf/cli_commands/
  report.py               # NEW: cyclopts App with `weka-trace` default
```

### Edits

```
src/aiperf/cli.py
  + app.command("aiperf.cli_commands.report:app", name="report")

src/aiperf/dataset/agentic_code_gen/reporting/report.py
  _print_target_table: early-return when data.comparisons is empty
```

### Data flow

```
weka JSON file/dir
  -> WekaTrace pydantic models (existing, weka_trace_models.py)
  -> weka_input.load_weka_as_parsed
  -> dict[session_id, list[ParsedTurn]]
  -> extract_metrics + extract_cache_metrics  (existing, metrics.py)
  -> render_plot_report                       (existing, plot_report.py)
  -> write_cache_structure + render_cache_explorer  (existing, cache_explorer.py)
  -> render_simulation                        (existing, simulation.py)
  -> _print_report_to_console                 (existing, report.py — minor guard)
```

No tokenizer, no `UserConfig`, no `PromptGenerator`. The light path skips
all KV-cache prompt synthesis the loader does.

## `weka_input.py` — light reader

### Public API

```python
def load_weka_as_parsed(
    path: Path,
    *,
    include_subagents: bool = True,
    max_context_length: int | None = None,
) -> dict[str, list[ParsedTurn]]:
    """Read a single weka trace file or a directory of *.json into ParsedTurn sessions.

    Each parent trace becomes one session with id = trace.id. Each subagent entry
    in the parent's request list becomes a session with id = f"{trace.id}::sa:{agent_id}"
    when include_subagents=True. max_context_length drops whole traces whose peak
    parent input_length exceeds the cap (matching WekaTraceLoader's filter).
    """
```

### Behavior

- File-vs-directory detection mirrors `WekaTraceLoader._enumerate_files`:
  if `path.is_dir()`, take `sorted(path.glob("*.json"))`; otherwise treat as a
  single file. (No `*.json.gz` support — the existing loader doesn't either.)
- Per file: `WekaTrace.model_validate(orjson.loads(path.read_bytes()))`. Reuse
  the existing model — no new schema.
- Per parent trace, walk `trace.requests`:
  - For each `WekaNormalRequest` / `WekaStreamingRequest`, append a `ParsedTurn`
    to the parent session with:
    - `session_id = trace.id`
    - `input_length = req.input_length`
    - `output_length = req.output_length`
    - `hash_ids = req.hash_ids`
    - `delay_ms = (req.t - prev_normal.t) * 1000.0` for index ≥ 1, else `0.0`
      where `prev_normal` is the previous **normal** request (subagent entries
      between two normals do not advance `prev_normal`, since their `t` is on
      the parent's clock and the parent's next normal is what the report cares
      about — matching how the existing `delay` semantics treat consecutive
      normals).
    - `group_id = None`, `is_restart = False` (weka traces don't carry these
      synth-only annotations).
  - For each `WekaSubagentEntry`, if `include_subagents`, build a separate
    session with id `f"{trace.id}::sa:{entry.agent_id}"` whose turns are the
    `WekaNormalRequest` items in `entry.requests`. Delays computed relative to
    the **first** entry in the subagent's `requests` list (so the subagent's
    first turn has `delay_ms = 0.0`, not parent-relative — these are local
    "session" delays for distribution analysis).
- `max_context_length` filter: applied at parent-trace level using the same
  rule as `WekaTraceLoader._filter_traces_by_max_context` — drop the trace
  (and its subagents) if the parent's peak `input_length` across its normal
  requests exceeds the cap.
- Returns `dict[session_id, list[ParsedTurn]]` insertion-ordered by trace
  filename order. Stable across runs because `_enumerate_files` sorts.

## `cli_commands/report.py` — CLI entry

```
aiperf report weka-trace <path>
  [--output DIR]                # default: Path(".")
  [--block-size INT]            # default: 512
  [--max-context-length N]      # optional pre-filter, drops oversized traces
  [--no-subagents]              # exclude subagent sessions
  [--prefill-tps FLOAT]         # default: 20000  (matches generate_report)
  [--decode-tps FLOAT]          # default: 60     (matches generate_report)
```

Cyclopts `App(name="report")` with a `weka-trace` default subcommand,
mirroring `synthesize.py`'s pattern (`Literal["agentic-code"]` target).

### Pipeline

```python
parsed = load_weka_as_parsed(path, include_subagents=not no_subagents,
                             max_context_length=max_context_length)
basename = path.stem if path.is_file() else path.name
ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
run_dir = output / f"weka-report_{basename}_{ts}"
run_dir.mkdir(parents=True, exist_ok=False)

metrics = extract_metrics(parsed, prefill_tps=prefill_tps, decode_tps=decode_tps)
metrics.update(extract_cache_metrics(parsed, block_size=block_size))
report_data = build_report_data(metrics, manifest=None)

render_plot_report(metrics, parsed, run_dir)
cache_payload = write_cache_structure(parsed, manifest=None, output_dir=run_dir)
render_cache_explorer(run_dir, cache_payload)

# Simulation: build sim shape from ParsedTurn (cumulative input length).
sim_sessions = _parsed_to_sim_sessions(parsed)
render_simulation(sim_sessions, run_dir / "simulation.html",
                  block_size=block_size, l1_tokens=None, l1_5_tokens=None)

_print_report_to_console(report_data)  # _print_target_table no-ops on empty
console.print(f"[green]Run directory: {run_dir}[/green]")
console.print(f"  Report:          {run_dir / 'report.html'}")
console.print(f"  Cache explorer:  {run_dir / 'cache_explorer.html'}")
console.print(f"  Simulation:      {run_dir / 'simulation.html'}")
```

`_parsed_to_sim_sessions` is a small helper colocated with `weka_input.py`
that re-implements the trivial bit of `load_simulation_sessions` (which
takes a JSONL path) for the in-memory `ParsedTurn` dict.

## Touch points to existing reporting code

### `report.py::_print_target_table`

Add a one-line guard:

```python
def _print_target_table(console: Console, data: ReportData) -> None:
    if not data.comparisons:
        return
    table = Table(title="Target vs Observed")
    ...
```

This is the only place in the reporting code that assumes a manifest is
present. `metrics.build_report_data(metrics, manifest=None)` already yields
`comparisons=[]` (verified at metrics.py:255-292). `cache_explorer.py:62-65`
reads from manifest opportunistically (`if manifest:`), defaulting block/L1
sizes when absent.

`render_plot_report` and `simulation.render_simulation` do not consume
manifest at all.

## Output layout

```
<output>/weka-report_<basename>_<UTC-ts>/
  report.html
  cache_explorer.html
  simulation.html
  cache_structure.json   # written by write_cache_structure
```

No `manifest.json` (real-trace mode has no synth-target distribution).
No `dataset.jsonl` copy.
No `comparison.txt` (synth-only artifact).

## Tests

`tests/unit/dataset/agentic_code_gen/test_weka_report_input.py`:

- **single-file fixture** → expected ParsedTurn list shape; assert
  `delay_ms` equals `(t_i - t_{i-1}) * 1000` for normals, `0.0` on turn 0.
- **directory of two traces** → two parent sessions present in the dict,
  insertion order matches `sorted(glob("*.json"))`.
- **trace with two subagents** → 1 parent + 2 subagent sessions; subagent
  session ids match `f"{trace_id}::sa:{agent_id}"`; each subagent's turn 0
  has `delay_ms=0.0`.
- **`include_subagents=False`** → only parent sessions appear.
- **`max_context_length` filter** → trace whose peak parent input_length
  exceeds cap is fully omitted (parent and subagents).
- **byte-comparable parent delays** — for a parent with no subagents,
  assert delays match what `WekaTraceLoader` computes (using an existing
  fixture from `tests/unit/dataset/loader/`, e.g. one of the
  `test_weka_trace_byte_exact_*` corpora).

`tests/unit/cli_commands/test_report_weka_cli.py` (smoke):

- Run the CLI on a tiny corpus dir against `tmp_path`; assert
  `report.html`, `cache_explorer.html`, `simulation.html`, and
  `cache_structure.json` exist and are non-empty.
- Auto-named run dir matches `weka-report_*` glob.

## Non-goals (explicit)

- No comparison overlay (synth distribution vs real distribution on the same
  plots).
- No rerouting of `synthesize agentic-code` output through `WekaTraceLoader`.
- No `*.json.gz` support unless `WekaTraceLoader` adds it first.
- No `mooncake` sibling (`aiperf report mooncake-trace`) — the existing
  `aiperf analyze-trace` already covers mooncake. This CLI lives at
  `aiperf report` to leave room for a mooncake/dag sibling later.
- No global hash-id namespace handling — weka v1 is `hash_id_scope: "local"`
  per-trace, which is what the report wants anyway (no cross-trace cache
  sharing inflation).

## Rollout

Single PR on `ajc/inferencex-agentx-mvp`. No env-var gates; the new CLI is
additive.
