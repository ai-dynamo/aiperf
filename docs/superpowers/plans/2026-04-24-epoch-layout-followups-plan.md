# Epoch-layout follow-ups implementation plan

> **For agentic workers:** Subagent-driven execution; each task runs in its own isolated worktree branched from `ajc/k8s`. Merge back to `ajc/k8s` after each wave.

**Goal:** Ship the four follow-ups deferred from the epoch-keyed results layout PR — list-runs endpoint, CLI, dashboard dropdown, age-based retention, env-vars generator coverage.

**Spec:** `docs/superpowers/specs/2026-04-24-epoch-layout-followups-design.md`
**Branch:** `ajc/k8s` (no feature branch; cherry-pick worktree commits back per standing preference for parallelism only).

**Per-task contract:**
- DCO sign-off on every commit (`git commit -s`) with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.
- ONE `uv run pytest -n auto tests/unit/` invocation per task (user standing feedback).
- `make check-ergonomics && make check-ruff-baselined` at end of task.
- `ruff format` + `ruff check --fix` on touched files before commit.

**Parallelization:**
- Wave 1 (3 parallel): Tasks A, D, E
- Wave 2 (2 parallel): Tasks B, C — both depend on Task A's endpoint

---

## Task A: `/api/v1/results/<ns>/<name>/runs` list-runs endpoint

**Files:**
- Modify: `src/aiperf/operator/results_layout.py` — add `list_runs(base, ns, name) -> list[RunEntry]` + `RunEntry` dataclass.
- Modify: `src/aiperf/operator/routers/results_schemas.py` — add `RunHistoryEntry` + `RunHistoryListResponse` pydantic models (if no compatible types exist).
- Modify: `src/aiperf/operator/routers/results_files.py` — add the route. Register BEFORE the `.../runs/{epoch}` (without `.zip`) route so `runs` literal matches first — or place it as the MOST SPECIFIC `runs` literal (no captured segment).
- Modify: `tests/unit/operator/test_results_server.py` — append 3 tests.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/operator/test_results_server.py` (use distinct function names from the Wave-2 `test_historical_*` names):

```python
def test_list_runs_returns_epochs_newest_first(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "a.json")
    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_NEW, "b.json")

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/job/runs")
        assert r.status_code == 200
        body = r.json()
        assert body["namespace"] == "ns"
        assert body["job_id"] == "job"
        assert body["latest_epoch"] == _EPOCH_NEW
        epochs = [run["epoch"] for run in body["runs"]]
        assert epochs == [_EPOCH_NEW, _EPOCH_OLD]
        latest_flags = [run["is_latest"] for run in body["runs"]]
        assert latest_flags == [True, False]
        for run in body["runs"]:
            assert run["file_count"] == 1
            assert run["total_size_bytes"] > 0


def test_list_runs_404_when_no_runs(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from aiperf.operator.results_server import create_app

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/absent/runs")
        assert r.status_code == 404


def test_list_runs_skips_non_epoch_dirs(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from aiperf.operator.results_layout import job_dir
    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "a.json")
    (job_dir(tmp_path, "ns", "job") / "not-an-epoch").mkdir()

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/job/runs")
        assert r.status_code == 200
        epochs = {run["epoch"] for run in r.json()["runs"]}
        assert epochs == {_EPOCH_OLD}
```

- [ ] **Step 2: Implement `list_runs` in `results_layout.py`**

```python
from dataclasses import dataclass


@dataclass(slots=True)
class RunEntry:
    """One run directory with summary metadata.

    Example:
        >>> entry = RunEntry(epoch="1714150923", mtime_epoch=1714150925,
        ...                  file_count=7, total_size_bytes=4823912,
        ...                  is_latest=True)
    """
    epoch: str
    mtime_epoch: int
    file_count: int
    total_size_bytes: int
    is_latest: bool


def list_runs(base: Path, namespace: str, name: str) -> list[RunEntry]:
    """Enumerate all run dirs under ``<base>/<ns>/<name>/``, newest first.

    Returns an empty list if no run dirs exist. The entry flagged
    ``is_latest=True`` matches ``latest.txt`` when the pointer is present
    and its target exists on disk.
    """
    parent = job_dir(base, namespace, name)
    if not parent.is_dir():
        return []
    latest = resolve_latest(base, namespace, name)
    runs: list[RunEntry] = []
    for p in parent.iterdir():
        if not p.is_dir() or not EPOCH_RE.match(p.name):
            continue
        files = [f for f in p.iterdir() if f.is_file()]
        runs.append(RunEntry(
            epoch=p.name,
            mtime_epoch=int(p.stat().st_mtime),
            file_count=len(files),
            total_size_bytes=sum(f.stat().st_size for f in files),
            is_latest=(p.name == latest),
        ))
    runs.sort(key=lambda r: r.mtime_epoch, reverse=True)
    return runs
```

Add `RunEntry` and `list_runs` to `__all__` and export them.

- [ ] **Step 3: Add schemas**

Look at `src/aiperf/operator/routers/results_schemas.py` for existing naming conventions. Add:

```python
class RunHistoryEntry(BaseModel):
    epoch: str = Field(description="Epoch-seconds key of this run.")
    mtime_epoch: int = Field(description="Directory modification-time epoch.")
    file_count: int = Field(description="Number of files in the run dir.")
    total_size_bytes: int = Field(description="Total bytes across all files.")
    is_latest: bool = Field(description="True when this run matches latest.txt.")


class RunHistoryListResponse(BaseModel):
    namespace: str
    job_id: str
    latest_epoch: str | None = Field(default=None, description="Current latest.txt target, or None.")
    runs: list[RunHistoryEntry] = Field(default_factory=list)
```

- [ ] **Step 4: Add the route**

In `src/aiperf/operator/routers/results_files.py`, inside `create_results_files_router`, register BEFORE the `{filename:path}` catch-all:

```python
    @router.get(
        "/results/{namespace}/{job_id}/runs",
        response_model=RunHistoryListResponse,
    )
    async def list_runs_endpoint(
        namespace: str, job_id: str
    ) -> RunHistoryListResponse:
        from aiperf.operator.results_layout import list_runs as _list_runs
        runs = await asyncio.to_thread(_list_runs, base_dir, namespace, job_id)
        if not runs:
            raise HTTPException(404, f"No runs for {namespace}/{job_id}")
        latest = next((r.epoch for r in runs if r.is_latest), None)
        return RunHistoryListResponse(
            namespace=namespace,
            job_id=job_id,
            latest_epoch=latest,
            runs=[
                RunHistoryEntry(
                    epoch=r.epoch,
                    mtime_epoch=r.mtime_epoch,
                    file_count=r.file_count,
                    total_size_bytes=r.total_size_bytes,
                    is_latest=r.is_latest,
                )
                for r in runs
            ],
        )
```

Import `RunHistoryEntry, RunHistoryListResponse` at top-of-file.

- [ ] **Step 5: Verify + commit**

```bash
uv run pytest -n auto tests/unit/
make check-ergonomics && make check-ruff-baselined
ruff format src/aiperf/operator/results_layout.py src/aiperf/operator/routers/results_files.py src/aiperf/operator/routers/results_schemas.py tests/unit/operator/test_results_server.py
ruff check --fix ...  # same files
git add src/aiperf/operator/results_layout.py src/aiperf/operator/routers/results_files.py src/aiperf/operator/routers/results_schemas.py tests/unit/operator/test_results_server.py
git commit -s -m "$(cat <<'EOF'
feat(operator): add /api/v1/results/<ns>/<name>/runs list-runs endpoint

Returns all run dirs under <ns>/<name>/ with epoch, mtime, file_count,
total size, and is_latest flag; newest first. Foundation for the
aiperf kube results list-runs CLI and the dashboard run-history
dropdown.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task B: `aiperf kube results list-runs` CLI

**Depends on:** Task A merged.

**Files:**
- Modify: `src/aiperf/cli_commands/kube/results.py` — add `list_runs` cyclopts sub-app.
- Create: `tests/unit/operator/test_cli_kube_results_list.py` — test table formatting + JSON mode.

- [ ] **Step 1: Inspect existing CLI patterns**

```bash
head -120 src/aiperf/cli_commands/kube/results.py
grep -rn "Literal\[.*text.*json\|Literal\[.*json.*text\]" src/aiperf/cli_commands/kube/ | head -10
grep -rn "kube_console\|from aiperf.kubernetes import console" src/aiperf/cli_commands/kube/ | head -5
```

Identify the existing port-forward helper, the `output: Literal["text","json"]` pattern, and the `kube_console` output chokepoint. Reuse.

- [ ] **Step 2: Implement the subcommand**

At the top of `src/aiperf/cli_commands/kube/results.py` (after the existing `app = App(name="results")` definition), register a sub-app:

```python
@app.command(name="list-runs")
async def list_runs(
    job_id: Annotated[str | None, Parameter(help="AIPerf job ID to list runs for (default: last deployed job).")] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    output: Annotated[Literal["text", "json"], Parameter(help="Output format.")] = "text",
    operator_namespace: Annotated[str, Parameter(name="--operator-namespace", help="Namespace where the operator is deployed.")] = "aiperf-system",
) -> None:
    """List all historical runs of a benchmark job.

    Examples:
        aiperf kube results list-runs                 # last deployed job
        aiperf kube results list-runs foo             # specific job
        aiperf kube results list-runs foo --output json
    """
    from aiperf import cli_utils

    manage_options = manage_options or KubeManageOptions()
    with cli_utils.exit_on_error(title="Error Listing Runs"):
        await _run_list_runs(
            job_id=job_id,
            manage_options=manage_options,
            output=output,
            operator_namespace=operator_namespace,
        )
```

Implementation (`_run_list_runs`) calls the operator API at `/api/v1/results/<ns>/<job>/runs` via the existing port-forward helper, then:
- `output == "text"` → format a table with columns `EPOCH, TIMESTAMP, FILES, SIZE, LATEST`. Convert `mtime_epoch` via `datetime.fromtimestamp(ts, timezone.utc).isoformat(sep=" ", timespec="seconds")`. Render with `kube_console`.
- `output == "json"` → downshift `aiperf.kube` logger to WARNING in a `try/finally`, print via `orjson.dumps(..., option=OPT_INDENT_2)`.

Fall back to the last-deployed job via `save_last_benchmark` helper if `job_id` is None (match existing `results.py` behavior).

- [ ] **Step 3: Write tests**

Create `tests/unit/operator/test_cli_kube_results_list.py`:

```python
# SPDX-FileCopyrightText: ...
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import orjson
import pytest


@pytest.mark.asyncio
async def test_list_runs_text_output_formats_table(capsys) -> None:
    # Mock the HTTP client that talks to the operator API to return a known payload.
    # Assert the stdout contains "EPOCH", "TIMESTAMP", and both epoch values.
    ...


@pytest.mark.asyncio
async def test_list_runs_json_output_emits_parseable_json(capsys) -> None:
    # Mock HTTP; assert stdout parses as JSON with "runs" key and 2 entries.
    ...


@pytest.mark.asyncio
async def test_list_runs_404_raises_informative_error() -> None:
    # Mock HTTP to return 404; assert an informative error message mentioning
    # the namespace and job_id.
    ...
```

Match the mocking pattern used by other kube CLI tests — grep `tests/unit/operator/test_cli_kube_*.py` to find the best template.

- [ ] **Step 4: Verify + commit**

```bash
uv run pytest -n auto tests/unit/
make check-ergonomics && make check-ruff-baselined
ruff format ...; ruff check --fix ...
git add src/aiperf/cli_commands/kube/results.py tests/unit/operator/test_cli_kube_results_list.py
git commit -s -m "$(cat <<'EOF'
feat(cli): add `aiperf kube results list-runs` subcommand

Lists historical runs for a benchmark job by hitting the operator's
/api/v1/results/<ns>/<name>/runs endpoint. Text or JSON output.
Defaults to the last-deployed job id when none is given.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Regenerate CLI docs**

```bash
make generate-cli-docs
```

If `docs/cli-options.md` updated, amend the commit or add a follow-on docs commit.

---

## Task C: Dashboard run-history dropdown

**Depends on:** Task A merged.

**Files:**
- Modify: `src/aiperf/operator/ui/views/job.js` (or equivalent — inspect `ui/views/` structure first).
- Modify: `src/aiperf/operator/ui/lib/api.js` — add `listRuns(ns, name)` helper.
- Modify: `src/aiperf/operator/ui/app.js` — URL router support for `#/job/<ns>/<name>/runs/<epoch>` (additive).
- Modify: `src/aiperf/operator/ui/style.css` — minimal dropdown styling.

- [ ] **Step 1: Inspect dashboard structure**

```bash
ls src/aiperf/operator/ui/
ls src/aiperf/operator/ui/views/
ls src/aiperf/operator/ui/lib/
grep -rn "fetch.*api/v1/results\|/api/v1/results" src/aiperf/operator/ui/ | head -20
grep -rn "hashchange\|window.location.hash" src/aiperf/operator/ui/ | head -5
```

Identify: the job-detail view file, the API client module, the URL-routing pattern (hash fragments).

- [ ] **Step 2: Add API helper**

In `src/aiperf/operator/ui/lib/api.js` (or wherever the fetch helpers live):

```javascript
export async function listRuns(namespace, jobId) {
  const resp = await fetch(`/api/v1/results/${namespace}/${jobId}/runs`);
  if (resp.status === 404) return { runs: [], latestEpoch: null };
  if (!resp.ok) throw new Error(`listRuns failed: ${resp.status}`);
  const body = await resp.json();
  return { runs: body.runs, latestEpoch: body.latest_epoch };
}
```

- [ ] **Step 3: Dropdown + epoch-aware file list**

In the job-detail view:

```javascript
// Parse selected epoch from URL hash; default "latest"
const selectedEpoch = parseEpochFromHash() || "latest";

// Fetch runs + render dropdown
const { runs, latestEpoch } = await listRuns(namespace, jobId);
const dropdown = document.createElement("select");
dropdown.append(createOption("latest", "Latest run"));
for (const run of runs) {
  const label = `${new Date(run.mtime_epoch * 1000).toISOString().replace("T", " ").slice(0, 19)} UTC`
    + (run.is_latest ? " (latest)" : "");
  dropdown.append(createOption(run.epoch, label));
}
dropdown.value = selectedEpoch;
dropdown.addEventListener("change", (e) => {
  const value = e.target.value;
  window.location.hash = value === "latest"
    ? `#/job/${namespace}/${jobId}`
    : `#/job/${namespace}/${jobId}/runs/${value}`;
});

// File list fetch uses selected epoch
const filesUrl = selectedEpoch === "latest"
  ? `/api/v1/results/${namespace}/${jobId}`
  : `/api/v1/results/${namespace}/${jobId}/runs/${selectedEpoch}`;
```

- [ ] **Step 4: URL routing**

Extend the router to match `#/job/:ns/:name/runs/:epoch` in addition to `#/job/:ns/:name`. Epoch flows into the view via props/state.

- [ ] **Step 5: Style nudge**

In `style.css`:

```css
.run-history-select {
  margin-left: 1rem;
  padding: 0.25rem 0.5rem;
  font-family: inherit;
}
```

- [ ] **Step 6: Start dev server and manually verify**

Since the UI is static + fetch-based, open the dashboard, navigate to a job with two runs, confirm the dropdown appears and flips the file list.

If no in-repo playwright/integration coverage of the dashboard exists, skip automated tests. If coverage exists (check `tests/integration/operator/test_dashboard_*.py`), add one test that asserts the dropdown is populated.

- [ ] **Step 7: Verify + commit**

```bash
uv run pytest -n auto tests/unit/
make check-ergonomics && make check-ruff-baselined
git add src/aiperf/operator/ui/
git commit -s -m "$(cat <<'EOF'
feat(ui): add run-history dropdown to job detail view

New <select> populated from /api/v1/results/<ns>/<name>/runs lets users
pin historical epochs; selection routes to #/job/<ns>/<name>/runs/<epoch>
and binds the file list to /runs/<epoch>/. Latest stays the default.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task D: `AIPERF_RESULTS_RETAIN_DAYS` age-based retention

**Files:**
- Modify: `src/aiperf/operator/environment.py` — add `RETAIN_DAYS` field to `_ResultsSettings`.
- Modify: `src/aiperf/operator/results_layout.py` — extend `enforce_retention` signature + logic.
- Modify: `src/aiperf/operator/handlers/completion.py` — pass `retain_days` at the success-gate call.
- Modify: `tests/unit/operator/test_results_layout.py` — append 3 tests.
- Modify: `tests/unit/operator/test_environment.py` — append 2 tests for the new field.

- [ ] **Step 1: Failing tests first**

Append to `tests/unit/operator/test_results_layout.py`:

```python
def test_enforce_retention_age_and_count_both_apply(tmp_path: Path) -> None:
    import time
    from aiperf.operator.results_layout import enforce_retention, run_dir

    now = time.time()
    # 3 runs: one 100-day-old, two recent
    old_epoch, recent1, recent2 = "1700000000", "1714000000", "1714100000"
    for epoch, age_days in [(old_epoch, 100), (recent1, 1), (recent2, 0)]:
        d = run_dir(tmp_path, "ns", "job", epoch)
        d.mkdir(parents=True)
        os.utime(d, (now - age_days * 86400, now - age_days * 86400))
    # keep=10 (everything in count window) AND retain_days=30 (only old_epoch eligible)
    # Intersection: only old_epoch is deleted.
    deleted = enforce_retention(
        tmp_path, "ns", "job",
        keep=10, protect_epoch=recent2, retain_days=30,
    )
    assert deleted == [old_epoch]


def test_enforce_retention_age_only_doesnt_delete_within_count_window(tmp_path: Path) -> None:
    import time
    from aiperf.operator.results_layout import enforce_retention, list_run_epochs, run_dir

    now = time.time()
    epoch = "1700000000"
    d = run_dir(tmp_path, "ns", "job", epoch)
    d.mkdir(parents=True)
    os.utime(d, (now - 100 * 86400, now - 100 * 86400))
    # keep=10 says "keep"; age says "too old". Intersection = keep (conservative).
    deleted = enforce_retention(
        tmp_path, "ns", "job",
        keep=10, protect_epoch=epoch, retain_days=30,
    )
    assert deleted == []
    assert epoch in list_run_epochs(tmp_path, "ns", "job")


def test_enforce_retention_retain_days_zero_disables_age_policy(tmp_path: Path) -> None:
    import time
    from aiperf.operator.results_layout import enforce_retention, list_run_epochs, run_dir

    now = time.time()
    epochs = ["1710000000", "1711000000", "1712000000"]
    for i, epoch in enumerate(epochs):
        d = run_dir(tmp_path, "ns", "job", epoch)
        d.mkdir(parents=True)
        os.utime(d, (now - (i + 1) * 86400, now - (i + 1) * 86400))
    # keep=1 forces reap of two; retain_days=0 = age policy off -> count alone.
    deleted = enforce_retention(
        tmp_path, "ns", "job",
        keep=1, protect_epoch=epochs[-1], retain_days=0,
    )
    assert len(deleted) == 2
```

Append to `tests/unit/operator/test_environment.py`:

```python
def test_retain_days_default_is_zero() -> None:
    from aiperf.operator.environment import _ResultsSettings
    assert _ResultsSettings().RETAIN_DAYS == 0


def test_retain_days_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AIPERF_RESULTS_RETAIN_DAYS", "90")
    from aiperf.operator.environment import _ResultsSettings
    assert _ResultsSettings().RETAIN_DAYS == 90
```

- [ ] **Step 2: Implement**

In `src/aiperf/operator/environment.py` `_ResultsSettings`, after `RETAIN_RUNS`:

```python
    RETAIN_DAYS: int = Field(
        default=0,
        ge=0,
        le=36500,
        description="Age-based retention cap in days. 0 disables age policy. "
        "A run is deleted only when BOTH this age cap AND RETAIN_RUNS "
        "agree the run is outside the keep window; protect_epoch still wins.",
    )
```

In `src/aiperf/operator/results_layout.py` `enforce_retention`:

```python
def enforce_retention(
    base: Path, namespace: str, name: str, *,
    keep: int, protect_epoch: str, retain_days: int = 0,
) -> list[str]:
    parent = job_dir(base, namespace, name)
    if not parent.is_dir():
        return []
    runs = [p for p in parent.iterdir() if p.is_dir() and EPOCH_RE.match(p.name)]
    if not runs:
        return []
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    count_keepers = {r.name for r in runs[:keep]}
    count_keepers.add(protect_epoch)
    deleted: list[str] = []
    if retain_days > 0:
        import time
        cutoff = time.time() - retain_days * 86400
    else:
        cutoff = None
    for r in runs:
        if r.name in count_keepers:
            continue
        if cutoff is not None and r.stat().st_mtime >= cutoff:
            # Age policy says keep (still within retain_days). Intersection wins.
            continue
        try:
            shutil.rmtree(r)
            deleted.append(r.name)
        except OSError as exc:
            logger.warning("retention: failed to remove %s/%s/%s: %s",
                           namespace, name, r.name, exc)
    return deleted
```

In `src/aiperf/operator/handlers/completion.py` success-gate call site (look for the existing `enforce_retention(...)` call):

```python
enforce_retention(
    OperatorEnvironment.RESULTS.DIR,
    namespace, job_id,
    keep=OperatorEnvironment.RESULTS.RETAIN_RUNS,
    protect_epoch=epoch,
    retain_days=OperatorEnvironment.RESULTS.RETAIN_DAYS,
)
```

- [ ] **Step 3: Verify + commit**

```bash
uv run pytest -n auto tests/unit/
make check-ergonomics && make check-ruff-baselined
ruff format ...; ruff check --fix ...
git add src/aiperf/operator/environment.py src/aiperf/operator/results_layout.py src/aiperf/operator/handlers/completion.py tests/unit/operator/test_environment.py tests/unit/operator/test_results_layout.py
git commit -s -m "$(cat <<'EOF'
feat(operator): add AIPERF_RESULTS_RETAIN_DAYS age-based retention

Complements RETAIN_RUNS. A run is deleted only when BOTH policies agree it
is outside the keep window (conservative intersection); protect_epoch still
wins over both. Default 0 disables age policy, preserving current behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task E: env-vars doc generator covers operator/environment.py

**Files:**
- Modify: `tools/generate_env_vars_docs.py` — extend `ENV_FILES` + whatever section-grouping logic needs adjustment.
- Regenerate: `docs/environment-variables.md`.

- [ ] **Step 1: Inspect generator structure**

```bash
wc -l tools/generate_env_vars_docs.py
grep -n "ENV_FILES\|ENV_FILE\|Generator\|section\|GROUP\|subsystem" tools/generate_env_vars_docs.py | head -30
```

Identify the list that drives parsed files, the section/group routing, and whether section names come from class docstrings or a hard-coded mapping.

- [ ] **Step 2: Extend**

Append operator paths:

```python
ENV_FILES = [
    ENV_FILE,
    Path("src/aiperf/common/_env_data.py"),
    Path("src/aiperf/common/_env_network.py"),
    Path("src/aiperf/common/_env_services.py"),
    Path("src/aiperf/operator/environment.py"),
]
```

If the generator needs a section label, add a mapping entry (or derive from the class name — `_ResultsSettings` → "Results", `_MonitorSettings` → "Monitor Timer", `_OperatorEnvironment` → "Operator (root)"). Match the existing mapping convention.

Ensure the generator handles `SettingsConfigDict(env_prefix=...)` — all three operator classes set `env_prefix`. If the parser assumes a single prefix per file, split the three classes into sub-groups.

- [ ] **Step 3: Regenerate + verify**

```bash
make generate-env-vars-docs
grep "AIPERF_RESULTS_RETAIN_RUNS\|AIPERF_OPERATOR_MONITOR_INTERVAL\|AIPERF_DEFAULT_IMAGE" docs/environment-variables.md
```

Expected: all three env vars appear. Spot-check formatting matches sibling sections.

- [ ] **Step 4: Verify + commit**

```bash
uv run pytest -n auto tests/unit/
make check-ergonomics && make check-ruff-baselined
git add tools/generate_env_vars_docs.py docs/environment-variables.md
git commit -s -m "$(cat <<'EOF'
fix(docs): env-vars generator scans operator/environment.py

ENV_FILES now includes src/aiperf/operator/environment.py, so
`make generate-env-vars-docs` picks up operator settings
(AIPERF_DEFAULT_IMAGE, AIPERF_OPERATOR_MONITOR_*, AIPERF_RESULTS_*)
in addition to the common ones.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Execution

- **Wave 1** (parallel, 3 worktreed agents): A, D, E.
- Merge A, D, E back to `ajc/k8s` via cherry-pick once all three complete.
- **Wave 2** (parallel, 2 worktreed agents): B, C.
- Merge B, C back to `ajc/k8s`.

Final gate:
```bash
uv run pytest -n auto tests/unit/
make check-ergonomics && make check-ruff-baselined
git log --oneline origin/main..HEAD | head -20
```

Expect 5 new feature commits + 1 plan commit + 1 spec commit on `ajc/k8s`, no new ergonomics or ruff-baseline violations, full unit suite green.
