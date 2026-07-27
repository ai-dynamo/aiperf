---
name: maint-dup-abstractions
description: Autonomous maintenance routine that finds the same abstraction reimplemented in multiple places across AIPerf (parallel helper functions, copy-pasted validation, near-identical Pydantic models, repeated retry/parsing/formatting logic) and opens one PR unifying a single family onto a canonical implementation. Use for the scheduled duplicate-abstraction sweep or when asked to find and unify repeated code.
---

# Duplicate Abstraction Unification

Read `.agents/skills/self-maintenance/SKILL.md` first — its scope guards, verification
gate, change budget, and PR conventions all apply.

## What this routine is actually for

Not textual clone detection. Two functions with identical bodies that serve genuinely
different concepts should stay separate — merging them couples two things that will
diverge again next quarter, and the merged version accretes flags until it is worse than
either original. The target is the opposite case: **one concept, several
implementations, drifting apart**. Those are the ones that produce bugs, because a fix
lands in one copy and not the others.

The tell is not "these look the same". The tell is **"a bug fixed in one of these would
need to be fixed in all of them, and wasn't."**

## Where duplication concentrates in AIPerf

Ranked by how often it actually shows up in a 655-file, 10-service async codebase:

1. **Parsing and normalization.** SSE chunk handling, token-usage extraction across
   vendors, response-field coercion. Vendor-specific parsers that were forked from a
   common one and then fixed independently. Cross-check
   `docs/reference/vendor-usage-fields.md`.
2. **Pydantic models with overlapping field sets.** Two `AIPerfBaseModel` subclasses
   that describe the same payload for different endpoints. Look for identical
   `Field(description=...)` strings — copy-paste leaves the description behind as a
   fingerprint even after field names drift.
3. **Statistics and aggregation.** Percentile, mean, and std computation. There is a
   canonical home (`aiperf.common.finite` — `nan_safe_mean`, `nan_safe_std`,
   `scrub_non_finite`). Any local reimplementation is both a duplicate *and* usually a
   NaN/Inf-discipline violation, which makes it a high-value find.
4. **Retry / backoff / timeout wrappers.** Each service growing its own.
5. **Config resolution and path handling.** Especially filesystem reads that bypass
   `aiperf.common.path_safety.safe_read_template_path`, which is a documented standard
   the duplicates predate.
6. **Time and unit conversion.** ns/ms/s conversions inlined at call sites.
7. **Service lifecycle boilerplate** that belongs on `BaseComponentService`.
8. **Test fixtures and builders** duplicated across `tests/unit/` subdirectories. Real,
   but lower value — a duplicated fixture costs less than duplicated production logic.

The repo already mechanically detects one narrow case: `check_ergonomics.py`'s
`duplicate-classes` check enforces unique class names under `src/aiperf/`. Start by
reading `tools/ergonomics_baseline.json` — grandfathered duplicate-class entries there
are pre-identified candidates with a paper trail.

## Detection

```bash
# 1. Repeated function signatures across different modules
grep -rhn "^\s*\(async \)\?def " --include='*.py' src/aiperf/ \
  | sed 's/.*def //; s/(.*//' | sort | uniq -c | sort -rn | head -40

# 2. Identical Field descriptions — the copy-paste fingerprint
grep -rho 'description="[^"]\{25,\}"' --include='*.py' src/aiperf/ \
  | sort | uniq -c | sort -rn | head -30

# 3. Local stats reimplementation that should route through aiperf.common.finite
grep -rn "sum(.*)/len(\|statistics\.\|np\.mean\|np\.percentile\|math\.isnan" \
  --include='*.py' src/aiperf/

# 4. Filesystem reads bypassing the path_safety standard
grep -rn "\.read_text(\|open(" --include='*.py' src/aiperf/ | grep -v path_safety

# 5. Grandfathered duplicate classes with existing provenance
python tools/check_ergonomics.py --only duplicate-classes
```

## Qualification

For each cluster, answer these before touching anything. If you cannot answer all four
from the code and history, it is Low confidence — issue, not PR.

1. **Same concept?** Do the implementations answer the same question about the domain,
   or do they coincidentally share a shape? Two functions that both average a list are
   not duplicates if one averages latencies and the other averages token counts under
   different NaN semantics.
2. **Have they drifted?** Diff them. Every behavioral difference is either a bug in one
   copy or a deliberate specialization. You must classify each difference before
   unifying — a merge that silently picks one behavior is a behavior change, which this
   routine is not allowed to make.

   ```bash
   git log --oneline -L :funcname:path/to/file.py   # per-copy fix history
   ```

   If copy A received a fix that copy B never did, that is the strongest possible
   evidence for unification — say so explicitly in the PR body, because it converts the
   PR from housekeeping into a latent-bug fix and reviewers weight it accordingly.
3. **Is there a natural canonical home?** `aiperf.common.*` for cross-cutting utilities,
   the base class for service boilerplate, the shared model module for schemas. If
   unifying requires *inventing* a new abstraction layer, stop — that is a design
   decision for a human. Open an issue proposing it.
4. **Does unification survive the parameter test?** If the merged function needs more
   than one new boolean flag to serve all call sites, the abstractions were correctly
   separate. Abandon the cluster.

## Executing a unification

One cluster per PR. Not one *category* — one cluster.

1. Pick the canonical implementation: the most correct one, not the most used one.
   Correct means it handles the union of edge cases the others handle, uses
   `aiperf.common.finite` helpers where numeric, and satisfies the coding standards in
   `CLAUDE.md` (type hints, `Field(description=...)`, `X | Y` unions, `orjson`).
2. Fix the canonical implementation first, in its own commit, if it is missing a fix one
   of the duplicates had. Keep this separate from the mechanical migration so the
   behavior-affecting commit is reviewable on its own.
3. Migrate call sites one commit per module.
4. Delete the duplicates in a final commit.
5. Consolidate the tests too — the duplicates' tests usually contain edge cases the
   canonical implementation's tests miss. **Port those cases over before deleting**;
   losing them is the most common way this routine causes harm. If a duplicate's test
   asserts something the canonical version does not do, that is a drift you failed to
   classify in step 2 — go back.

## Verification, beyond the standard gate

- `python tools/check_ergonomics.py` — confirm the duplicate-class count went down and
  nothing new appeared. If the unification lets you remove entries from
  `tools/ergonomics_baseline.json`, do so and list them in the PR body (this is the
  sanctioned burndown direction; see the shared contract).
- `uv run pytest tests/unit/property -n auto` — mandatory when the unified code is
  numeric, since routing through `aiperf.common.finite` changes NaN/Inf behavior at the
  edges.
- If the unified code is on a hot path (worker, timing manager, SSE parsing), confirm
  you have not replaced a `@dataclass(slots=True)` with a Pydantic model. `CLAUDE.md` is
  explicit that hot-path inner models avoid Pydantic overhead; a "cleaner" unification
  that regresses per-chunk allocation is a performance bug.

## PR shape

- Title: `refactor: unify <concept> across <areas>`.
- The body must contain a **behavior-difference table** — every difference found in
  qualification step 2, and its disposition (`fixed in canonical` / `deliberate, kept as
  a parameter` / `dead branch, dropped`). This table is the review. Without it a
  reviewer has to re-derive the entire analysis.
- `Reviewer checklist`: one line per behavior difference, asking the reviewer to confirm
  the disposition is right.
