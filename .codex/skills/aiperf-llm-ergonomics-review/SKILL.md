---
name: aiperf-llm-ergonomics-review
description: Review AIPerf code for semantic LLM-ergonomics issues that mechanical checks cannot catch, including error-message usefulness, type specificity, docstring examples, naming discoverability, comment density, convention documentation, and reference-file quality. Use after make check-ergonomics and make check-ruff-baselined pass, before shipping an AIPerf branch or PR, or when the user asks for an ergonomics, agent-friendliness, or LLM-readability review.
---

# AIPerf LLM-Ergonomics Review

## Purpose

AIPerf has two mechanical LLM-ergonomics gates:

- `tools/check_ergonomics.py`: custom AST checks for file size, function size, nesting, keyword-only args, module state, duplicate classes, Pydantic field descriptions, stdlib JSON, and exception-message minimums.
- `tools/ruff_baselined.py`: baselined Ruff checks for PLR0915, PLR0912, C901, TID251, BLE001, S110, S112, ANN201, and D103.

Those checks are the floor. This skill reviews semantic quality that AST checks cannot reliably judge:

1. Error-message informativeness
2. Type-hint descriptiveness
3. Docstring example usefulness
4. Naming disambiguation and synonym discoverability
5. Comment semantic density
6. Convention explicitness
7. Reference-file exemplariness

Every finding must cite a specific file and line, quote the current text, and propose a concrete rewrite. Do not report vague "could be clearer" feedback.

## Preflight

Run these commands in order:

```bash
make check-ergonomics
make check-ruff-baselined
git rev-parse --abbrev-ref HEAD
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
```

If either mechanical check fails, stop the review and report those failures. This skill reviews semantic quality after the mechanical floor is clean.

## Scope

Default scope is files changed on the current branch vs. `origin/main`, restricted to `src/aiperf/`.

Invocation overrides the user may specify:

- `aiperf-llm-ergonomics-review <path>` - scope to a file or subdirectory.
- `aiperf-llm-ergonomics-review PR <number>` - pull the PR diff via `gh pr diff <number>`.
- `aiperf-llm-ergonomics-review full` - review all of `src/aiperf/`, not just changed files.

Only review code added or modified by the branch unless the user explicitly asks for existing-code findings.

Skip tiny changes (<=5 lines total) - the skill's overhead exceeds the signal there.

## Execution Discipline

Use `update_plan` (or the equivalent task tool) with one item per axis - seven items total. Mark only the current axis `in_progress` BEFORE inspecting code; this prevents batching findings across axes. If no plan tool is available, keep the same seven-axis checklist in your notes.

Work the axes in order. For each axis:

1. Produce a findings list in memory first, then format into the report.
2. Do not batch findings across axes - per-axis isolation is what makes the report scannable for the fixer.
3. Do not skip an axis because "there is nothing to find" - every axis gets a section in the final report, even if the section says "no findings on this axis".

Do not edit code during this review. The deliverable is a markdown report under `artifacts/`.

## Axis 1 - Error-Message Informativeness

Review every `raise <Exception>(...)` site added or modified.

Check whether the message:

- Names the operation that failed.
- Names the specific input, identifier, file, row, field, plugin, dataset, or endpoint involved.
- Suggests a next step or likely cause when non-obvious.
- Preserves original exception context with `raise ... from exc` when wrapping.
- Uses meaningful f-string values instead of opaque object reprs or discarded exception details.

High-severity examples:

- `raise ValueError("bad input")`
- `raise RuntimeError("validation failed")`
- `raise ConfigError(f"error: {type(exc).__name__}")`
- Wrapping an exception without `from exc`.

For each finding, record file:line, severity, current text, and a verbatim suggested rewrite.

## Axis 2 - Type-Hint Descriptiveness

Review every public function signature added or modified. Skip private helpers unless they define a cross-file contract.

Check whether:

- `Any` can be replaced with a concrete model, protocol, dataclass, `TypedDict`, or generic.
- Fixed string domains should be `Literal[...]` or an enum instead of plain `str`.
- Containers are parameterized, e.g. `list[RequestInfo]`, not `list`.
- Project convention uses `X | None` and `X | Y`, not `Optional[X]` or `Union[X, Y]`.
- `Callable` types specify argument and return types.
- `*args: Any` or `**kwargs: Any` hide a narrower real contract.

High-severity examples:

- `-> Any` on non-passthrough code.
- `-> dict` or `-> list` with no type parameters.
- Plain `str` for a small closed set of valid values.

For each finding, record file:line, current signature, and a suggested signature.

## Axis 3 - Docstring Example Usefulness

Review public functions, classes, methods, and services added or modified.

Check whether docstrings:

- Include a realistic runnable example when the public API is non-obvious.
- Match the current signature.
- Use meaningful AIPerf-like identifiers, not `foo`, `bar`, or `x`.
- Explain side effects such as publishing messages, writing files, mutating state, or registering plugins.
- Document invariants on public classes or services.
- Include useful `Raises:` content for project-specific exceptions.

Medium-severity examples:

- A one-line restatement such as "Parses the config."
- An `Args:` block that says only `x: the x parameter`.
- A public class or method with no docstring. `D103` only catches public **functions**; `D100`/`D101`/`D102`/`D107` (module/class/method/`__init__` docstrings) are not enabled, so missing docstrings on public classes and methods are Axis 3 findings.

Do not require examples for every trivial method. Flag cases where an agent would likely misuse the API without an example.

## Axis 4 - Naming Disambiguation And Synonym Discoverability

Review new or renamed classes, public functions, module paths, constants, and config fields.

Search the repo for similar concepts before flagging. Look for near-collisions such as:

- Client, user, customer, session
- Metric, record, result, sample
- Worker, service, component, process
- Credit, request, turn, conversation

Check whether:

- The name describes behavior for functions and identity for classes.
- Domain acronyms are expanded on first use in a docstring or nearby comment.
- Synonyms are mentioned where external docs or users use different terms.
- Names avoid generic suffixes like `Helper`, `Manager`, `Utility`, or `Handler` unless the role is specific.
- Names do not collide with stdlib or framework concepts in a misleading way.

Medium-severity examples:

- Class name ends in `Helper`, `Manager`, `Utility`, or `Handler` without a specific role (`StatsHandler` is a shrug; `StatsFlushHandler` is specific).
- Name is technically unique but is a plural where the singular exists elsewhere (`Workers` the class vs `Worker` the class — agents will grep wrong).
- Acronym used without first-use expansion in a docstring (`CR`, `CRD`, `PPT`, `OSL`).
- A synonym exists in the repo or external docs and is not mentioned in a docstring or comment.

Cross-reference: `check_ergonomics.py`'s duplicate-class-name check catches exact collisions. This axis catches the near-misses.

For each finding, record the search evidence and the concrete rename or clarifying doc/comment text.

## Axis 5 - Comment Semantic Density

Review every added or modified comment.

Good comments explain non-local constraints, past bugs, protocol invariants, or why the obvious alternative is wrong. Bad comments restate the code.

Flag comments that:

- Explain what the next line already says.
- Use `TODO`, `HACK`, or `FIXME` with no issue, owner, date, or concrete next action.
- Reference stale code, stale line numbers, or deleted behavior.
- Could be eliminated by a better variable, function, or class name.
- Preserve dead code as comments.

Low/Medium-severity examples:

- `# increment counter` above `counter += 1` — restates the code.
- `# was: old_logic()` — dead-code comment; git log is authoritative.
- `# TODO: fix this` with no owner, issue link, or concrete next step.
- Copy-pasted comment from another file that does not fit here.

A clean diff with zero comment findings is acceptable. Do not manufacture comment findings - AIPerf's convention (in `CLAUDE.md`) is that comments are genuinely optional and only needed to explain "why".

## Axis 6 - Convention Explicitness

Review every new pattern introduced by the branch.

A new pattern includes a new service type, plugin category, message-handler style, exception hierarchy, lifecycle invariant, config convention, test convention, or external integration pattern that future agents will need to copy.

Check whether:

- The pattern is documented in `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc`, or `docs/dev/patterns.md`.
- The invariant is stated where an agent will find it, not only in one implementation.
- Replacements for older patterns are documented and stale patterns are removed or marked deprecated.
- The four synced agent files stayed aligned when one of them changed.

AIPerf's sync rule: `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` must contain identical content except for platform-specific headers/frontmatter.

High-severity examples:

- A load-bearing pattern used in two or more files with no agent-facing documentation.
- One synced agent file updated without the other three.
- Docs still tell agents to use a deprecated pattern.

## Axis 7 - Reference-File Exemplariness

`docs/dev/patterns.md` is AIPerf's canonical pattern catalog. Its code blocks use leading path comments such as `# aiperf/cli.py` or `# cli_commands/foo.py` to identify reference implementations.

Check whether:

- Any reference files named in `docs/dev/patterns.md` were modified by the branch.
- Modified reference files still exemplify the documented pattern.
- Snippets in `docs/dev/patterns.md` are stale after branch changes.
- New exemplary implementations should be added to `docs/dev/patterns.md`.
- Reference files are not newly grandfathered in `tools/ruff_baseline.json` or `tools/ergonomics_baseline.json`.
- Any `# noqa` in a reference file has a narrow rule and an explanatory comment.

Use a permissive search for reference paths:

```bash
grep -nE '^# [a-z_/.]+\.py' docs/dev/patterns.md
```

Resolve paths against both repo root and `src/`.

High-severity examples:

- `docs/dev/patterns.md` shows `aiperf/foo.py` demonstrating a pattern, and this branch changed `foo.py` in a way that breaks the shown example.
- A reference file has new entries in `tools/ruff_baseline.json` or `tools/ergonomics_baseline.json` - it is now grandfathering violations of the rules it is supposed to teach.
- An unexplained `# noqa: <RULE>` in a reference file. Narrow, commented `# noqa` (e.g. `# noqa: BLE001 - fault-tolerant telemetry`) is acceptable; a bare `# noqa` in a gold-standard file is a finding.

## Report

Write the report to:

```text
artifacts/code-review-YYYY-MM-DD/llm-ergonomics-<branch-slug>.md
```

Use the current local date. Replace slashes in the branch name with hyphens.

Use a readable PR-comment style. Avoid wide finding tables; they become hard to
read in GitHub comments. Keep the counts compact, then give one section per
finding.

Required structure:

````markdown
# LLM-Ergonomics Review - <branch> @ YYYY-MM-DD

I reviewed <N> changed `src/aiperf/` files on <branch-or-commit>.

Model used: <specific model/version and settings, e.g. "GPT-5.5 xhigh" or "Claude Opus 4.7 (1M context) max">

## Preflight

- `make check-ergonomics`: exit 0
- `make check-ruff-baselined`: exit 0
- Scope: <N> files changed vs origin/main

## Summary

I found <N> LLM-ergonomics issues:

- High: <h>
- Medium: <m>
- Low: <l>

Top issue: <one sentence naming the highest-priority finding and why it matters>.

## Findings

### <Severity> - <Short Finding Title>

Axis: <axis number and name>

File: `<path>:<line>`

Current code/text:

```python
<quote the current code or text>
```

Why this matters:

<Explain the semantic risk in 1-3 sentences.>

Suggested rewrite:

```python
<verbatim concrete rewrite>
```

<details>
<summary>Prompt for AI Agents</summary>

```
Verify each finding against the current code and only fix it if needed.

In `@<path>` around lines <line-start> - <line-end>, re-open the current code
and confirm <finding-specific condition> still applies before making changes.
Inspect <related symbols, call sites, tests, docs, or generated files>. If
confirmed, make the smallest change that <finding-specific semantic fix>, while
preserving <relevant AIPerf pattern or local convention>. Treat the suggested
rewrite as guidance, not a patch to apply blindly. Update the specific affected
tests, docs, or generated files. Do not perform unrelated refactors while
addressing this finding.
```

</details>

Repeat one section per finding, ordered by severity, then by file.

## Axes With No Findings

- Error messages: <brief note or omit if this axis has findings>
- Type hints: <brief note or omit if this axis has findings>
- Docstring examples: <brief note or omit if this axis has findings>
- Naming: <brief note or omit if this axis has findings>
- Comments: <brief note or omit if this axis has findings>
- Conventions: <brief note or omit if this axis has findings>
- Reference files: <brief note or omit if this axis has findings>

## Prioritized Action List

1. `<path>:<line>` - <highest-priority action>.
2. ...

## What Looks Good

- <High-signal positive observation, especially where an axis is clean.>
````

Every finding must include severity, axis, file:line, quoted current text, why it matters, a concrete suggested rewrite, and a collapsed `Prompt for AI Agents` block with a fenced prompt immediately after the suggested rewrite. Every axis must be accounted for either by a finding or by the "Axes With No Findings" section. High-severity findings must include a verbatim suggested rewrite.

## Anti-Patterns (Self-Check Before Submitting)

| If you find yourself... | Stop because... |
|---|---|
| ...skimming rather than reading each `raise` site | This skill exists because skimming misses what matters. Slow down. |
| ...saying "most of these are fine" | Run the checklist literally. Quote the current text; propose a rewrite. |
| ...flagging things the mechanical tools already catch | Those are not in scope. If you are duplicating BLE001/ANN201/D103 findings, you are in the wrong document. |
| ...writing "this could be clearer" | Insufficient. Give the exact rewrite or do not flag it. |
| ...batching findings across axes | Per-axis isolation is what makes the report scannable for the fixer. |
| ...editing code | This is REVIEW only. Do not fix during the pass. Findings go to the report; fixes are the user's call. |
| ...claiming done without writing the report | The deliverable is the markdown file. No report = not done. |

## Finish Criteria

Before reporting complete:

- The report exists at the expected path.
- All seven axes are accounted for.
- Counts in the summary match the findings.
- Every finding ends with a collapsed `Prompt for AI Agents` block with a fenced prompt after the suggested rewrite.
- High-severity findings have concrete rewrites.
- The action list is ordered by severity and grouped by file when practical.
- The user gets the report path and the top-priority issue.
- Code remains unchanged.

## Research Basis

The axes are based on agent-friendly software engineering principles from:

- arxiv 2604.07502, "Beyond Human-Readable": semantic density, compression paradox, naming conventions.
- arxiv 2504.09246, "Type-Constrained Code Generation with Language Models": type constraints reduce model coding errors.
- Marmelab 2026, "Agent Experience": examples, synonyms in comments, reference files.
- Stack Overflow 2026, "Building shared coding guidelines for AI": explicit conventions and gold-standard examples.
- Missing Semester MIT 2026 and Honeycomb: `AGENTS.md`, `CLAUDE.md`, and similar session-bootstrap docs.
- Anthropic, "Effective Context Engineering for AI Agents": compact, high-signal context and agent workflows.
