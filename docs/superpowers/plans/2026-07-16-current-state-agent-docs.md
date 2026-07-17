# Current-State Agent Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared agent instructions describe only the repository's current state.

**Architecture:** Rewrite the shared body once, then apply it to all four required agent files while preserving each file's tool-specific header. Validate byte-identical shared bodies and scan for historical framing.

**Tech Stack:** Markdown, Python repository checks, ripgrep

## Global Constraints

- Preserve current architecture, behavior, commands, coding rules, and verification requirements.
- Remove migration history, porting narratives, former names, deleted systems, implementation provenance, comparisons with retired implementations, aspirations, future work, and gaps.
- Keep the shared body synchronized across all four files.
- Do not commit unless explicitly requested.

---

### Task 1: Rewrite and synchronize the agent instructions

**Files:**
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `.github/copilot-instructions.md`
- Modify: `.cursor/rules/python.mdc`

**Interfaces:**
- Consumes: the current repository architecture and the synchronization convention documented in the files.
- Produces: four agent files with tool-specific headers and one byte-identical, present-tense shared body.

- [ ] **Step 1: Rewrite the shared body**

Retain only factual present-state sections:

1. Product and repository architecture.
2. Supported execution modes and runtime behavior.
3. Crate and module responsibilities.
4. Active extension seams and coding standards.
5. Current build, test, packaging, and usage commands.
6. Current verification and documentation-maintenance requirements.

Delete historical and aspirational framing rather than replacing it with new chronology.

- [ ] **Step 2: Apply the shared body to all four files**

Preserve the YAML frontmatter in `.cursor/rules/python.mdc`. Preserve the SPDX header in every file. Make all content beginning at `# AIPerf` byte-identical.

- [ ] **Step 3: Scan for prohibited framing**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && rg -n -i '\b(migrat(e|ed|ion)|port(ed|ing)?|former|retired|deleted|replaced|legacy|aspirational|future work|not built|gap(s)?)\b' AGENTS.md CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
```

Expected: no historical, migration, aspiration, or gap-oriented statements. Any match retained as a current identifier must be reviewed manually.

- [ ] **Step 4: Verify synchronization**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && python tools/check_agent_files_sync.py
```

Expected: exit code 0.

- [ ] **Step 5: Check documentation guards**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && python tools/check_docs_current.py
```

Expected: exit code 0.
