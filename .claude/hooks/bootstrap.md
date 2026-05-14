# AIPerf bootstrap — auto-injected on session start

You are working in the **AIPerf** repository. This repo ships a focused `aiperf-*` skill set that you MUST consult before responding to any aiperf-flavored task — including before clarifying questions.

**The iron rule:** if there is even a 1% chance an `aiperf-*` skill applies, invoke it via the `Skill` tool BEFORE responding. Invoke first, decide second.

**The skill set:**

- **`aiperf`** — top-level router. Use when intent is broad or spans categories.
- **Review:** `aiperf-review` (router) → `aiperf-code-review` / `aiperf-re-review` / `aiperf-llm-ergonomics-review`.
- **Runtime testing (executes CLI):** `aiperf-correctness-testing`, `aiperf-adversarial-testing`.
- **Authoring rituals:** `aiperf-add-plugin`, `aiperf-add-service`, `aiperf-add-metric`, `aiperf-add-env-var`, `aiperf-add-cli`.
- **Daily rituals:** `aiperf-pytest` (test invocation), `aiperf-commit` (commit hygiene), `aiperf-debug` (symptom → known gotcha lookup).
- **Workspace utilities:** `aiperf-worktree`, `aiperf-pr-checkout`, `aiperf-mock-server`.
- **Specialized:** `aiperf-profile-export`, `aiperf-integration-test`, `aiperf-merge-from-main`.

**Red flag — STOP:** "I know how to do this without the skill." Invoke the skill anyway. Knowing the concept ≠ using the skill.

Full routing + per-skill details: invoke `aiperf` skill.
