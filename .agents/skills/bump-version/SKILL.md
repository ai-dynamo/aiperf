---
name: bump-version
description: Bump the AIPerf package version (e.g. 0.10.0 -> 0.11.0) across pyproject.toml, the mock server, and the version strings shown in docs, then open a PR to main
disable-model-invocation: true
allowed-tools: Bash(git fetch *), Bash(git checkout *), Bash(git branch *), Bash(git status *), Bash(git add *), Bash(git commit *), Bash(git push *), Bash(git log *), Bash(grep *), Bash(pre-commit run *), Bash(gh pr create *), Read, Edit, AskUserQuestion
---

# Bump AIPerf Version

Bump the AIPerf package version everywhere it is hard-coded, then open a PR to `main`.

The runtime version is read dynamically from package metadata
(`importlib.metadata.version("aiperf")` in `src/aiperf/__init__.py`), so
`pyproject.toml` is the single source of truth. Every other occurrence is a
copy in a doc example or in the mock-server package and must be updated by hand
to stay in sync.

## Inputs

- **Target version** (e.g. `0.11.0`). If the user did not provide one, read the
  current version from `pyproject.toml` and use `AskUserQuestion` to confirm the
  target (default: next minor).

## Steps

1. **Confirm you are starting from an up-to-date `main`:**
   ```bash
   git fetch origin
   git checkout main
   git status
   ```
   Only proceed if the working tree is clean (ignore untracked files that are
   not part of the bump).

2. **Find the current version:**
   Read the `version` field under `[project]` in `pyproject.toml`. Call this
   `OLD_VERSION` and the requested version `NEW_VERSION`.

3. **Create the branch:**
   Use the `<username>/chore-update-version-<NEW_VERSION>` convention, e.g.
   ```bash
   git checkout -b harrison/chore-update-version-0.11.0
   ```

4. **Find every hard-coded occurrence of `OLD_VERSION`:**
   ```bash
   grep -rn --include="*.py" --include="*.toml" --include="*.md" \
     "<OLD_VERSION>" . | grep -v "/.git/"
   ```
   Review each hit. Update only strings that mean "the AIPerf version". Do NOT
   touch unrelated version-like strings (dependency pins, `schema_version`,
   sample UUIDs, etc.).

   The canonical set of files (as of 0.10.0 -> 0.11.0) is:
   - `pyproject.toml` — `version = "..."` under `[project]`
   - `tests/aiperf_mock_server/pyproject.toml` — `version = "..."` under `[project]`
   - `docs/plugins/creating-your-first-plugin.md` — the `Version: ...` line in the
     `pip show` example output
   - `docs/server-metrics/server-metrics-json-schema.md` — every
     `"aiperf_version": "..."` JSON example and the `(e.g., "...")` description
     (3 occurrences)

   If the grep surfaces a NEW occurrence not in this list, update it too and add
   it to this skill's canonical list in the same PR.

5. **Do NOT update `uv.lock`:**
   `uv.lock` is gitignored and not tracked, so it is not part of the bump.
   Tests that need the version import `aiperf.__version__` rather than hard-coding
   it; leave those alone.

6. **Verify nothing stale remains:**
   ```bash
   grep -rn "<OLD_VERSION>" pyproject.toml tests/aiperf_mock_server/pyproject.toml \
     docs/plugins/creating-your-first-plugin.md \
     docs/server-metrics/server-metrics-json-schema.md
   ```
   This must return no matches. Then confirm `NEW_VERSION` is present in each.

7. **Run pre-commit on the staged files:**
   ```bash
   pre-commit run
   ```
   Fix anything it flags before committing.

8. **Commit the version bump (commit 1):**
   Stage only the version files explicitly (never `git add -A` — the working
   tree may hold unrelated untracked files):
   ```bash
   git add pyproject.toml tests/aiperf_mock_server/pyproject.toml \
     docs/plugins/creating-your-first-plugin.md \
     docs/server-metrics/server-metrics-json-schema.md
   git commit -s -F - <<'EOF'
   chore: bump aiperf version to <NEW_VERSION>

   Updates the package and mock-server pyproject.toml versions and refreshes
   the version strings shown in the plugin tutorial example and the
   server-metrics JSON schema example output.

   Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
   EOF
   ```

9. **Push the branch:**
   ```bash
   git push -u origin HEAD
   ```

10. **Open the PR to `main`:**
    ```bash
    gh pr create \
      --repo ai-dynamo/aiperf \
      --base main \
      --head harrison/chore-update-version-<NEW_VERSION> \
      --title "chore: bump aiperf version to <NEW_VERSION>" \
      --body "$(cat <<'EOF'
    ## Summary
    - Bump AIPerf package version from `<OLD_VERSION>` to `<NEW_VERSION>`.
    - Updates `pyproject.toml`, the mock-server `pyproject.toml`, and the version
      strings in the plugin tutorial and server-metrics JSON schema docs.

    The runtime version is read dynamically from package metadata, so
    `pyproject.toml` is the source of truth; the other files are doc/example copies.

    ## Test plan
    - `grep` confirms no `<OLD_VERSION>` strings remain in the bumped files.
    - Pre-commit hooks pass.
    EOF
    )"
    ```
    If `gh` is unavailable, print the manual compare URL:
    `https://github.com/ai-dynamo/aiperf/compare/main...harrison/chore-update-version-<NEW_VERSION>`

11. **Print a summary** with the old/new version and the PR link.

## Rules

- NEVER manually add a `Signed-off-by` line — `git commit -s` adds it.
- NEVER use `--no-verify` or `git commit -n`.
- NEVER `git add -A` / `git add .`; stage the version files by explicit path so
  unrelated untracked files stay out of the commit.
- NEVER touch `uv.lock` (gitignored) or non-AIPerf version-like strings.
- Branch from an up-to-date `main`; one PR = one concern (just the bump).
