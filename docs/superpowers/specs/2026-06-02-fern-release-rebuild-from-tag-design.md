# Fern `release-version`: Rebuild Version Snapshots From the Tagged Commit

**Date:** 2026-06-02
**Status:** Approved (design)
**Scope:** `.github/workflows/fern-docs.yml` — the `release-version` job only.

## Problem

When a `vX.Y.Z` tag is pushed, the `release-version` job builds the versioned
docs snapshot (`fern/pages-vX.Y.Z/`) by copying `fern/pages-dev/` on the
`docs-website` branch:

```bash
cp -r fern/pages-dev "fern/pages-$TAG"
```

`fern/pages-dev/` is continuously re-synced from `main` by the
`validate-and-publish-docs` job on every push to `main`. Therefore the snapshot
captures **`main` at tag-processing time**, not the docs at the tagged commit.

Tags are cut from `release/X.Y.Z` branches that can diverge from `main`
(cherry-picks, backports, doc fixes that never land on `main`, or simply `main`
having advanced). `main`/`pages-dev` never tracks the release branch, so the
snapshot is, in the general case, **the wrong content under a version label**.

Secondary defect: because `pages-dev` is already MDX (the
`validate-and-publish-docs` job runs `md_to_mdx.py` on it), the job's
subsequent `md_to_mdx.py` pass on the copy is a **double conversion**.

## Goal

The `release-version` job must build `fern/pages-vX.Y.Z/` and its navigation
from the **docs as they exist at the tagged commit**, not from `pages-dev`.

## Decisions (from brainstorming)

1. **Rebuild from the tagged commit.** Mandatory — `main` never tracks the
   release branch.
2. **Fix forward only.** Existing snapshots (`v0.7.0`, `v0.8.0`, `v0.9.0`,
   built from dev) are corrected manually out-of-band; this design does not
   rebuild them.
3. **Use the tag's own `md_to_mdx.py`** to convert the snapshot, for maximum
   fidelity to that release's pipeline. Accepted risk: an older converter
   feeding the site's newer renderer may emit MDX that does not render cleanly.
4. **Rendering stays site-wide.** The published site is a single
   `fern generate --docs` build using one Fern CLI version (from
   `docs-website/fern/fern.config.json`, currently `5.37.10`). There is no
   per-version renderer; the tag's `fern.config.json` version (e.g. `v0.9.0`
   pins `4.80.3`) does **not** apply to the published render. Only the markdown
   content, navigation, and `md_to_mdx.py` conversion are version-pinned.
5. **Add a whole-site validation guard** before commit/publish (see below),
   accepting that it validates all versions and will block a release if any
   snapshot (including pre-existing ones) has broken links.

## Design

### Checkouts

The job currently checks out only `docs-website`. It gains a second checkout
of the source at the tag:

- `source-checkout/` — `actions/checkout` with `ref: ${{ steps.version.outputs.tag }}`,
  `fetch-depth: 1`. Provides the tagged commit's `docs/` (incl. `docs/index.yml`)
  and `fern/md_to_mdx.py`. Explicit `ref` is required so `workflow_dispatch`
  (which runs on a branch) resolves the tag rather than the dispatch ref.
- `docs-checkout/` — `actions/checkout` with `ref: docs-website` (the target
  branch, as today), with write token.

### Build content (replaces `cp -r pages-dev`)

```bash
rm -rf docs-checkout/fern/pages-$TAG
mkdir -p docs-checkout/fern/pages-$TAG
rsync -a --exclude='index.yml' \
  source-checkout/docs/ docs-checkout/fern/pages-$TAG/
```

This mirrors the `validate-and-publish-docs` sync of `pages-dev`, but sourced
from the tag and targeting `pages-$TAG`. Because the source is **raw markdown**
(not already-converted MDX), the later `md_to_mdx.py` pass converts exactly
once — the double-conversion defect disappears.

### Build navigation (replaces copy-from-`dev.yml`)

```bash
cp source-checkout/docs/index.yml docs-checkout/fern/versions/$TAG.yml
# transform paths for docs-website layout, pinned to this version's pages
yq -i '(.. | select(has("path")).path) |= sub("^([a-zA-Z])", "../pages-'"$TAG"'/${1}")' \
  docs-checkout/fern/versions/$TAG.yml
```

Navigation must come from the **tag's** `index.yml` so it references the pages
that actually exist in `pages-$TAG` (not whatever pages `dev` currently has).
The path transform is the same one `validate-and-publish-docs` applies when
building `versions/dev.yml`, with the `../pages-$TAG/` prefix instead of
`../pages-dev/`.

### Link pinning (order: rewrite, then convert)

Operating on `docs-checkout/fern/pages-$TAG/`, before MDX conversion:

```bash
# tag source uses blob/main + tree/main (repo convention); pin to this version
find ... -exec sed -i 's|github.com/ai-dynamo/aiperf/blob/main|.../blob/$TAG|g'
find ... -exec sed -i 's|github.com/ai-dynamo/aiperf/tree/main|.../tree/$TAG|g'
```

Same logic as today, now correctly operating on tag-sourced content.

### Convert with the tag's converter

```bash
python3 source-checkout/fern/md_to_mdx.py --dir docs-checkout/fern/pages-$TAG
```

Uses the tagged commit's `md_to_mdx.py` (decision 3), not `docs-checkout`'s.

### Update `docs.yml`

Unchanged — insert the new version after `dev`, repoint `Latest`. (Existing
`yq` logic.)

### Validation guard (new, before commit/push/publish)

After assembling the snapshot and updating `docs.yml`, and before committing:

```bash
# install site Fern CLI (from docs-checkout, the publish env)
npm install -g fern-api@$(jq -r '.version' docs-checkout/fern/fern.config.json)
cd docs-checkout
fern check --warnings --strict-broken-links
fern docs broken-links
```

If the tag's converter produced output the site renderer rejects, or any link
is broken, the **release fails before committing or publishing**. This guard
validates the whole site (all versions); pre-existing broken snapshots must be
clean (handled manually per decision 2) or the release is blocked — accepted as
a "never publish broken links" property.

### Existing-version guard

The job refuses to release a version whose `pages-$TAG/` or `versions/$TAG.yml`
already exists on `docs-website` (append-only by default). Because the rebuild is
now deterministic from the tag, a re-run would otherwise silently overwrite an
existing snapshot — including the out-of-band manual corrections to the legacy
`v0.7.0`/`v0.8.0`/`v0.9.0` snapshots (decision 2). The guard prevents that.

A `force_rebuild` boolean `workflow_dispatch` input provides a deliberate,
logged escape hatch: when `true`, the guard emits a `::warning::` and proceeds,
overwriting the existing snapshot. The input is empty on tag-push events, so the
guard is never bypassable on an automated tag release — only on a manual dispatch
where someone explicitly opts in.

### Commit, push, publish

Unchanged from today (commit `pages-$TAG/` + `versions/$TAG.yml` + `docs.yml`
to `docs-website`, then `fern generate --docs`).

## Step Sequence (after redesign)

1. Determine version tag *(unchanged)*
2. Checkout source @ tag → `source-checkout/` *(NEW)*
3. Checkout `docs-website` → `docs-checkout/` *(now pathed)*
4. Check if version already exists *(operates on `docs-checkout`; bypassable via
   the `force_rebuild` workflow_dispatch input to intentionally overwrite an
   existing snapshot — see "Existing-version guard" below)*
5. Setup Git *(unchanged)*
6. Build content: rsync tag `docs/` → `pages-$TAG/` *(REPLACES `cp pages-dev`)*
7. Link pinning: `sed` blob/tree main→$TAG *(unchanged logic)*
8. Convert: tag's `md_to_mdx.py` on `pages-$TAG/` *(now single conversion)*
9. Build `versions/$TAG.yml` from tag `index.yml` *(REPLACES copy-from-dev.yml)*
10. Update `docs.yml` *(unchanged)*
11. Validation guard: site Fern strict check + broken-links *(NEW)*
12. Commit + push to `docs-website` *(unchanged)*
13. Setup Node + install Fern + publish *(unchanged)*

## Non-Goals

- Rebuilding existing `v0.7.0`/`v0.8.0`/`v0.9.0` snapshots.
- Changing how `pages-dev` is synced (the `validate-and-publish-docs` job).
- Per-version Fern CLI rendering (architecturally a single site build).
- Any change to source repo `docs/` or `fern/` tooling.

## Risks

- **Old converter / new renderer mismatch** (accepted, decision 3) — mitigated
  by the Section "Validation guard" failing the release loudly.
- **Whole-site guard couples releases to old-snapshot cleanliness** — accepted;
  pre-existing debt is fixed manually.
- **Tag must carry `fern/md_to_mdx.py` and `docs/index.yml`** — verified present
  at `v0.7.0`/`v0.8.0`/`v0.9.0`; a tag missing them would fail fast at the
  convert/build step (acceptable: such a tag could not be released anyway).

## Verification

- Dry-run via `workflow_dispatch` with `tag: v0.9.0` against a scratch
  docs-website-like branch (or with the existing-version guard temporarily
  bypassed) to confirm `pages-v0.9.0/` rebuilt from the tag matches the
  manually-corrected snapshot and passes the strict guard.
- Confirm `versions/v0.9.0.yml` navigation references only pages present in the
  rebuilt `pages-v0.9.0/`.
