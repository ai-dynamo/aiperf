#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Local dry-run of the `release-version` job in .github/workflows/fern-docs.yml.
#
# Reproduces the build + strict-validation half of the job against throwaway
# git worktrees of the tag and the docs-website branch, then STOPS before the
# commit/push/publish steps (those need FERN_TOKEN and a bot push and cannot run
# locally). Use this to confirm a versioned snapshot rebuilt from a tag converts
# cleanly and passes the strict link guard.
#
# Usage:
#   tools/fern_release_dryrun.sh [TAG]        # default TAG: v0.9.0
#   tools/fern_release_dryrun.sh v0.9.0
#   KEEP=1 tools/fern_release_dryrun.sh v0.9.0  # keep worktrees for inspection
#
# Requires: git, fern, yq, jq, rsync, python3 (all but yq are usually present;
# install yq with `brew install yq`).

set -euo pipefail

TAG="${1:-v0.9.0}"
DOCS_WEBSITE_REF="${DOCS_WEBSITE_REF:-origin/docs-website}"

# --- preconditions ----------------------------------------------------------

if ! echo "$TAG" | grep -qE '^v[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "error: invalid tag '$TAG' (must be vX.Y.Z, e.g. v0.9.0)" >&2
  exit 2
fi

for tool in git fern yq jq rsync python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    hint=""
    [ "$tool" = "yq" ] && hint="  (install: brew install yq)"
    echo "error: required tool '$tool' not found$hint" >&2
    exit 2
  fi
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "error: tag '$TAG' not found locally. Try: git fetch --tags" >&2
  exit 2
fi

echo "Fetching $DOCS_WEBSITE_REF ..."
git fetch origin docs-website >/dev/null 2>&1 || \
  echo "  warning: could not fetch docs-website; using the local ref if present"
git rev-parse -q --verify "$DOCS_WEBSITE_REF" >/dev/null || {
  echo "error: ref '$DOCS_WEBSITE_REF' not found" >&2
  exit 2
}

# --- isolated worktrees (mirrors the job's two checkouts) -------------------

WORK="$(mktemp -d "${TMPDIR:-/tmp}/fern-release-dryrun.XXXXXX")"
SOURCE_CHECKOUT="$WORK/source-checkout"
DOCS_CHECKOUT="$WORK/docs-checkout"

cleanup() {
  if [ "${KEEP:-0}" = "1" ]; then
    echo "KEEP=1 set; leaving worktrees at $WORK"
    return
  fi
  git worktree remove --force "$SOURCE_CHECKOUT" 2>/dev/null || true
  git worktree remove --force "$DOCS_CHECKOUT" 2>/dev/null || true
  rm -rf "$WORK" 2>/dev/null || true
  git worktree prune 2>/dev/null || true
}
trap cleanup EXIT

echo "Creating worktrees under $WORK ..."
git worktree add --detach "$SOURCE_CHECKOUT" "$TAG" >/dev/null
git worktree add --detach "$DOCS_CHECKOUT" "$DOCS_WEBSITE_REF" >/dev/null

# --- existing-version guard (informational only in a dry-run) ---------------

if [ -d "$DOCS_CHECKOUT/fern/pages-$TAG" ] || [ -f "$DOCS_CHECKOUT/fern/versions/$TAG.yml" ]; then
  echo "note: $TAG already exists on docs-website; the real job's existing-version"
  echo "      guard would stop here unless dispatched with force_rebuild=true."
  echo "      Rebuilding it anyway for dry-run validation."
fi

# --- build versioned pages from the tagged commit ---------------------------

echo "Building fern/pages-$TAG/ from source @ $TAG docs/ ..."
rm -rf "$DOCS_CHECKOUT/fern/pages-$TAG"
mkdir -p "$DOCS_CHECKOUT/fern/pages-$TAG"
rsync -a --exclude='index.yml' \
  "$SOURCE_CHECKOUT/docs/" "$DOCS_CHECKOUT/fern/pages-$TAG/"

# --- pin GitHub links main -> tag (raw markdown, before conversion) ---------

echo "Pinning tree/main and blob/main links to $TAG ..."
find "$DOCS_CHECKOUT/fern/pages-$TAG" -name "*.md" -o -name "*.mdx" | while read -r file; do
  if grep -q "github.com/ai-dynamo/aiperf/tree/main" "$file"; then
    sed -i.bak "s|github.com/ai-dynamo/aiperf/tree/main|github.com/ai-dynamo/aiperf/tree/$TAG|g" "$file"
  fi
done
find "$DOCS_CHECKOUT/fern/pages-$TAG" -name "*.md" -o -name "*.mdx" | while read -r file; do
  if grep -q "github.com/ai-dynamo/aiperf/blob/main" "$file"; then
    sed -i.bak "s|github.com/ai-dynamo/aiperf/blob/main|github.com/ai-dynamo/aiperf/blob/$TAG|g" "$file"
  fi
done
find "$DOCS_CHECKOUT/fern/pages-$TAG" -name "*.bak" -delete

# --- convert with the tag's own md_to_mdx.py --------------------------------

echo "Converting Markdown to Fern MDX with the tag's md_to_mdx.py ..."
python3 "$SOURCE_CHECKOUT/fern/md_to_mdx.py" --dir "$DOCS_CHECKOUT/fern/pages-$TAG"

# --- build versions/<tag>.yml from the tag's index.yml ----------------------

echo "Building fern/versions/$TAG.yml from the tag's index.yml ..."
VERSION_FILE="$DOCS_CHECKOUT/fern/versions/$TAG.yml"
cp "$SOURCE_CHECKOUT/docs/index.yml" "$VERSION_FILE"
yq -i '(.. | select(has("path")).path) |= sub("^([a-zA-Z])", "../pages-'"$TAG"'/${1}")' "$VERSION_FILE"

# --- update docs.yml (skips if the version is already present) --------------

DOCS_FILE="$DOCS_CHECKOUT/fern/docs.yml"
if yq -e ".products[0].versions[] | select(.\"display-name\" == \"$TAG\")" "$DOCS_FILE" >/dev/null 2>&1; then
  echo "docs.yml already lists $TAG; leaving version list unchanged."
else
  echo "Inserting $TAG into docs.yml ..."
  DEV_IDX=$(yq '.products[0].versions | to_entries | map(select(.value."display-name" == "dev")) | .[0].key' "$DOCS_FILE")
  INSERT_IDX=$((DEV_IDX + 1))
  yq -i "
    .products[0].versions |= (
      .[:$INSERT_IDX] +
      [{\"display-name\": \"$TAG\", \"path\": \"./versions/$TAG.yml\", \"slug\": \"$TAG\", \"availability\": \"stable\"}] +
      .[$INSERT_IDX:]
    )
  " "$DOCS_FILE"
  yq -i ".products[0].path = \"./versions/$TAG.yml\"" "$DOCS_FILE"
  yq -i ".products[0].versions[0].path = \"./versions/$TAG.yml\"" "$DOCS_FILE"
  yq -i ".products[0].versions[0].display-name = \"Latest ($TAG)\"" "$DOCS_FILE"
  yq -i ".products[0].versions[0].slug = \"latest\"" "$DOCS_FILE"
  yq -i ".products[0].versions[0].availability = \"stable\"" "$DOCS_FILE"
fi

# --- strict validation guard ------------------------------------------------

WANT_FERN="$(jq -r '.version' "$DOCS_CHECKOUT/fern/fern.config.json")"
HAVE_FERN="$(fern --version 2>/dev/null | tr -d '[:space:]')"
if [ "$WANT_FERN" != "$HAVE_FERN" ]; then
  echo "note: local fern is $HAVE_FERN but docs-website pins $WANT_FERN."
  echo "      Pin locally with: npm install -g fern-api@$WANT_FERN"
fi

echo
echo "Running strict validation guard in $DOCS_CHECKOUT ..."
cd "$DOCS_CHECKOUT"
fern check --warnings --strict-broken-links
fern docs broken-links

echo
echo "DRY RUN OK: $TAG snapshot built and passed the strict guard."
echo "Skipped (cannot run locally): git push origin docs-website; fern generate --docs."
