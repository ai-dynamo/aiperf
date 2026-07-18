#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

npm ci
npm run test
npm run build

REPO="${GITHUB_REPOSITORY:-ajcasagrande/aiperf}"
BRANCH="${GITHUB_PAGES_BRANCH:-gh-pages}"

echo "Publishing $ROOT/dist to $REPO ($BRANCH)..."
npx --yes gh-pages@latest -d dist -r "git@github.com:${REPO}.git" -b "$BRANCH"
