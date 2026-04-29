---
name: docs-to-fern
description: Migrate a plain Markdown docs/ folder to a Fern documentation site from scratch. Use this skill when a project has no existing publishing framework and needs to scaffold Fern config, migrate content, and build navigation.
---

# Docs to Fern Migration

Migrate a plain Markdown `docs/` folder to a fully functional [Fern](https://buildwithfern.com) documentation site. This skill covers everything from scaffolding the Fern project to publishing.

**Assumes:**
- Source is plain Markdown files (no Sphinx, no RST)
- No existing Fern setup in the repo
- NVIDIA branding (colors, logos, SPDX headers)

**For Sphinx-to-Fern migrations**, see the `fern-migration` skill instead.

---

## Phase 0: Discover and Inventory

Before migrating, analyze the current `docs/` folder.

### Step 0.1: Inventory Source Files

```bash
# Count Markdown files
find docs -type f -name "*.md" | wc -l

# List all files
find docs -type f -name "*.md" | sort

# List top-level folders
ls -d docs/*/

# Find all images
find docs -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.svg" -o -name "*.gif" \) | sort
```

### Step 0.2: Check for Naming Issues

```bash
# Files with underscores (need renaming to hyphens)
find docs -type f -name "*_*.md"

# Files with uppercase names (Fern prefers lowercase-hyphen)
find docs -type f -name "*.md" | grep '[A-Z]' | grep -v README

# Images with underscores
find docs -type f \( -name "*_*.png" -o -name "*_*.jpg" -o -name "*_*.svg" \)
```

### Step 0.3: Detect MDX-Breaking Patterns

```bash
# HTML comments (will break MDX)
grep -rl '<!--' docs/ --include="*.md" | wc -l

# Bare < in prose (will break MDX)
grep -rn '<[a-zA-Z0-9]' docs/ --include="*.md" | grep -v '```' | grep -v 'http' | wc -l

# <details>/<summary> HTML blocks
grep -rl '<details>' docs/ --include="*.md" | wc -l

# Blockquote admonitions (> **Note**)
grep -rn '> \*\*Note' docs/ --include="*.md" | wc -l
```

### Step 0.4: Generate Migration Summary

```markdown
## Migration Summary for [PROJECT]

### Source
- Total Markdown files: X
- Images: X (png: X, jpg: X, svg: X)
- Top-level folders: X

### Naming Issues
- Files with underscores: X
- Files with uppercase: X
- Images with underscores: X

### MDX Issues to Fix
- HTML comments: X files
- Bare angle brackets: X occurrences
- <details> blocks: X files
- Blockquote admonitions: X occurrences

### Estimated Effort
- ~X files to migrate
- ~X images to copy
- ~X MDX fixes needed
```

---

## Phase 1: Scaffold Fern Project

Create the Fern directory structure and all required config files from scratch.

### Step 1.1: Create Directory Structure

```bash
mkdir -p fern/pages
mkdir -p fern/assets/img
mkdir -p fern/versions
```

### Step 1.2: Create `fern/fern.config.json`

This file identifies your organization and pins the Fern CLI version:

```json
{
    "organization": "YOUR_PROJECT_NAME",
    "version": "3.29.1"
}
```

Replace `YOUR_PROJECT_NAME` with your Fern organization name (lowercase, hyphens ok). To find the latest CLI version, run `npm show fern-api version`.

### Step 1.3: Create `fern/docs.yml`

This is the main configuration file controlling theme, branding, and site structure:

```yaml
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

instances:
  - url: YOUR_PROJECT.docs.buildwithfern.com

title: NVIDIA YOUR_PROJECT Documentation

# Version configuration
versions:
  - display-name: Next
    path: ./versions/next.yml

# GitHub repository link in navbar
navbar-links:
  - type: github
    value: https://github.com/YOUR_ORG/YOUR_REPO

# NVIDIA branding colors
colors:
  accent-primary:
    dark: "#76B900"
    light: "#4A7300"
  background:
    dark: "#1A1A1A"
    light: "#FFFFFF"

# Logo and favicon
logo:
  href: /
  light: ./assets/img/nvidia-logo.svg
  dark: ./assets/img/nvidia-logo-dark.svg
  height: 50

favicon: ./assets/img/favicon.png
```

**Replace these placeholders:**

| Placeholder | Example |
|-------------|---------|
| `YOUR_PROJECT` | `dynamo`, `nemo`, `triton` |
| `YOUR_ORG/YOUR_REPO` | `ai-dynamo/dynamo` |

**Required assets:** You need these files in `fern/assets/img/`:
- `nvidia-logo.svg` (light mode logo)
- `nvidia-logo-dark.svg` (dark mode logo)
- `favicon.png`

Copy these from an existing NVIDIA Fern project or request from your design team.

### Step 1.4: Create `fern/versions/next.yml`

Start with a minimal navigation skeleton. You will fill this in during Phase 5:

```yaml
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

navigation:
  - page: Home
    path: ../pages/index.md
```

### Step 1.5: Create a Placeholder Home Page

```bash
cat > fern/pages/index.md << 'EOF'
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

## Welcome

Documentation site is under construction.
EOF
```

### Verify Directory Structure

After scaffolding, your repo should look like:

```
fern/
├── assets/
│   └── img/
│       ├── favicon.png
│       ├── nvidia-logo.svg
│       └── nvidia-logo-dark.svg
├── docs.yml
├── fern.config.json
├── pages/
│   └── index.md
└── versions/
    └── next.yml
```

---

## Phase 2: Install and Verify Fern

### Step 2.1: Install Fern CLI

```bash
# Requires Node.js 18+
npm install -g fern-api

# Verify installation
fern --version
```

### Step 2.2: Validate Configuration

```bash
cd /path/to/your-repo
fern check --warnings
```

This should pass with zero errors on the empty scaffold. If it fails, check:
- `fern.config.json` has valid JSON
- `docs.yml` has valid YAML
- `next.yml` references an existing page file
- Logo/favicon files exist at the paths specified in `docs.yml`

### Step 2.3: Local Preview

```bash
fern docs dev --port 3000
```

Open `http://localhost:3000`. You should see the skeleton site with the NVIDIA branding, navbar, and your placeholder home page. If this works, the scaffold is correct and you can proceed to content migration.

---

## Phase 3: Migrate Content

### Step 3.1: Bulk Copy with Hyphen Renaming

Copy all Markdown files from `docs/` to `fern/pages/`, converting underscores to hyphens:

```bash
#!/usr/bin/env bash
# Run from repo root. Copies docs/ to fern/pages/ with hyphen naming.

find docs -type f -name "*.md" | while read -r src; do
    # Build target path: docs/foo/bar_baz.md -> fern/pages/foo/bar-baz.md
    rel="${src#docs/}"
    target="fern/pages/$(echo "$rel" | tr '_' '-')"
    mkdir -p "$(dirname "$target")"
    cp "$src" "$target"
    echo "Copied: $src -> $target"
done
```

**Exception:** Keep `README.md` as-is (do not rename to `r-e-a-d-m-e.md`).

### Step 3.2: Update Heading Hierarchy

Fern auto-generates h1 from the navigation title. All page content should start at h2:

```bash
# Find files that start with h1
grep -rl '^# ' fern/pages/ --include="*.md" | head -20
```

For each file, remove or downgrade the first `# Title` line. The content should begin with `## First Section`.

### Step 3.3: Add SPDX Frontmatter

All NVIDIA files require SPDX copyright headers. Add as YAML frontmatter (NOT HTML comments -- those break MDX):

```yaml
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
```

Bulk-add script:

```bash
#!/usr/bin/env bash
# Add SPDX frontmatter to all fern/pages/*.md files that lack it.

HEADER='---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
'

find fern/pages -name '*.md' | while read -r f; do
    if ! head -1 "$f" | grep -q '^---'; then
        echo "${HEADER}$(cat "$f")" > "$f"
        echo "Added SPDX: $f"
    fi
done
```

### Step 3.4: Fix MDX-Breaking Patterns

These patterns are valid Markdown but break Fern's MDX parser. Fix ALL of them before running `fern docs dev`.

#### HTML Comments

MDX does NOT support `<!-- -->`. Convert to JSX comments or remove:

```python
#!/usr/bin/env python3
"""Remove HTML comments from Fern markdown files."""
import re
from pathlib import Path

def fix_file(path):
    text = path.read_text()
    original = text

    # Convert SPDX HTML comment blocks to YAML frontmatter
    spdx_pattern = r'<!--\s*(SPDX-FileCopyrightText:.*?SPDX-License-Identifier:.*?)-->'
    match = re.search(spdx_pattern, text, re.DOTALL)
    if match:
        spdx_content = match.group(1).strip()
        spdx_lines = '\n'.join(f'# {line.strip()}' for line in spdx_content.splitlines() if line.strip())
        text = text[:match.start()] + f'---\n{spdx_lines}\n---' + text[match.end():]

    # Convert remaining HTML comments to JSX
    text = re.sub(r'<!--(.*?)-->', r'{/* \1 */}', text, flags=re.DOTALL)

    if text != original:
        path.write_text(text)
        print(f"Fixed: {path}")

for f in Path('fern/pages').rglob('*.md'):
    fix_file(f)
```

#### Bare Angle Brackets

MDX treats ANY `<` as JSX. Escape bare `<` in prose:

| Pattern | Breaks MDX | Fix |
|---------|-----------|-----|
| `<1B parameters` | Yes | `\<1B parameters` or `less than 1B` |
| `<name>` in prose | Yes | `` `<name>` `` (backticks) |
| `<container_id>` in prose | Yes | `` `<container_id>` `` |
| `<Note>` Fern component | No | Leave as-is |
| `<` inside backticks | No | Already safe |
| `<` inside code blocks | No | Already safe |

Scan command:

```bash
grep -rn '<[a-zA-Z0-9]' fern/pages/ --include="*.md" | grep -v '```' | grep -v 'http' | grep -v '<Note>' | grep -v '<Warning>' | grep -v '<Tip>'
```

#### Blockquote Admonitions

Convert GitHub-style admonitions to Fern components:

```markdown
{/* BEFORE */}
> **Note:** This is important.

{/* AFTER */}
<Note>
This is important.
</Note>
```

Also convert `> **Warning:**` to `<Warning>` and `> **Tip:**` to `<Tip>`.

#### HTML Details/Summary

Convert to Fern Accordion component:

```markdown
{/* BEFORE */}
<details>
<summary>Click to expand</summary>

Hidden content here.

</details>

{/* AFTER */}
<Accordion title="Click to expand">
Hidden content here.
</Accordion>
```

---

## Phase 4: Migrate Images

### Step 4.1: Find ALL Images

Images may be scattered across subdirectories, not just a top-level `images/` folder:

```bash
find docs -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.svg" -o -name "*.gif" \) | sort
```

### Step 4.2: Copy with Hyphen Naming

```bash
#!/usr/bin/env bash
# Copy all images from docs/ to fern/assets/img/ with hyphen naming.

find docs -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.svg" -o -name "*.gif" \) | while read -r src; do
    filename=$(basename "$src" | tr '_' '-')
    cp "$src" "fern/assets/img/$filename"
    echo "Copied: $src -> fern/assets/img/$filename"
done
```

**Note:** This flattens all images into a single directory. If you have name collisions, prefix with the source folder name (e.g., `observability-dashboard.png`).

### Step 4.3: Update Image Paths

After copying images, update all references in `fern/pages/` files. The path from any page to the assets folder follows this pattern:

| Page location | Image path |
|---------------|------------|
| `fern/pages/guide.md` | `../assets/img/image.png` |
| `fern/pages/section/page.md` | `../../assets/img/image.png` |
| `fern/pages/section/sub/page.md` | `../../../assets/img/image.png` |

Count the directory depth from your page to `fern/` and add that many `../` prefixes before `assets/img/`.

```bash
# Find all image references to update
grep -rn '!\[' fern/pages/ --include="*.md"
```

---

## Phase 5: Build Navigation

### Step 5.1: Understand Navigation Structure

All navigation lives in `fern/versions/next.yml`. Key patterns:

```yaml
navigation:
  # Simple page
  - page: Installation
    path: ../pages/getting-started/installation.md

  # Section with child pages
  - section: Guides
    contents:
      - page: Quickstart
        path: ../pages/guides/quickstart.md
      - page: Configuration
        path: ../pages/guides/configuration.md

  # Section with clickable overview (the section itself is a page)
  - section: API Reference
    path: ../pages/api/README.md
    contents:
      - page: Endpoints
        path: ../pages/api/endpoints.md

  # Hidden page (accessible by URL, not in sidebar)
  - page: Draft Feature
    path: ../pages/drafts/feature.md
    hidden: true

  # External link
  - link: GitHub
    href: https://github.com/YOUR_ORG/YOUR_REPO
```

### Step 5.2: Auto-Generate Navigation Skeleton

Run this script to generate a starting `next.yml` from the `fern/pages/` directory tree:

```python
#!/usr/bin/env python3
"""Generate fern/versions/next.yml from fern/pages/ directory structure."""
from pathlib import Path
import yaml

def title_from_filename(name):
    """Convert filename to title: 'getting-started.md' -> 'Getting Started'"""
    stem = Path(name).stem
    if stem == 'README':
        return 'Overview'
    return stem.replace('-', ' ').title()

def build_nav(pages_dir):
    nav = []
    items = sorted(pages_dir.iterdir())

    # Process files first, then directories
    files = [f for f in items if f.is_file() and f.suffix == '.md' and f.name != 'index.md']
    dirs = [d for d in items if d.is_dir()]

    for f in files:
        rel = f.relative_to(pages_dir.parent)
        nav.append({
            'page': title_from_filename(f.name),
            'path': f'../{rel}'
        })

    for d in dirs:
        section = {'section': title_from_filename(d.name + '.md'), 'contents': []}
        readme = d / 'README.md'
        if readme.exists():
            rel = readme.relative_to(pages_dir.parent)
            section['path'] = f'../{rel}'

        sub_files = sorted(f for f in d.rglob('*.md') if f.name != 'README.md')
        for f in sub_files:
            rel = f.relative_to(pages_dir.parent)
            section['contents'].append({
                'page': title_from_filename(f.name),
                'path': f'../{rel}'
            })

        if section['contents'] or 'path' in section:
            nav.append(section)

    return nav

pages = Path('fern/pages')
nav = build_nav(pages)

header = """# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""

output = header + yaml.dump({'navigation': nav}, default_flow_style=False, sort_keys=False)
Path('fern/versions/next.yml').write_text(output)
print(output)
print("\nWrote fern/versions/next.yml")
print("Review and reorder sections manually before proceeding.")
```

### Step 5.3: Review and Refine

The auto-generated navigation is a starting point. Review and adjust:

- **Reorder sections** to match your preferred reading order
- **Rename page titles** to be user-friendly (not just filename-derived)
- **Group related pages** into sections
- **Add external links** (GitHub, API docs, etc.)
- **Hide draft pages** with `hidden: true`

---

## Phase 6: Fix Links

### Step 6.1: Keep `.md` Extensions

If your repo has a CI broken links checker, keep `.md` extensions on internal links. Fern handles them transparently:

```markdown
{/* Both work in Fern, but CI needs .md */}
[Guide](../guides/quickstart.md)
```

### Step 6.2: Detect Cross-Repo Links

Links to directories outside `fern/pages/` (e.g., `src/`, `examples/`, `scripts/`) will break. Convert to absolute GitHub URLs:

```markdown
{/* WRONG - resolves outside fern/pages/ */}
[Example](../../examples/quickstart/README.md)

{/* CORRECT - absolute GitHub URL */}
[Example](https://github.com/YOUR_ORG/YOUR_REPO/tree/main/examples/quickstart/README.md)
```

### Step 6.3: Link Auditor Script

```bash
#!/usr/bin/env bash
# Audit relative links in fern/pages/. Run from repo root.

errors=0
while IFS= read -r file; do
    dir=$(dirname "$file")
    grep -oP '\[.*?\]\(\K[^)]+' "$file" | while read -r link; do
        # Skip external URLs and anchors
        [[ "$link" =~ ^https?:// ]] && continue
        [[ "$link" =~ ^# ]] && continue
        # Strip anchor
        link_path="${link%%#*}"
        [[ -z "$link_path" ]] && continue
        # Resolve relative to file directory
        target="$dir/$link_path"
        if [[ ! -f "$target" ]]; then
            echo "BROKEN: $file -> $link_path"
            ((errors++))
        fi
    done
done < <(find fern/pages -name '*.md')
echo "Total broken links: $errors"
```

---

## Phase 7: Validate

### Step 7.1: Run All Checks

```bash
# 1. Fern config validation
fern check --warnings

# 2. Link audit
bash scripts/fern-link-audit.sh

# 3. Local preview
fern docs dev --port 3000
# Browse every page, check images, click links

# 4. Navigation verification (see utility scripts below)
bash scripts/fern-nav-verify.sh
```

### Step 7.2: Per-Page Checklist

For each migrated page, verify:

- [ ] Page renders without MDX parse errors
- [ ] All images display correctly
- [ ] Internal links navigate correctly
- [ ] Code blocks render with syntax highlighting
- [ ] No remaining HTML comments (`<!-- -->`)
- [ ] No bare `<` in prose outside backticks or code blocks
- [ ] Heading hierarchy starts at h2 (no duplicate h1)

### Step 7.3: Definition of Done

The migration is complete when ALL of the following are true:

- [ ] Every `docs/*.md` file has a corresponding `fern/pages/*.md` file
- [ ] `fern check --warnings` passes with zero errors
- [ ] `fern docs dev` renders all pages without MDX parse errors
- [ ] All images display correctly in local preview
- [ ] Navigation in `next.yml` covers all pages (no orphans)
- [ ] No HTML comments (`<!-- -->`) remain in any `fern/pages/` file
- [ ] No bare `<` in prose outside backticks or code blocks
- [ ] All cross-repo links are absolute GitHub URLs
- [ ] All internal links resolve correctly
- [ ] SPDX frontmatter present on all pages
- [ ] PR reviewed and approved

---

## Adding New Pages

After the initial migration, add new pages like this:

### Step 1: Create the File

```bash
mkdir -p fern/pages/guides/
touch fern/pages/guides/new-feature.md
```

### Step 2: Write Content

```markdown
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

## Overview

Description of the new feature...

## Getting Started

Step-by-step instructions...
```

Start with h2. Fern generates h1 from the navigation title.

### Step 3: Add to Navigation

Edit `fern/versions/next.yml`:

```yaml
- section: Guides
  contents:
    - page: New Feature          # <-- add this
      path: ../pages/guides/new-feature.md
    - page: Existing Guide
      path: ../pages/guides/existing.md
```

### Step 4: Validate

```bash
fern check --warnings
fern docs dev --port 3000
```

---

## Fern Config Reference

### `fern/fern.config.json`

| Field | Purpose | Example |
|-------|---------|---------|
| `organization` | Your Fern org name | `"my-project"` |
| `version` | Fern CLI version to use | `"3.29.1"` |

### `fern/docs.yml`

| Field | Purpose |
|-------|---------|
| `instances[].url` | Published site URL |
| `title` | Browser tab title |
| `versions` | List of version configs (each points to a navigation YAML) |
| `navbar-links` | Links in the top navigation bar |
| `colors.accent-primary` | Primary brand color (dark/light mode) |
| `colors.background` | Page background color (dark/light mode) |
| `logo` | Logo images and link (light/dark mode variants) |
| `favicon` | Browser tab icon |

### `fern/versions/next.yml`

| Pattern | Purpose |
|---------|---------|
| `- page: Title` + `path:` | Single page in sidebar |
| `- section: Title` + `contents:` | Collapsible section with children |
| `- section: Title` + `path:` + `contents:` | Section whose header is also a clickable page |
| `hidden: true` | Page accessible by URL but not shown in sidebar |
| `- link: Title` + `href:` | External link in sidebar |

---

## MDX Gotchas

These patterns are valid Markdown but break Fern's MDX parser. This table was built from real CI failures during the Dynamo migration (PR [#6050](https://github.com/ai-dynamo/dynamo/pull/6050)).

### Syntax Errors

| Error Message | Cause | Fix |
|---------------|-------|-----|
| `Unexpected character before name` | Bare `<` followed by letter/number in prose | Escape with `\<` or wrap in backticks |
| `Expected closing tag` | Bare `<name>` in table/prose | Wrap in backticks: `` `<name>` `` |
| `Unexpected token` | HTML comment `<!-- -->` | Convert to `{/* */}` or remove |
| `Adjacent JSX elements` | Multiple root elements | Wrap in `<>...</>` or single parent |
| `Unknown component` | Typo in component name | Check spelling: `<Note>` not `<note>` |
| `Unterminated JSX` | Missing closing tag | Ensure `<Note>...</Note>` is complete |

### Link Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| CI broken links failure | Missing `.md` extension | Keep `.md` on relative links |
| Link resolves outside `fern/pages/` | Cross-repo relative link | Convert to GitHub URL |
| `fern docs dev` shows blank page | MDX parse error in any page | Check terminal logs for the failing file |
| Stale error after fix | Dev server cache | `rm -rf ~/.fern/app-preview` and restart |

---

## Troubleshooting FAQ

**Q: `fern check` fails on the empty scaffold.**
A: Verify `fern.config.json` is valid JSON, `docs.yml` is valid YAML, and the page file referenced in `next.yml` exists. Check that logo and favicon files exist at the paths specified in `docs.yml`.

**Q: `fern docs dev` shows a blank page or crashes.**
A: Check the terminal output for a file path and error message. The most common cause is an MDX parse error (bare `<`, HTML comment, or unclosed JSX tag). Fix the file and restart.

**Q: I fixed the error but `fern docs dev` still shows the old error.**
A: The Fern dev server caches aggressively. Clear the cache and restart:
```bash
rm -rf ~/.fern/app-preview
fern docs dev --port 3000
```

**Q: `fern check` passes but CI fails on broken links.**
A: Your CI link checker likely requires `.md` extensions on relative links. Fern handles both with and without, but CI resolves links as file paths. Add `.md` to all internal relative links.

**Q: Images are broken in the preview.**
A: Verify the relative path depth. From `fern/pages/section/page.md`, the path to an image is `../../assets/img/image.png` (two levels up to `fern/`, then into `assets/img/`). Count the directory depth.

**Q: I added a page but it does not appear in the sidebar.**
A: Every page must have an entry in `fern/versions/next.yml`. Adding the `.md` file alone is not enough.

**Q: How do I link to source code or examples in the repo?**
A: Use absolute GitHub URLs for anything outside `fern/pages/`:
```markdown
[Source](https://github.com/YOUR_ORG/YOUR_REPO/tree/main/src/module.py)
```

**Q: `fern init` vs manual setup -- which should I use?**
A: `fern init` generates a scaffold but uses Fern's default branding. For NVIDIA projects, manual setup (Phase 1 of this skill) is faster because you can paste the NVIDIA branding directly.

---

## Utility Scripts

### Navigation Verifier

Checks that every `next.yml` entry maps to an existing file and finds orphan pages:

```bash
#!/usr/bin/env bash
# fern-nav-verify.sh -- Run from repo root.

echo "=== Nav entries pointing to missing files ==="
grep -oP 'path:\s*\K\S+' fern/versions/next.yml | while read -r p; do
    target="fern/versions/$p"
    [[ ! -f "$target" ]] && echo "MISSING: $p"
done

echo ""
echo "=== Pages not in navigation ==="
nav_files=$(grep -oP 'path:\s*\.\./pages/\K\S+' fern/versions/next.yml | sort)
actual_files=$(find fern/pages -name '*.md' -printf '%P\n' | sort)
comm -13 <(echo "$nav_files") <(echo "$actual_files")
```

### Cross-Repo Link Detector

Finds relative links that escape `fern/pages/`:

```bash
#!/usr/bin/env bash
# fern-cross-repo-links.sh -- Run from repo root.

echo "Links that should become GitHub URLs:"
while IFS= read -r file; do
    dir=$(dirname "$file")
    grep -oP '\[.*?\]\(\K[^)]+' "$file" | while read -r link; do
        [[ "$link" =~ ^https?:// ]] && continue
        [[ "$link" =~ ^# ]] && continue
        link_path="${link%%#*}"
        [[ -z "$link_path" ]] && continue
        resolved=$(cd "$dir" && realpath -m "$link_path" 2>/dev/null)
        if [[ -n "$resolved" ]] && [[ ! "$resolved" =~ fern/pages ]]; then
            echo "  $file: $link_path"
        fi
    done
done < <(find fern/pages -name '*.md')
```

---

## CI Integration

### Set Up Publishing Workflow

Create `.github/workflows/publish-fern-docs.yml`:

```yaml
name: Publish Fern Docs

on:
  push:
    branches: [main]
    paths: ['fern/**']
  workflow_dispatch:

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install Fern
        run: npm install -g fern-api

      - name: Publish docs
        env:
          FERN_TOKEN: ${{ secrets.FERN_TOKEN }}
        run: fern generate --docs
```

### Required Setup

1. **Get a Fern token:** Sign up at [buildwithfern.com](https://buildwithfern.com), create your organization, and generate an API token.
2. **Add repository secret:** Go to repo Settings > Secrets > Actions > New secret. Name: `FERN_TOKEN`, Value: your token.
3. **First publish:** Push a commit touching `fern/` to main, or manually trigger the workflow.

### Optional: Broken Links CI Check

Add a link checker to PRs:

```yaml
name: Check Docs Links

on:
  pull_request:
    paths: ['fern/**', 'docs/**']

jobs:
  check-links:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check links
        uses: lycheeverse/lychee-action@v2
        with:
          args: --offline --no-progress fern/pages/
          fail: true
```

---

## Related Skills

| Skill | When to Use |
|-------|-------------|
| `fern-migration` | Migrating from Sphinx (RST + MD) to Fern with an existing Fern setup |
| `check-links` | Pre/post-migration link validation |
| `write-docs` | Writing new documentation content |
| `lint-docs` | Checking markdown quality after migration |
| `new-pr` | Creating the migration PR |
| `fix-pr` | Addressing review feedback on migration PRs |

---

## Handoff Notes

If you are picking up this skill for the first time:

1. **Study a reference migration.** Review the [Dynamo PR #6050](https://github.com/ai-dynamo/dynamo/pull/6050) to see the scope and patterns of a full Fern migration (126 files). The source was Sphinx, but the target structure and gotchas are identical.

2. **Budget for the fix cycle.** The initial migration (copy, convert, build nav) takes ~30% of the effort. The remaining ~70% is fixing MDX parse errors and broken links. Plan accordingly.

3. **Start small.** Scaffold Fern (Phase 1-2), migrate 3-5 files (Phase 3), build a minimal nav (Phase 5), and validate (Phase 7). Only scale up after confirming the workflow works end-to-end.

4. **Run the HTML comment remover and angle bracket escaper FIRST.** After bulk-copying files, immediately run these scripts before anything else. This prevents cascading MDX parse errors that make `fern docs dev` unusable.

5. **Use a worktree.** Isolate migration work from your main workspace:
   ```bash
   git worktree add ../worktrees/fern-setup -b yourname/fern-setup origin/main
   ```

6. **Get logo assets early.** The NVIDIA logo SVGs and favicon are required before `fern docs dev` will render correctly. Copy from an existing NVIDIA Fern project or request from design