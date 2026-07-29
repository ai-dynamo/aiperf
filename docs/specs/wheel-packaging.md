<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wheel packaging

## Purpose

One installable artifact carries two things: the `src/aiperf` Python tree and the
native `aiperf` executable that is the product entry point. This record states how
that artifact is built and why its tag is `py3-none-<platform>`: platform-specific
because it carries an ELF, interpreter-agnostic because nothing in it links a
CPython ABI.

## Built

### The artifact

`make wheel` produces one wheel per platform. On x86-64 Linux that is
`aiperf-0.11.0-py3-none-manylinux_2_39_x86_64.whl`, 30,765,063 bytes over 1,089
entries. Compressed composition:

| Entry | Uncompressed | Compressed | Share |
|---|---|---|---|
| `aiperf-0.11.0.data/scripts/aiperf` | 59,409,320 | 24,781,581 | 81.0% |
| `aiperf/dataset/generator/assets/shakespeare.txt` | 5,020,475 | 1,924,295 | 6.3% |
| 4 × `aiperf/dataset/generator/assets/source_images/*.jpg` | 574,212 | 543,836 | 1.8% |
| `aiperf/config/schema/aiperf-config.schema.json` | 996,885 | 91,366 | 0.3% |
| remaining 1,082 entries (all rows above are 7 entries) | — | ~3,259,259 | 10.7% |

Separately, 1,082 of the 1,089 entries are the `aiperf/` package payload; the other 7 are
metadata the build emits, not package data — 6 `dist-info/` entries (`METADATA`,
`WHEEL`, `RECORD`, `entry_points.txt`, `licenses/LICENSE`,
`licenses/ATTRIBUTIONS.md`) plus the injected `.data/scripts/aiperf`.

Of the 1,082 package entries, 961 are `.py` and **121** are package data that
hatchling's `packages = ["src/aiperf"]` sweeps in implicitly: 67 `.js`, 31
`.yaml`, 6 `.html`, 6 `.json`, 4 `.jpg`, 3 `.md`, 2 `.jsonl`, 1 `.txt`, and 1
`.css`, spread over 17 leaf directories under six top-level packages —
`operator/` (68), `config/` (33), `dataset/` (12), `plugin/` (4), `analysis/`
(3), and `plot/` (1). hatchling includes every file under the declared package
directory, not an enumerated `package-data` pattern list, which is what keeps
those 121 files in.

On disk `src/aiperf` holds 122 non-`.py` files (excluding `__pycache__`). The
one file that does not reach the wheel is `src/aiperf/plugin/enums.pyi`, a
generated stub ignored at `.gitignore:40`; hatchling honors VCS ignore rules, so
122 on disk yields 121 in the payload.

The native binary dominates. It is built by the `optimized` profile
(`rust/Cargo.toml` `[profile.optimized]`: `inherits = "release"`, `lto = "fat"`,
`codegen-units = 1`, `strip = "symbols"`) with `CLI_FEATURES ?= --features full`.

### The three-step build

1. `bundle-cli` (`Makefile:217-218`) runs
   `cargo build --profile optimized -p aiperf-cli $(CLI_FEATURES)`, writing
   `rust/target/optimized/aiperf`.
2. `wheel` (`Makefile:220-222`) runs `python -m build --wheel --outdir dist`
   against the `hatchling.build` backend (`pyproject.toml:17-19`), packaging
   `src/aiperf` into a `py3-none-any` wheel with no compiled extension.
3. `wheel_repack.py` (`Makefile:226`) injects the step-1 binary at
   `<dist>-<ver>.data/scripts/aiperf` with mode 0755, rewrites
   `dist-info/WHEEL`, renames the file to the platform tag, and regenerates
   `dist-info/RECORD`. PEP 427 makes pip install `*.data/scripts/` entries
   straight into the environment's `bin/`, so the `aiperf` command *is* the ELF
   binary — no Python launcher shim. `[project.scripts]` therefore declares only
   `aiperf-python = "aiperf.entrypoint:main"` (`pyproject.toml:94-101`).

`Dockerfile:123-136` performs the same three steps inline, selecting
`CLI_FEATURES` from `AIPERF_RUNNER_PROFILE` (`offline` → `--features full`,
`online` → `--features parquet`).

### The platform tag

`wheel_repack.py` derives the tag from the injected binary itself, not from a
hardcoded value and not from a separate compiled artifact. `glibc_versions()`
(`tools/wheel_repack.py:53`) reads the ELF's `.gnu.version_r`
(`SHT_GNU_verneed`) table directly, `manylinux_tag()` (`:132`) takes the highest
`GLIBC_x.y` need as the floor, and `platform_tag_for()` (`:138`) composes
`py3-none-manylinux_<major>_<minor>_<machine>`. `rewrite_wheel_tag()` (`:146`)
then collapses `dist-info/WHEEL` to that single `Tag:` line and pins
`Root-Is-Purelib: false` — a pure-Python backend defaults it to `true`, which
would move the tree from platlib to purelib — and the output file is renamed to
match. `RECORD` is regenerated so both the injected script and the rewritten
`WHEEL` carry correct hashes.

The single wheel installs across the whole `requires-python = ">=3.11,<3.14"`
range (`pyproject.toml:31`).

### Known documentation drift

- **`tools/rename_wheel.py` drops the executable bit.** It repacks with
  `zf.writestr(rel, data)` (`:274`), which does not carry `external_attr`, so
  `.data/scripts/aiperf` goes from `0o100755` in the repacked wheel to `0o600` in
  the `aiperf-nightly` variant.

## Future requirements

1. **Preserve the executable bit through the nightly rename.**
   `tools/rename_wheel.py:274` MUST write `.data/scripts/aiperf` with its
   `external_attr` intact so the `aiperf-nightly` variant installs an executable
   `aiperf`, or the record MUST state explicitly that the variant is
   non-executable.

## Source anchors

- `pyproject.toml` (`[build-system]`, `[tool.hatch.build.targets.*]`,
  `[project.scripts]`, `requires-python`).
- `Makefile` (`bundle-cli`, `wheel`, `CLI_FEATURES`).
- `tools/wheel_repack.py`; `tools/rename_wheel.py`.
- `rust/Cargo.toml` (`[workspace].members`, `[profile.optimized]`).
- `Dockerfile` (wheel-builder stage).
- `.github/workflows/nightly.yml` (arch matrix, wheel build/extract, rename,
  validate). Because the glibc minor in the platform tag comes from the built
  binary, the workflow computes no wheel filename up front: `Validate wheels`
  globs `dist/` and sweeps CPython 3.11, 3.12, and 3.13, `Upload wheels to S3`
  globs `dist/`, and `Stage Wheel (S3 → Artifactory)` lists the S3 version
  prefix and forwards the discovered names to the GitLab trigger.
