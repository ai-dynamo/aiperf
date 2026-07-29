<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wheel packaging

## Purpose

One installable artifact carries two things: the `src/aiperf` Python tree and the
native `aiperf` executable that is the product entry point. This record states how
that artifact is built today, why its filename claims a CPython ABI it does not
use, and the contract for removing that claim.

## Built

### The artifact

`make wheel` produces one wheel per (Python minor × platform). On x86-64 Linux
under CPython 3.12 that is
`aiperf-0.11.0-cp312-cp312-manylinux_2_34_x86_64.whl`, 30,949,757 bytes over 1,091
entries. Compressed composition:

| Entry | Uncompressed | Compressed | Share |
|---|---|---|---|
| `aiperf-0.11.0.data/scripts/aiperf` | 59,409,320 | 24,781,581 | 80.5% |
| `aiperf/dataset/generator/assets/shakespeare.txt` | 5,020,475 | 1,924,295 | 6.3% |
| `aiperf/_native.cpython-312-x86_64-linux-gnu.so` | 377,072 | 180,770 | 0.6% |
| 4 × `aiperf/dataset/generator/assets/source_images/*.jpg` | 574,212 | 543,836 | 1.8% |
| `aiperf/config/schema/aiperf-config.schema.json` | 996,885 | 91,366 | 0.3% |
| remaining 1,081 entries | — | ~3,262,825 | 10.6% |

The native binary dominates. It is built by the `optimized` profile
(`rust/Cargo.toml` `[profile.optimized]`: `inherits = "release"`, `lto = "fat"`,
`codegen-units = 1`, `strip = "symbols"`) with `CLI_FEATURES ?= --features full`.

### The three-step build

1. `bundle-cli` (`Makefile:210-211`) runs
   `cargo build --profile optimized -p aiperf-cli $(CLI_FEATURES)`, writing
   `rust/target/optimized/aiperf`.
2. `wheel` (`Makefile:213-215`) runs `maturin build --release --out dist`. maturin
   compiles the `rust/pyext` cdylib into
   `aiperf/_native.cpython-3XX-<plat>.so`, packages `src/aiperf`
   (`[tool.maturin] python-source = "src"`), and runs its built-in auditwheel
   repair to derive the `manylinux_2_34` tag.
3. `wheel_repack.py` (`Makefile:218`) injects the step-1 binary at
   `<dist>-<ver>.data/scripts/aiperf` with mode 0755 and regenerates
   `dist-info/RECORD`. PEP 427 makes pip install `*.data/scripts/` entries
   straight into the environment's `bin/`, so the `aiperf` command *is* the ELF
   binary — no Python launcher shim. `[project.scripts]` therefore declares only
   `aiperf-python = "aiperf.entrypoint:main"` (`pyproject.toml:88-95`).

`Dockerfile:131-140` performs the same three steps inline, selecting
`CLI_FEATURES` from `AIPERF_RUNNER_PROFILE` (`offline` → `--features full`,
`online` → `--features parquet`).

### Why a pyo3 module exists

The wheel needs a platform tag (it carries an ELF) but must not use maturin's
`bindings = "bin"`, which is illegal alongside a pyo3 module plus
`[project.scripts]` (`pyproject.toml:103-105`, `tools/wheel_repack.py:5-8`). The
resolution was to give maturin a pyo3 module to compile so it emits a
platform-tagged wheel, then bolt the real binary on in step 3.

That module is `rust/pyext`, one file, `rust/pyext/src/lib.rs`, 48 lines. Its
entire surface is four `#[pyfunction]`s returning compile-time constants:

| Function | Returns |
|---|---|
| `runner_filename()` | `"aiperf"` |
| `runner_relpath()` | `"_bin/aiperf"` |
| `build_profile()` | `"debug"` / `"release"` from `cfg!(debug_assertions)` |
| `pyext_version()` | `env!("CARGO_PKG_VERSION")` |

**No caller exists.** A repository-wide search of `src/`, `tools/`, and `tests/`
for `aiperf._native`, `runner_filename`, `runner_relpath`, `pyext_version`, and
`build_profile` returns zero import sites and zero call sites. The module's
docstring (`rust/pyext/src/lib.rs:6-8`) states it "does not discover or launch
that executable". It is inert.

### Consequences of the inert module

`rust/pyext/Cargo.toml:30` declares
`pyo3 = { version = "0.23", features = ["extension-module"] }` with **no `abi3`
feature**, so the emitted `.so` is bound to one CPython minor version. With
`requires-python = ">=3.11,<3.14"` (`pyproject.toml:25`), full coverage of
macOS arm64, Linux x64, Linux arm64, Windows x64, and Windows arm64 requires
**3 interpreters × 5 platforms = 15 wheels** — and nightly's
`tools/rename_wheel.py` step doubles each into an `aiperf-nightly` variant, so 30
files. Every one of those 15 differs only in a 377 KB module returning four
strings.

Today's nightly builds 2 of the 15: `.github/workflows/nightly.yml:405-414`
matrixes `amd64`/`arm64` Linux at a single Python version, yielding 4 files after
the rename. macOS and Windows are not in the matrix.

A second consequence is build time. Step 2 compiles the pyo3 cdylib under
maturin's own `profile = "release"` (`pyproject.toml:114`), a different profile
from step 1's `optimized`, so a full `make wheel` compiles the workspace twice.

### Known documentation drift

`pyproject.toml:6-8` states the binary is interned "as package data
(`aiperf/_bin/aiperf`)". `pyproject.toml:103-108` states it is "NOT interned as
package data" and lands in `.data/scripts/`. The latter matches
`tools/wheel_repack.py:68` (`f"{distribution}-{version}.data/scripts/{_SCRIPT_NAME}"`),
so `:6-8` and `runner_relpath()`'s `"_bin/aiperf"` are both stale. `nightly.yml:494-497`
repeats the "interned as package data" phrasing.

## Future requirements

Delete `rust/pyext` and drop the CPython-ABI dimension from the wheel matrix. The
wheel becomes `py3-none-<platform>`: platform-tagged because it carries an ELF,
Python-agnostic because nothing in it links a CPython ABI. Coverage of the five
platforms above falls from 15 wheels to **5**.

`abi3-py311` on the existing dep is the cheaper alternative (15 → 5 with a
one-line change) but is a floor, not a fix: it pins the minimum interpreter and
leaves a compiled artifact, and its reasoning must be revisited at every Python
version boundary. Removing the module removes the question.

### Required contract

1. **Build backend.** Replace maturin (`pyproject.toml:11-13`) with a
   pure-Python backend and delete `[tool.maturin]` (`:109-115`). The replacement
   MUST package the `src/aiperf` tree, preserving what `python-source = "src"`
   provides today, including the non-Python package data the table above
   enumerates (`dataset/generator/assets/`, `config/schema/`, `plugin/*.yaml`).
2. **Platform tag.** `wheel_repack.py` MUST rewrite `dist-info/WHEEL`'s `Tag:`
   from `py3-none-any` to `py3-none-<platform>` and rename the output file to
   match. It already regenerates `RECORD` from surviving lines
   (`tools/wheel_repack.py:88-98`); `WHEEL` becomes a second rewritten entry
   subject to the same RECORD hash update.
3. **Tag derivation.** The platform tag MUST be derived from the injected binary,
   not hardcoded. maturin's bundled auditwheel repair computes the manylinux glibc
   floor today; dropping maturin drops that check, and a hardcoded
   `manylinux_2_34` would silently lie the first time the binary is built against
   a newer glibc. `auditwheel` is not currently a project dependency and MUST be
   added to the packaging path (or an equivalent ELF-versioned-symbol scan
   implemented) to keep this honest. **This is the one part of the change that
   trades away an existing correctness guarantee, and it is the part that must not
   be shortcut.**
4. **Interface preservation.** `aiperf` MUST remain installed from
   `.data/scripts/aiperf` with mode 0755 and no launcher shim;
   `aiperf-python` MUST remain the `[project.scripts]` entry. Wheel-install
   validation in `nightly.yml:543-560` MUST pass unchanged.
5. **Idempotence.** Re-running the repack on an already-repacked wheel MUST
   remain safe (`tools/wheel_repack.py:16-18`).

### Coupled edits

- `rust/Cargo.toml:10-22` — remove `"pyext"` from `[workspace].members` and its
  comment at `:15-16`.
- `docs/specs/repository-layout.md` — delete the `rust/pyext` row from the
  directory table, the `aiperf-pyext → pyo3` dependency-direction clause, the
  `aiperf-pyext` crate-responsibility bullet, and naming rule 4's `[lib].name =
  "_native"` exception, which exists only for this crate. Rule 4 then reads with
  no exception.
- `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`,
  `.cursor/rules/python.mdc` — all four carry the identical `pyext` bullet at
  line 64 and the `pyext` entry in the workspace-crates list. Bodies are
  byte-identical from `# AIPerf`; edit together and run
  `tools/check_agent_files_sync.py`.
- `llms.txt` — crate topology and the `docs/specs/` index entry.
- `pyproject.toml:6-8` — rewrite the stale `aiperf/_bin/aiperf` claim rather than
  carrying it forward.
- `Dockerfile:123-129` — the comment block describes maturin compiling the cdylib
  and running auditwheel repair.
- `.github/workflows/nightly.yml:494-502` — the per-arch wheel comment and the
  stale "interned as package data" phrasing.
- `tools/check_crate_layout.py` — verify it passes with the member removed; it
  encodes no `pyext` special case today, so no edit is expected.

### Verification

- `make wheel` produces one `py3-none-manylinux_*_x86_64` wheel; `unzip -l`
  shows no `_native*.so` and `.data/scripts/aiperf` present at 0755.
- Fresh-venv `pip install` of that wheel: `aiperf --version` runs the ELF
  (`file $(command -v aiperf)` reports an ELF, not a script), `aiperf-python
  --help` works, `python -c "import aiperf"` succeeds.
- The same wheel installs under 3.11, 3.12, and 3.13 — the property the change
  buys.
- `tools/check_agent_files_sync.py` and `tools/check_docs_current.py` exit zero.

## Source anchors

- `pyproject.toml` (`[build-system]`, `[tool.maturin]`, `[project.scripts]`,
  `requires-python`).
- `Makefile` (`bundle-cli`, `wheel`, `CLI_FEATURES`).
- `tools/wheel_repack.py`; `tools/rename_wheel.py`.
- `rust/pyext/Cargo.toml`; `rust/pyext/src/lib.rs`.
- `rust/Cargo.toml` (`[workspace].members`, `[profile.optimized]`).
- `Dockerfile` (wheel-builder stage).
- `.github/workflows/nightly.yml` (arch matrix, wheel build/extract, rename,
  validate).
