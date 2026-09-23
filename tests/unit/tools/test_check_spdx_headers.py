# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the repository-wide SPDX header validator."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[3]


def load_checker() -> ModuleType:
    """Load the checker script as a module without packaging it."""
    path = ROOT / "tools/check_spdx_headers.py"
    spec = importlib.util.spec_from_file_location("check_spdx_headers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SPDX checker from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = load_checker()
HASH_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)


@pytest.mark.parametrize(
    ("filename", "header"),
    [
        pytest.param("example.py", HASH_HEADER, id="hash-comment"),
        pytest.param(
            "example.mjs",
            HASH_HEADER.replace("# ", "// "),
            id="javascript-comment",
        ),
        pytest.param(
            "example.tsx",
            HASH_HEADER.replace("# ", "// "),
            id="typescript-comment",
        ),
        pytest.param("example.yaml.tmpl", HASH_HEADER, id="yaml-template-comment"),
        pytest.param(
            "example.mmd",
            HASH_HEADER.replace("# ", "%% "),
            id="mermaid-comment",
        ),
        pytest.param(
            "example.css",
            "/* "
            + HASH_HEADER.replace("# ", "", 1).replace("\n# ", "\n   ").rstrip()
            + " */\n",
            id="css-comment",
        ),
        pytest.param(
            "example.html",
            "<!DOCTYPE html>\n<!--\n" + HASH_HEADER.replace("# ", "") + "-->\n",
            id="html-comment",
        ),
        pytest.param(
            "example.md",
            "---\n" + HASH_HEADER + "title: Example\n---\n",
            id="markdown-frontmatter",
        ),
    ],
)
def test_supported_header_styles_are_accepted(
    tmp_path: Path, filename: str, header: str
) -> None:
    """AIPerf source formats accept their established comment syntax."""
    (tmp_path / filename).write_text(header, encoding="utf-8")
    assert CHECKER.validate_file(tmp_path, Path(filename)) == []


def test_shebang_may_precede_header(tmp_path: Path) -> None:
    """An executable script may place its shebang before the SPDX lines."""
    (tmp_path / "tool.py").write_text(
        "#!/usr/bin/env python3\n" + HASH_HEADER,
        encoding="utf-8",
    )
    assert CHECKER.validate_file(tmp_path, Path("tool.py")) == []


def test_third_party_copyright_is_accepted(tmp_path: Path) -> None:
    """Apache-licensed contributions need not claim NVIDIA ownership."""
    (tmp_path / "contribution.py").write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 Example Contributor\n"
        "# SPDX-License-Identifier: Apache-2.0\n",
        encoding="utf-8",
    )
    assert CHECKER.validate_file(tmp_path, Path("contribution.py")) == []


@pytest.mark.parametrize(
    "copyright_line",
    [
        pytest.param(
            "# SPDX-FileCopyrightText: Copyright (c) 2026 Nvidia Corporation "
            "& AFFILIATES. All rights reserved.\n",
            id="mixed-case-company",
        ),
        pytest.param(
            "# SPDX-FileCopyrightText: Copyright (c) 2026-2025 NVIDIA "
            "CORPORATION & AFFILIATES. All rights reserved.\n",
            id="reversed-year-range",
        ),
    ],
)
def test_malformed_nvidia_copyright_is_rejected(
    tmp_path: Path, copyright_line: str
) -> None:
    """NVIDIA ownership text cannot bypass the canonical header format."""
    (tmp_path / "example.py").write_text(
        copyright_line + "# SPDX-License-Identifier: Apache-2.0\n",
        encoding="utf-8",
    )
    violations = CHECKER.validate_file(tmp_path, Path("example.py"))
    assert "malformed NVIDIA copyright header" in violations[0]


def test_non_apache_identifier_is_rejected(tmp_path: Path) -> None:
    """A complete copyright cannot mask a non-Apache license identifier."""
    (tmp_path / "style.css").write_text(
        HASH_HEADER.replace("# ", "/* ", 1).replace(
            "# SPDX-License-Identifier: Apache-2.0",
            " * SPDX-License-Identifier: MIT\n */",
        ),
        encoding="utf-8",
    )
    violations = CHECKER.validate_file(tmp_path, Path("style.css"))
    assert "malformed or missing Apache-2.0 header" in violations[0]


@pytest.mark.parametrize(
    "filename",
    [
        *(f"example{suffix}" for suffix in sorted(CHECKER.SOURCE_SUFFIXES)),
        *sorted(CHECKER.SOURCE_FILENAMES),
        *(f"{prefix}example" for prefix in CHECKER.SOURCE_FILENAME_PREFIXES),
    ],
)
def test_bare_spdx_lines_are_rejected_for_every_source_type(
    tmp_path: Path, filename: str
) -> None:
    """SPDX tags must use the source type's supported comment syntax."""
    (tmp_path / filename).write_text(
        HASH_HEADER.replace("# ", "") + "key: value\n",
        encoding="utf-8",
    )

    violations = CHECKER.validate_file(tmp_path, Path(filename))

    assert "invalid comment syntax" in violations[0]


def test_exempt_artifacts_are_accepted(tmp_path: Path) -> None:
    """Third-party, non-commentable, and symlink artifacts are exempt."""
    vendor = tmp_path / "src/aiperf/api/static/vendor/prism-core.js"
    vendor.parent.mkdir(parents=True)
    vendor.write_text("vendor code\n", encoding="utf-8")
    (tmp_path / "data.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "target.py").write_text(HASH_HEADER, encoding="utf-8")
    (tmp_path / "link.py").symlink_to("target.py")

    paths = [
        Path("src/aiperf/api/static/vendor/prism-core.js"),
        Path("data.json"),
        Path("link.py"),
    ]
    assert CHECKER.validate_paths(tmp_path, paths) == []


def test_empty_source_file_is_rejected(tmp_path: Path) -> None:
    """An empty source file cannot silently bypass the header policy."""
    (tmp_path / "empty.py").touch()
    violations = CHECKER.validate_file(tmp_path, Path("empty.py"))
    assert "empty source file is missing an SPDX header" in violations[0]


def test_binary_and_unknown_files_are_rejected(tmp_path: Path) -> None:
    """Unclassified artifacts fail closed instead of escaping the scan."""
    (tmp_path / "binary.py").write_bytes(b"\x00data")
    (tmp_path / "unknown.newtype").write_text("data", encoding="utf-8")
    assert "binary files" in CHECKER.validate_file(tmp_path, Path("binary.py"))[0]
    assert (
        "unsupported tracked file type"
        in CHECKER.validate_file(tmp_path, Path("unknown.newtype"))[0]
    )


def test_all_paths_are_reported(tmp_path: Path) -> None:
    """Validation reports every offending path instead of stopping early."""
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("value: b\n", encoding="utf-8")
    violations = CHECKER.validate_paths(
        tmp_path,
        [Path("b.yaml"), Path("a.py")],
    )
    assert len(violations) == 2
    assert violations[0].startswith("a.py:")
    assert violations[1].startswith("b.yaml:")


def test_git_index_supplies_the_scan_set(tmp_path: Path) -> None:
    """Only paths recorded by Git are returned for repository validation."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.py").write_text(HASH_HEADER, encoding="utf-8")
    (tmp_path / "untracked.py").write_text(HASH_HEADER, encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    assert CHECKER.tracked_files(tmp_path) == [Path("tracked.py")]


def test_failure_output_points_to_fixer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed repository scan gives contributors a copyable repair command."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "missing.py").write_text("print('missing')\n", encoding="utf-8")
    subprocess.run(["git", "add", "missing.py"], cwd=tmp_path, check=True)

    assert CHECKER.main(["--root", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "make add-copyright args=missing.py" in captured.err


def test_cmake_lists_is_not_exempt_as_plain_text(tmp_path: Path) -> None:
    """The exact CMake source filename takes precedence over the text exemption."""
    path = Path("CMakeLists.txt")
    (tmp_path / path).write_text("project(aiperf)\n", encoding="utf-8")
    violations = CHECKER.validate_file(tmp_path, path)
    assert "malformed or missing copyright header" in violations[0]


def test_dockerfile_variant_is_not_exempt_by_suffix(tmp_path: Path) -> None:
    """Dockerfile variants are source even when their suffix looks unfamiliar."""
    path = Path("Dockerfile.mock-server")
    (tmp_path / path).write_text("FROM scratch\n", encoding="utf-8")
    violations = CHECKER.validate_file(tmp_path, path)
    assert "malformed or missing copyright header" in violations[0]
