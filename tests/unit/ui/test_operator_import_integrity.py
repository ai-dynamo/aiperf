# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static import integrity checks for the operator UI modules."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_INDEX_HTML = _UI_ROOT / "index.html"

_IMPORT_FROM_RE = re.compile(
    r"\b(?:import|export)\s+(?:[\s\S]*?)\s+from\s*(['\"])(?P<specifier>[^'\"]+)\1",
    re.MULTILINE,
)
_SIDE_EFFECT_IMPORT_RE = re.compile(r"\bimport\s*(['\"])(?P<specifier>[^'\"]+)\1")
_DYNAMIC_IMPORT_RE = re.compile(r"\bimport\s*\(\s*(['\"])(?P<specifier>[^'\"]+)\1\s*\)")
_IMPORTMAP_RE = re.compile(
    r"<script\s+type=[\"']importmap[\"']\s*>\s*(?P<json>.*?)\s*</script>",
    re.DOTALL,
)
_MODULE_SCRIPT_RE = re.compile(
    r"<script\b[^>]*\btype=[\"']module[\"'][^>]*\bsrc=[\"'](?P<src>[^\"']+)[\"']"
)
_STYLESHEET_RE = re.compile(
    r"<link\b[^>]*\brel=[\"']stylesheet[\"'][^>]*\bhref=[\"'](?P<href>[^\"']+)[\"']"
)


@dataclass(frozen=True)
class ImportSpec:
    module_path: Path
    specifier: str


def _js_modules() -> list[Path]:
    return sorted(
        path for suffix in ("*.js", "*.mjs") for path in _UI_ROOT.rglob(suffix)
    )


def _strip_js_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*", "", source)


def _imports_from(path: Path) -> list[ImportSpec]:
    source = _strip_js_comments(path.read_text())
    specs: list[ImportSpec] = []
    for regex in (_IMPORT_FROM_RE, _SIDE_EFFECT_IMPORT_RE, _DYNAMIC_IMPORT_RE):
        specs.extend(
            ImportSpec(path, match.group("specifier"))
            for match in regex.finditer(source)
        )
    return specs


def _all_imports() -> list[ImportSpec]:
    return [
        import_spec for path in _js_modules() for import_spec in _imports_from(path)
    ]


def _is_relative(specifier: str) -> bool:
    return specifier.startswith(("./", "../"))


def _is_bare(specifier: str) -> bool:
    return not specifier.startswith(("./", "../", "/", "http://", "https://"))


def _resolve_relative(module_path: Path, specifier: str) -> Path:
    return (module_path.parent / specifier).resolve()


def _index_html() -> str:
    return _INDEX_HTML.read_text()


def _import_map() -> dict[str, str]:
    match = _IMPORTMAP_RE.search(_index_html())
    assert match is not None
    parsed = json.loads(match.group("json"))
    imports = parsed.get("imports")
    assert isinstance(imports, dict)
    return imports


def test_all_relative_js_imports_resolve_to_existing_modules() -> None:
    missing = {
        f"{import_spec.module_path.relative_to(_UI_ROOT)} -> {import_spec.specifier}": str(
            _resolve_relative(
                import_spec.module_path, import_spec.specifier
            ).relative_to(_UI_ROOT)
        )
        for import_spec in _all_imports()
        if _is_relative(import_spec.specifier)
        and not _resolve_relative(
            import_spec.module_path, import_spec.specifier
        ).is_file()
    }

    assert missing == {}


def test_relative_js_imports_use_browser_module_paths_not_python_or_css() -> None:
    wrong_suffixes = {
        f"{import_spec.module_path.relative_to(_UI_ROOT)} -> {import_spec.specifier}"
        for import_spec in _all_imports()
        if _is_relative(import_spec.specifier)
        and Path(import_spec.specifier).suffix in {".py", ".css"}
    }
    missing_module_suffixes = {
        f"{import_spec.module_path.relative_to(_UI_ROOT)} -> {import_spec.specifier}"
        for import_spec in _all_imports()
        if _is_relative(import_spec.specifier)
        and Path(import_spec.specifier).suffix not in {".js", ".mjs"}
    }

    assert wrong_suffixes == set()
    assert missing_module_suffixes == set()


def test_index_html_import_map_covers_all_bare_module_imports() -> None:
    import_map = _import_map()
    bare_imports = sorted(
        {
            import_spec.specifier
            for import_spec in _all_imports()
            if _is_bare(import_spec.specifier)
        }
    )
    missing = [specifier for specifier in bare_imports if specifier not in import_map]

    assert missing == []


def test_index_html_local_entrypoints_exist() -> None:
    local_entrypoints = [
        *(
            _INDEX_HTML.parent / match.group("src")
            for match in _MODULE_SCRIPT_RE.finditer(_index_html())
        ),
        *(
            _INDEX_HTML.parent / match.group("href")
            for match in _STYLESHEET_RE.finditer(_index_html())
        ),
    ]
    missing = sorted(
        str(path.relative_to(_UI_ROOT))
        for path in local_entrypoints
        if not path.is_file()
    )

    assert missing == []
