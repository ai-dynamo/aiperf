#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "aiperf"
API_MD_PATH = PROJECT_ROOT / "docs" / "api.md"

MODULES = []


def module_path_from_file(py_path: Path) -> str:
    rel_path = py_path.relative_to(SRC_ROOT.parent)
    parts = rel_path.with_suffix("").parts
    return ".".join(parts)


def find_all_python_modules():
    for dirpath, _, filenames in os.walk(SRC_ROOT):
        for filename in filenames:
            if filename.endswith(".py") and filename != "__init__.py":
                py_path = Path(dirpath) / filename
                mod_path = module_path_from_file(py_path)
                MODULES.append(mod_path)


def write_api_md():
    with open(API_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# API Reference\n\n")
        f.write(
            "This page contains the API documentation for all Python modules in the codebase (excluding __init__.py files).\n\n"
        )
        for mod in sorted(MODULES):
            f.write(f"## {mod}\n\n::: {mod}\n\n")


def main():
    find_all_python_modules()
    write_api_md()
    print(f"Generated {API_MD_PATH} with {len(MODULES)} modules.")


if __name__ == "__main__":
    main()
