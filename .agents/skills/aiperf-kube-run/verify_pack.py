"""Mechanical verifier for the aiperf-kube-* skill pack.

Checks every ``aiperf kube`` subcommand and flag, referenced repo path,
``AIPERF_*`` environment variable, and Helm value named in the pack against the
live CLI and the working tree.

Run from anywhere: ``uv run python .agents/skills/aiperf-kube-run/verify_pack.py``
"""

import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
SKILLS = REPO / ".agents" / "skills"
PACK = [
    "aiperf-kube-run",
    "aiperf-kube-setup",
    "aiperf-kube-triage",
    "aiperf-kube-sweep",
]
SHORT_FLAGS = {"-o", "-n", "-j", "-v", "-t", "-a", "-i", "-e", "-w", "-f", "-A", "-d"}
HELM_ROOTS = {
    "image",
    "storage",
    "benchmarkNamespace",
    "operator",
    "kueue",
    "dashboard",
    "serviceMonitor",
    "rbac",
    "networkPolicy",
    "resultsServer",
    "serviceAccount",
}
# Env vars that are read by aiperf but intentionally absent from the generated
# env-var doc (set by the operator on pods, not by users).
ENV_ALLOWLIST = {"AIPERF_OPERATOR_MANAGED"}

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    print("FAIL:", message)


def cli_help(*args: str) -> str:
    env = {**os.environ, "COLUMNS": "200"}
    return subprocess.run(
        ["uv", "run", "aiperf", "kube", *args, "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    ).stdout


def main() -> int:
    subcommands = set(re.findall(r"^│ ([a-z-]+) ", cli_help(), re.M))
    print("subcommands:", " ".join(sorted(subcommands)))
    flags = {}
    for sub in sorted(subcommands):
        text = cli_help(sub)
        flags[sub] = set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", text)) | set(
            re.findall(r"\s(-[a-zA-Z])\s", text)
        )

    env_doc = (REPO / "docs" / "environment-variables.md").read_text()
    values = (REPO / "deploy/helm/aiperf-operator/values.yaml").read_text()

    for name in PACK:
        path = SKILLS / name / "SKILL.md"
        if not path.exists():
            fail(f"{name}: missing SKILL.md")
            continue
        body = path.read_text()

        matched = re.match(r"^---\nname: (.+)\ndescription: (.+)\n---\n", body)
        if not matched:
            fail(f"{name}: unparseable frontmatter")
            continue
        if matched.group(1) != name:
            fail(f"{name}: frontmatter name is {matched.group(1)}")
        frontmatter_len = len(body.split("---\n")[1])
        if frontmatter_len > 1024:
            fail(f"{name}: frontmatter {frontmatter_len} chars exceeds 1024")
        if not matched.group(2).startswith("Use when"):
            fail(f"{name}: description must start with 'Use when'")
        print(
            f"{name}: frontmatter ok ({frontmatter_len} chars), "
            f"{len(body.split())} words"
        )

        for sub, rest in re.findall(r"aiperf kube ([a-z-]+)((?: [^\n`|]*)?)", body):
            if sub not in subcommands:
                fail(f"{name}: unknown subcommand 'aiperf kube {sub}'")
                continue
            pattern = r"(?<![\w-])(--[a-zA-Z][a-zA-Z0-9-]*|-[a-zA-Z])(?![\w-])"
            for flag in re.findall(pattern, rest):
                known = flag in SHORT_FLAGS or flag.startswith("--")
                if known and flag not in flags[sub]:
                    fail(f"{name}: 'aiperf kube {sub}' has no flag {flag}")

        for ref in set(re.findall(r"`([a-zA-Z0-9_./-]+\.(?:md|py|yaml|yml))`", body)):
            candidates = [ref]
            if not ref.startswith(("docs/", "src/", "deploy/", "tools/")):
                candidates.append("docs/kubernetes/" + ref.split("/")[-1])
            if not any((REPO / c.split("#")[0]).exists() for c in candidates):
                fail(f"{name}: referenced path not found: {ref}")

        for ref in set(re.findall(r"`(docs/[a-zA-Z0-9_./-]+)`", body)):
            if not (REPO / ref.split("#")[0]).exists():
                fail(f"{name}: referenced doc not found: {ref}")

        for var in set(re.findall(r"AIPERF_[A-Z0-9_]+", body)):
            if var not in env_doc and var not in ENV_ALLOWLIST:
                fail(f"{name}: env var not in docs/environment-variables.md: {var}")

        for value in set(
            re.findall(r"`([a-z][a-zA-Z]*(?:\.[a-zA-Z][a-zA-Z]*)+)`", body)
        ):
            if value.split(".")[0] in HELM_ROOTS and value.split(".")[-1] not in values:
                fail(f"{name}: Helm value not in values.yaml: {value}")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURES"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
