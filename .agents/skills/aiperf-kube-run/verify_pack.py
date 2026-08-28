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

# Defaults the pack quotes as literals. If one of these changes, the pack text is
# wrong and must be updated in the same change that moves the default.
ENV_DEFAULTS = {
    "AIPERF_K8S_WATCH_DEFAULT_TIMEOUT_SECONDS": "600",
    "AIPERF_K8S_JOBSET_TTL_SECONDS_AFTER_FINISHED": "300",
    "AIPERF_K8S_JOBSET_DIRECT_MODE_TTL_SECONDS": "28800",
    "AIPERF_K8S_RECORDS_MANAGER_CPU": "75m",
    "AIPERF_K8S_SYSTEM_CONTROLLER_CPU": "75m",
    "AIPERF_K8S_CONTROLLER_HEARTBEAT_EXPIRY_SECONDS": "30.0",
    "AIPERF_K8S_JOBSET_WORKER_BACKOFF_LIMIT": "20",
    "AIPERF_K8S_JOBSET_SWEEP_AGGREGATE_INLINE_MAX_BYTES": "600000",
    "AIPERF_K8S_DIAGNOSIS_STALLED_PENDING_THRESHOLD_SECONDS": "60.0",
    "AIPERF_K8S_DIAGNOSIS_STALLED_RUNNING_THRESHOLD_SECONDS": "30.0",
    "AIPERF_K8S_DIAGNOSIS_HIGH_ERROR_RATE_THRESHOLD": "0.05",
    "AIPERF_K8S_DIAGNOSIS_FAIL_ABOVE_ERROR_RATE": "1.0",
    "AIPERF_K8S_DIAGNOSIS_HIGH_LATENCY_P99_MULTIPLIER": "10.0",
}

# Constant name -> value quoted by the pack.
CONSTANT_DEFAULTS = {
    "DEFAULT_BENCHMARK_NAMESPACE": "aiperf-benchmarks",
    "DEFAULT_OPERATOR_NAMESPACE": "aiperf-system",
}

# Helm values.yaml lines the setup skill reproduces verbatim.
HELM_DEFAULTS = [
    "repository: nvcr.io/nvidia/aiperf",
    "pullPolicy: IfNotPresent",
    "size: 1Ti",
    'accessMode: "ReadWriteOnce"',
    'name: "aiperf-benchmarks"',
]

# Literal facts the pack quotes from files outside the chart and env docs.
# Each entry is (repo path, substring that must still be present, what the pack says).
SOURCE_FACTS = [
    ("dev/versions.py", 'JOBSET_VERSION = "v0.8.0"', "JobSet pin is v0.8.0"),
    (
        "deploy/helm/aiperf-operator/values.yaml",
        "value: user-workload",
        "chart tolerates dedicated=user-workload by default",
    ),
    (
        "deploy/helm/aiperf-operator/templates/deployment.yaml",
        ".Values.imagePullSecrets",
        "chart imagePullSecrets applies to the operator Deployment",
    ),
]

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

        # Fold shell line-continuations so flags on later lines of a multi-line
        # example are scanned too; the command regex stops at a newline.
        folded = re.sub(r"\\\n\s*", " ", body)
        for sub, rest in re.findall(r"aiperf kube ([a-z-]+)((?: [^\n`|]*)?)", folded):
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

    documented = dict(
        re.findall(r"^\| `(AIPERF_[A-Z0-9_]+)` \| `?([^|`]*?)`? \|", env_doc, re.M)
    )
    for var, expected in ENV_DEFAULTS.items():
        actual = documented.get(var)
        if actual != expected:
            fail(f"default drift: {var} is {actual!r}, pack says {expected!r}")

    constants = (REPO / "src/aiperf/kubernetes/constants.py").read_text()
    for const, expected in CONSTANT_DEFAULTS.items():
        if f'{const} = "{expected}"' not in constants:
            fail(f"constant drift: {const} is no longer {expected!r}")

    for line in HELM_DEFAULTS:
        if line not in values:
            fail(f"Helm default drift: values.yaml no longer contains {line!r}")

    for rel, needle, claim in SOURCE_FACTS:
        if needle not in (REPO / rel).read_text():
            fail(f"source drift: {rel} no longer supports the claim that {claim}")

    print(
        f"checked {len(ENV_DEFAULTS)} env defaults, "
        f"{len(CONSTANT_DEFAULTS)} constants, {len(HELM_DEFAULTS)} Helm defaults, "
        f"{len(SOURCE_FACTS)} source facts"
    )
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURES"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
