# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = (REPOSITORY_ROOT / ".github/workflows/aiperf-k8s-operator.yml").read_text()
NATIVE_CONTRACT = (
    REPOSITORY_ROOT / "rust/e2e-tests/tests/kube_cli_contract.rs"
).read_text() + (
    REPOSITORY_ROOT / "rust/e2e-tests/tests/test_kube_sweep_parity.rs"
).read_text()


def workflow_job(name: str, next_name: str | None = None) -> str:
    """Return one top-level job body from the narrowly formatted workflow."""
    body = WORKFLOW.split(f"\n  {name}:\n", 1)[1]
    if next_name is not None:
        body = body.split(f"\n  {next_name}:\n", 1)[0]
    return body


def test_kind_job_selects_live_integration() -> None:
    """The provisioned kind job must not silently skip its live assertion."""
    kind_job = workflow_job("kind", "native-cli-kind")
    assert 'AIPERF_K8S_INTEGRATION: "1"' in kind_job


def test_workflow_triggers_for_its_contract_inputs() -> None:
    """Editing CI or the results sidecar must run the Kubernetes checks."""
    trigger = WORKFLOW.split("jobs:", 1)[0]
    for path in [
        ".github/workflows/aiperf-k8s-operator.yml",
        "rust/cli/src/results_sidecar.rs",
    ]:
        assert trigger.count(f'"{path}"') == 2


def test_kind_jobs_run_the_operator_image_built_from_the_checkout() -> None:
    """Live checks must not fall back to an externally published operator."""
    for job in (
        workflow_job("kind", "native-cli-kind"),
        workflow_job("native-cli-kind"),
    ):
        build = "docker build --tag aiperf-k8s-operator:ci ./aiperf-k8s-operator"
        load = (
            "kind load docker-image aiperf-k8s-operator:ci --name aiperf-k8s-operator"
        )
        helm = "helm install aiperf-k8s"
        assert job.index(build) < job.index(load) < job.index(helm)
        assert "cluster_name: aiperf-k8s-operator" in job
        assert "--set image.repository=aiperf-k8s-operator" in job
        assert "--set image.tag=ci" in job
        assert "--set image.pullPolicy=Never" in job

    dockerfile = REPOSITORY_ROOT / "aiperf-k8s-operator/Dockerfile"
    assert dockerfile.is_file()
    image_recipe = dockerfile.read_text()
    assert "COPY pyproject.toml README.md ./" in image_recipe
    assert "COPY src ./src" in image_recipe
    assert "pip install --no-cache-dir ." in image_recipe


def test_native_kind_job_runs_the_serial_durable_results_acceptance() -> None:
    """The native fixture must hand its image to the serial mutating acceptance."""
    kind_job = workflow_job("kind", "native-cli-kind")
    native_job = workflow_job("native-cli-kind")

    for job in (kind_job, native_job):
        assert "--namespace aiperf-system --create-namespace" in job

    ignored_attribute = "\n#[ignore]\n"
    assert NATIVE_CONTRACT.count(ignored_attribute) == 2
    assert "-- --ignored --test-threads=1" in native_job
    assert "AIPERF_E2E_OPERATOR_IMAGE: aiperf-k8s-operator:ci" in native_job
    live_contract = NATIVE_CONTRACT.split(ignored_attribute, 1)[1]
    assert 'Command::new("helm")' not in live_contract
    assert '"profile"' not in live_contract
    assert (
        "kind_results_survive_producer_deletion_and_operator_restart" in live_contract
    )
