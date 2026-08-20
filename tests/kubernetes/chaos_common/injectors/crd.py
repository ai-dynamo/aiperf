# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CR-level and operator-pod fault injector for the unified-chaos interface.

Generalizes the AIPerf-specific ``ChaosInjector`` (CR kind / API group /
operator namespace / operator selector are constructor arguments) so the
same injector class can run against any operator-driven CRD (AIPerfJob,
DynamoGraphDeployment, ...). Handlers it owns:

* ``crd.delete`` -- ``kubectl delete <kind> <name> -n <ns> --wait=false``.
* ``crd.delete_twice`` -- rapid double-delete (~0.5 s spacing) to exercise
  finalizer idempotence on the apiserver.
* ``crd.apply_invalid`` -- apply a caller-supplied malformed CR manifest;
  restore deletes the CR with ``--ignore-not-found``.
* ``crd.patch`` -- ``kubectl patch`` against the CR. Restore runs the
  caller-supplied inverse patch in ``spec.params["restore_patch"]``; if
  absent, restore logs a warning and no-ops.
* ``crd.annotate`` -- stamp an annotation via ``kubectl annotate``. Restore
  clears the annotation via the ``key-`` removal syntax.
* ``operator.kill`` -- ``kubectl delete pod -l <selector> -n <ns> --force``.
  Restore is a no-op (the ReplicaSet brings the pod back).
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


_DELETE_TWICE_SPACING_S: float = 0.5
"""Sleep between the two delete calls in ``crd.delete_twice``.

Chosen large enough that the apiserver has had a chance to commit the
first delete (so the second one races finalizer removal, not the initial
write), and small enough that the test stays fast.
"""


class _CRDAppliedFault(AppliedFault):
    """Restore handle for a CR-targeted or operator-pod fault.

    The ``restore`` callable is captured at construction so each fault_id's
    dispatch branch can build its own inverse operation without subclassing.
    """

    def __init__(
        self,
        spec: FaultSpec,
        restore_coro_factory: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(spec=spec, metadata=metadata)
        self._restore_coro_factory = restore_coro_factory

    async def restore(self) -> None:
        if self._restore_coro_factory is None:
            return
        await self._restore_coro_factory()


class CRDInjector(FaultInjector):
    """Inject CR-level and operator-pod faults against any operator-driven CRD.

    Construction is parameterized so the same class works against AIPerfJob
    (``cr_kind="aiperfjob"``, ``cr_api_group="aiperf.nvidia.com"``) and
    DynamoGraphDeployment (``cr_kind="dynamographdeployment"``,
    ``cr_api_group="nvidia.com"``) without subclassing. Use a separate
    ``CRDInjector`` instance per (operator, namespace) target.
    """

    HANDLES: ClassVar[tuple[str, ...]] = ("crd", "operator")

    def __init__(
        self,
        kubectl: KubectlClient,
        *,
        cr_kind: str,
        cr_api_group: str,
        operator_namespace: str,
        operator_selector: str,
    ) -> None:
        """Initialize the injector.

        Args:
            kubectl: Async kubectl wrapper pinned to the target cluster.
            cr_kind: Lowercase singular CR kind, e.g. ``"aiperfjob"`` or
                ``"dynamographdeployment"``. Passed verbatim to kubectl.
            cr_api_group: API group of the CRD, e.g. ``"aiperf.nvidia.com"``.
                Currently only stored for diagnostic use; reserved for future
                manifest validation.
            operator_namespace: Namespace hosting the operator Deployment.
            operator_selector: kubectl ``-l`` selector that uniquely matches
                the operator Pod(s).
        """
        self._kubectl = kubectl
        self._cr_kind = cr_kind
        self._cr_api_group = cr_api_group
        self._operator_namespace = operator_namespace
        self._operator_selector = operator_selector

    async def inject(self, spec: FaultSpec) -> AppliedFault:
        fault_id = spec.fault_id
        if fault_id == "crd.delete":
            return await self._inject_crd_delete(spec)
        if fault_id == "crd.delete_twice":
            return await self._inject_crd_delete_twice(spec)
        if fault_id == "crd.apply_invalid":
            return await self._inject_crd_apply_invalid(spec)
        if fault_id == "crd.patch":
            return await self._inject_crd_patch(spec)
        if fault_id == "crd.annotate":
            return await self._inject_crd_annotate(spec)
        if fault_id == "operator.kill":
            return await self._inject_operator_kill(spec)
        raise FaultPreconditionError(
            f"CRDInjector does not handle fault_id={fault_id!r}; "
            f"supported: crd.delete, crd.delete_twice, crd.apply_invalid, "
            f"crd.patch, crd.annotate, operator.kill"
        )

    def _require_cr_target(self, spec: FaultSpec) -> tuple[str, str]:
        """Extract ``(namespace, name)`` from ``spec.target`` or raise."""
        ns = spec.target.get("ns")
        name = spec.target.get("name")
        if not ns or not name:
            raise FaultPreconditionError(
                f"fault_id={spec.fault_id!r} requires "
                f"target={{'ns': ..., 'name': ...}}; got target={spec.target!r}"
            )
        return ns, name

    async def _delete_cr_no_wait(self, ns: str, name: str) -> int:
        """Issue ``kubectl delete <kind> <name> -n <ns> --wait=false``.

        Returns the kubectl exit code. NotFound is *not* swallowed here so
        the caller can decide whether to treat it as success (the wrappers
        for ``crd.delete`` and ``crd.delete_twice`` both pass ``check=False``
        and accept any rc).
        """
        result = await self._kubectl.run(
            "delete",
            self._cr_kind,
            name,
            "-n",
            ns,
            "--wait=false",
            check=False,
        )
        return result.returncode

    async def _inject_crd_delete(self, spec: FaultSpec) -> AppliedFault:
        ns, name = self._require_cr_target(spec)
        rc = await self._delete_cr_no_wait(ns, name)
        # Non-zero rc here is informational, not fatal: a CR that disappeared
        # before the delete (e.g. operator already cleaned up) is acceptable
        # for the "force delete mid-run" chaos scenario.
        return _CRDAppliedFault(
            spec=spec,
            restore_coro_factory=None,
            metadata={"first_rc": rc},
        )

    async def _inject_crd_delete_twice(self, spec: FaultSpec) -> AppliedFault:
        ns, name = self._require_cr_target(spec)
        first_rc = await self._delete_cr_no_wait(ns, name)
        await asyncio.sleep(_DELETE_TWICE_SPACING_S)
        second_rc = await self._delete_cr_no_wait(ns, name)
        return _CRDAppliedFault(
            spec=spec,
            restore_coro_factory=None,
            metadata={"first_rc": first_rc, "second_rc": second_rc},
        )

    async def _inject_crd_apply_invalid(self, spec: FaultSpec) -> AppliedFault:
        ns, name = self._require_cr_target(spec)
        manifest = spec.params.get("manifest")
        if manifest is None:
            raise FaultPreconditionError(
                "fault_id='crd.apply_invalid' requires "
                "params['manifest']: a manifest dict or YAML string"
            )
        manifest_text = _serialize_manifest(manifest)
        apply_ok = True
        apply_error = ""
        try:
            await self._kubectl.apply(manifest_text, namespace=ns)
        except Exception as exc:
            # The apply itself is allowed to fail (the operator may reject
            # the malformed CR), so we wrap rather than propagate the raw
            # RuntimeError. Tests asserting "operator surfaces validation
            # error" inspect the CR after the apply attempt; the apply
            # returning non-zero here is itself a valid mechanism outcome
            # for some chaos scenarios.
            #
            # But record it. KubectlClient.apply raises with the apiserver's
            # exact rejection, and that string is the only evidence of why the
            # CR is absent. Discarded, downstream phase polls just report
            # phase='' -- which reads identically to "CR exists, operator never
            # wrote status" and is unfalsifiable from the test output alone.
            apply_ok = False
            apply_error = str(exc)
            logger.warning(
                f"crd.apply_invalid: kubectl apply for {ns}/{name} returned "
                f"non-success (may be expected for this scenario): {exc!r}"
            )

        async def _restore() -> None:
            await self._kubectl.run(
                "delete",
                self._cr_kind,
                name,
                "-n",
                ns,
                "--wait=false",
                "--ignore-not-found",
                check=False,
            )

        return _CRDAppliedFault(
            spec=spec,
            restore_coro_factory=_restore,
            metadata={"apply_ok": apply_ok, "apply_error": apply_error},
        )

    async def _inject_crd_patch(self, spec: FaultSpec) -> AppliedFault:
        ns, name = self._require_cr_target(spec)
        patch_type = spec.params.get("patch_type", "merge")
        patch_body = spec.params.get("patch")
        restore_patch = spec.params.get("restore_patch")
        if patch_body is None:
            raise FaultPreconditionError(
                "fault_id='crd.patch' requires params['patch']: the "
                "kubectl patch payload string"
            )

        result = await self._kubectl.run(
            "patch",
            self._cr_kind,
            name,
            "-n",
            ns,
            "--type",
            patch_type,
            "-p",
            patch_body,
            check=False,
        )
        if result.returncode != 0:
            raise FaultMechanismError(
                f"crd.patch failed (rc={result.returncode}) for "
                f"{self._cr_kind}/{ns}/{name}: stderr={result.stderr!r}"
            )

        async def _restore() -> None:
            if restore_patch is None:
                logger.warning(
                    lambda ns=ns, name=name: (
                        f"crd.patch restore: no params['restore_patch'] "
                        f"provided for {self._cr_kind}/{ns}/{name}; the "
                        f"forward patch will not be reversed"
                    )
                )
                return
            await self._kubectl.run(
                "patch",
                self._cr_kind,
                name,
                "-n",
                ns,
                "--type",
                patch_type,
                "-p",
                restore_patch,
                check=False,
            )

        return _CRDAppliedFault(spec=spec, restore_coro_factory=_restore)

    async def _inject_crd_annotate(self, spec: FaultSpec) -> AppliedFault:
        ns, name = self._require_cr_target(spec)
        key = spec.params.get("annotation_key")
        if not key:
            raise FaultPreconditionError(
                "fault_id='crd.annotate' requires params['annotation_key']"
            )
        value = spec.params.get("value", "")

        result = await self._kubectl.run(
            "annotate",
            self._cr_kind,
            name,
            "-n",
            ns,
            f"{key}={value}",
            "--overwrite",
            check=False,
        )
        if result.returncode != 0:
            raise FaultMechanismError(
                f"crd.annotate failed (rc={result.returncode}) for "
                f"{self._cr_kind}/{ns}/{name} key={key!r}: "
                f"stderr={result.stderr!r}"
            )

        async def _restore() -> None:
            await self._kubectl.run(
                "annotate",
                self._cr_kind,
                name,
                "-n",
                ns,
                f"{key}-",
                check=False,
            )

        return _CRDAppliedFault(
            spec=spec,
            restore_coro_factory=_restore,
            metadata={"annotation_key": key, "value": value},
        )

    async def _inject_operator_kill(self, spec: FaultSpec) -> AppliedFault:
        result = await self._kubectl.run(
            "delete",
            "pod",
            "-l",
            self._operator_selector,
            "-n",
            self._operator_namespace,
            "--force",
            "--grace-period=0",
            "--ignore-not-found",
            check=False,
        )
        if result.returncode != 0:
            raise FaultMechanismError(
                f"operator.kill failed (rc={result.returncode}) for "
                f"selector={self._operator_selector!r} in "
                f"namespace={self._operator_namespace!r}: "
                f"stderr={result.stderr!r}"
            )
        # ReplicaSet recreates the Pod; the test owns "wait for ready" via
        # whatever readiness probe it relies on. Restore is a no-op.
        return _CRDAppliedFault(
            spec=spec,
            restore_coro_factory=None,
            metadata={
                "selector": self._operator_selector,
                "namespace": self._operator_namespace,
            },
        )


def _serialize_manifest(manifest: Any) -> str:
    """Coerce a manifest dict-or-string into the YAML/JSON kubectl accepts.

    kubectl apply takes either YAML or JSON on stdin; orjson output is
    valid YAML for a single-document apply. A bare string is assumed to
    already be in apply-ready form (YAML or JSON).
    """
    if isinstance(manifest, str):
        return manifest
    return orjson.dumps(manifest).decode()
