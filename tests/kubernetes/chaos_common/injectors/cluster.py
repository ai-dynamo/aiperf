# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster-scoped :py:class:`FaultInjector` for the unified-chaos interface.

Handles fault ids under the ``cluster.`` namespace (see spec §3.4):

* ``cluster.resource_quota`` -- apply / delete a ``ResourceQuota`` on a
  namespace. Restore deletes the quota; the inject site also records a
  :py:class:`recovery.ClusterScopedMutation` so a crashed session can be
  swept via ``pytest --chaos-sweep`` (Phase 1 plumbing).
* ``cluster.network_policy.deny_egress`` -- apply a NetworkPolicy that
  denies egress (optionally allowing intra-cluster traffic). Restore
  deletes the policy. Only ENFORCES on NetworkPolicy-aware CNIs
  (Cilium/Calico); on kindnet the policy applies but does nothing -- see
  ``chaos_common/README.md`` Cilium-on-kind section. Tests that depend on
  actual egress blocking must gate on ``cilium_on_kind_required``.
* ``cluster.rbac.revoke`` -- ``kubectl patch`` a Role or ClusterRole to
  drop a single (api_group, resource, verb) tuple. Restore re-adds the
  verb. Uses JSON patch (``--type=json``) so the rule/verb indices are
  unambiguous on both sides.
"""

from __future__ import annotations

from typing import Any, ClassVar

import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos_common import recovery
from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


class _ResourceQuotaAppliedFault(AppliedFault):
    """Restore handle for a ``cluster.resource_quota`` injection.

    ``metadata`` keys:

    * ``namespace`` -- target namespace.
    * ``name`` -- ResourceQuota resource name.
    * ``hard_limits`` -- the limits dict that was applied (for diagnostics).
    """

    def __init__(
        self,
        spec: FaultSpec,
        chaos: ChaosInjector,
        namespace: str,
        name: str,
        hard_limits: dict[str, str],
    ) -> None:
        super().__init__(
            spec=spec,
            metadata={
                "namespace": namespace,
                "name": name,
                "hard_limits": dict(hard_limits),
            },
        )
        self._chaos = chaos
        self._namespace = namespace
        self._name = name

    async def restore(self) -> None:
        # `delete_resource_quota` already swallows NotFound via
        # `--ignore-not-found`, so this is idempotent by construction.
        try:
            await self._chaos.delete_resource_quota(self._namespace, self._name)
        except Exception as exc:
            raise FaultMechanismError(
                f"failed to delete ResourceQuota {self._name!r} in "
                f"namespace {self._namespace!r}: {exc!r}"
            ) from exc


class _NetworkPolicyAppliedFault(AppliedFault):
    """Restore handle for a ``cluster.network_policy.deny_egress`` injection.

    ``metadata`` keys:

    * ``namespace`` -- target namespace.
    * ``name`` -- NetworkPolicy resource name.
    * ``allow_cluster_egress`` -- whether intra-cluster egress was allowed.
    """

    def __init__(
        self,
        spec: FaultSpec,
        kubectl: KubectlClient,
        namespace: str,
        name: str,
        allow_cluster_egress: bool,
    ) -> None:
        super().__init__(
            spec=spec,
            metadata={
                "namespace": namespace,
                "name": name,
                "allow_cluster_egress": allow_cluster_egress,
            },
        )
        self._kubectl = kubectl
        self._namespace = namespace
        self._name = name

    async def restore(self) -> None:
        try:
            await self._kubectl.run(
                "delete",
                "networkpolicy",
                self._name,
                "-n",
                self._namespace,
                "--ignore-not-found",
                check=False,
            )
        except Exception as exc:
            raise FaultMechanismError(
                f"failed to delete NetworkPolicy {self._name!r} in "
                f"namespace {self._namespace!r}: {exc!r}"
            ) from exc


class _RbacRevokeAppliedFault(AppliedFault):
    """Restore handle for a ``cluster.rbac.revoke`` injection.

    ``metadata`` keys:

    * ``scope`` -- ``"role"`` or ``"clusterrole"``.
    * ``name`` -- role/clusterrole name.
    * ``namespace`` -- namespace for ``scope=role``; ``None`` for clusterrole.
    * ``api_group`` / ``resource`` / ``verb`` -- the revoked tuple.
    * ``rule_index`` -- index of the matched rule (used by restore).
    """

    def __init__(
        self,
        spec: FaultSpec,
        kubectl: KubectlClient,
        scope: str,
        name: str,
        namespace: str | None,
        api_group: str,
        resource: str,
        verb: str,
        rule_index: int,
    ) -> None:
        super().__init__(
            spec=spec,
            metadata={
                "scope": scope,
                "name": name,
                "namespace": namespace,
                "api_group": api_group,
                "resource": resource,
                "verb": verb,
                "rule_index": rule_index,
            },
        )
        self._kubectl = kubectl
        self._scope = scope
        self._name = name
        self._namespace = namespace
        self._verb = verb
        self._rule_index = rule_index

    async def restore(self) -> None:
        # Re-add the verb at the end of the rule's verbs list. JSON-patch
        # `add` to `/rules/<idx>/verbs/-` appends; if the verb is already
        # present (e.g. operator/admin reconciled it back), we tolerate
        # the resulting apiserver behaviour rather than hard-fail.
        patch = [
            {
                "op": "add",
                "path": f"/rules/{self._rule_index}/verbs/-",
                "value": self._verb,
            }
        ]
        args = [
            "patch",
            self._scope,
            self._name,
        ]
        if self._namespace is not None:
            args.extend(["-n", self._namespace])
        args.extend(["--type=json", "-p", orjson.dumps(patch).decode()])
        try:
            await self._kubectl.run(*args, check=False)
        except Exception as exc:
            raise FaultMechanismError(
                f"failed to restore verb {self._verb!r} on {self._scope} "
                f"{self._name!r} (ns={self._namespace!r}): {exc!r}"
            ) from exc


class ClusterInjector(FaultInjector):
    """Injector for cluster-scoped fault primitives.

    Implements ``cluster.resource_quota``, ``cluster.network_policy.deny_egress``,
    and ``cluster.rbac.revoke``. All three record a
    :py:class:`recovery.ClusterScopedMutation` BEFORE the apiserver call so a
    session crash between record and apply leaves a no-op sweep entry rather
    than a missing one.
    """

    HANDLES: ClassVar[tuple[str, ...]] = ("cluster",)

    def __init__(self, kubectl: KubectlClient) -> None:
        self._kubectl = kubectl

    async def inject(self, spec: FaultSpec) -> AppliedFault:
        if spec.fault_id == "cluster.resource_quota":
            return await self._inject_resource_quota(spec)
        if spec.fault_id == "cluster.network_policy.deny_egress":
            return await self._inject_network_policy_deny_egress(spec)
        if spec.fault_id == "cluster.rbac.revoke":
            return await self._inject_rbac_revoke(spec)
        raise FaultPreconditionError(
            f"ClusterInjector does not implement fault_id={spec.fault_id!r}"
        )

    async def _inject_resource_quota(self, spec: FaultSpec) -> AppliedFault:
        namespace = self._require(spec.target, "ns", where="spec.target")
        name = self._require(spec.params, "name", where="spec.params")
        hard_limits_raw = self._require(spec.params, "hard_limits", where="spec.params")
        if not isinstance(hard_limits_raw, dict):
            raise FaultPreconditionError(
                "cluster.resource_quota requires spec.params['hard_limits'] "
                f"to be a dict; got {type(hard_limits_raw).__name__}"
            )
        hard_limits: dict[str, str] = dict(hard_limits_raw)

        # Record BEFORE the apply so a crash between record + apply leaves
        # a sweep entry the recovery cache can no-op on (delete with
        # --ignore-not-found). Recovery cache write failures are logged
        # and swallowed so a flaky disk cannot block the test itself.
        self._safe_record(
            recovery.ClusterScopedMutation(
                kind="resourcequota",
                api_version="v1",
                name=name,
                op="create",
                namespace=namespace,
            )
        )

        chaos = ChaosInjector(self._kubectl)
        try:
            await chaos.apply_resource_quota(namespace, name, hard_limits)
        except Exception as exc:
            raise FaultMechanismError(
                f"failed to apply ResourceQuota {name!r} in namespace "
                f"{namespace!r} with hard={hard_limits!r}: {exc!r}"
            ) from exc

        return _ResourceQuotaAppliedFault(
            spec=spec,
            chaos=chaos,
            namespace=namespace,
            name=name,
            hard_limits=hard_limits,
        )

    async def _inject_network_policy_deny_egress(self, spec: FaultSpec) -> AppliedFault:
        namespace = self._require(
            spec.target, "ns", where="spec.target", fault="network_policy.deny_egress"
        )
        name = self._require(
            spec.params, "name", where="spec.params", fault="network_policy.deny_egress"
        )
        allow_cluster_egress_raw = spec.params.get("allow_cluster_egress", True)
        if not isinstance(allow_cluster_egress_raw, bool):
            raise FaultPreconditionError(
                "cluster.network_policy.deny_egress requires spec.params"
                "['allow_cluster_egress'] to be a bool when set; got "
                f"{type(allow_cluster_egress_raw).__name__}"
            )
        allow_cluster_egress: bool = allow_cluster_egress_raw

        manifest_obj: dict[str, Any] = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {"name": name, "namespace": namespace},
            "spec": {
                "podSelector": {},
                "policyTypes": ["Egress"],
            },
        }
        if allow_cluster_egress:
            # Allow egress to any pod in any namespace; external destinations
            # remain blocked. On kindnet this is a no-op (policy ignored).
            manifest_obj["spec"]["egress"] = [
                {"to": [{"namespaceSelector": {}}]},
            ]

        self._safe_record(
            recovery.ClusterScopedMutation(
                kind="networkpolicy",
                api_version="networking.k8s.io/v1",
                name=name,
                op="create",
                namespace=namespace,
            )
        )

        try:
            await self._kubectl.apply(
                orjson.dumps(manifest_obj).decode(), namespace=namespace
            )
        except Exception as exc:
            raise FaultMechanismError(
                f"failed to apply NetworkPolicy {name!r} in namespace "
                f"{namespace!r}: {exc!r}"
            ) from exc

        return _NetworkPolicyAppliedFault(
            spec=spec,
            kubectl=self._kubectl,
            namespace=namespace,
            name=name,
            allow_cluster_egress=allow_cluster_egress,
        )

    async def _inject_rbac_revoke(self, spec: FaultSpec) -> AppliedFault:
        scope_raw = self._require(
            spec.target, "scope", where="spec.target", fault="rbac.revoke"
        )
        if scope_raw not in ("role", "clusterrole"):
            raise FaultPreconditionError(
                "cluster.rbac.revoke requires spec.target['scope'] to be "
                f"'role' or 'clusterrole'; got {scope_raw!r}"
            )
        scope: str = scope_raw
        name = self._require(
            spec.target, "name", where="spec.target", fault="rbac.revoke"
        )
        namespace: str | None
        if scope == "role":
            namespace = self._require(
                spec.target, "ns", where="spec.target", fault="rbac.revoke"
            )
        else:
            if spec.target.get("ns") is not None:
                raise FaultPreconditionError(
                    "cluster.rbac.revoke with scope='clusterrole' must not "
                    "specify spec.target['ns']; clusterroles are cluster-scoped"
                )
            namespace = None

        api_group = self._require(
            spec.params, "api_group", where="spec.params", fault="rbac.revoke"
        )
        resource = self._require(
            spec.params, "resource", where="spec.params", fault="rbac.revoke"
        )
        verb = self._require(
            spec.params, "verb", where="spec.params", fault="rbac.revoke"
        )

        # Read the current role to discover rule + verb indices.
        get_args = ["get", scope, name]
        if namespace is not None:
            get_args.extend(["-n", namespace])
        get_args.extend(["-o", "json"])
        try:
            result = await self._kubectl.run(*get_args, check=True)
        except Exception as exc:
            raise FaultMechanismError(
                f"failed to read {scope} {name!r} (ns={namespace!r}) for "
                f"rbac.revoke: {exc!r}"
            ) from exc

        try:
            body: dict[str, Any] = orjson.loads(result.stdout or b"{}")
        except orjson.JSONDecodeError as exc:
            raise FaultMechanismError(
                f"kubectl returned non-JSON body for {scope} {name!r}: {exc!r}"
            ) from exc

        rules: list[dict[str, Any]] = body.get("rules") or []
        rule_index, verb_index = _find_rule_and_verb(
            rules, api_group=api_group, resource=resource, verb=verb
        )
        if rule_index is None or verb_index is None:
            raise FaultPreconditionError(
                f"cluster.rbac.revoke could not locate verb {verb!r} on "
                f"({api_group!r}, {resource!r}) in {scope} {name!r} "
                f"(ns={namespace!r}); the rule may already be absent"
            )

        self._safe_record(
            recovery.ClusterScopedMutation(
                kind="role" if scope == "role" else "clusterrole",
                api_version="rbac.authorization.k8s.io/v1",
                name=name,
                op="patch",
                namespace=namespace if scope == "role" else None,
                payload={
                    "rule_index": rule_index,
                    "verb": verb,
                    "api_group": api_group,
                    "resource": resource,
                },
            )
        )

        patch = [
            {
                "op": "remove",
                "path": f"/rules/{rule_index}/verbs/{verb_index}",
            }
        ]
        patch_args = ["patch", scope, name]
        if namespace is not None:
            patch_args.extend(["-n", namespace])
        patch_args.extend(["--type=json", "-p", orjson.dumps(patch).decode()])
        try:
            await self._kubectl.run(*patch_args, check=True)
        except Exception as exc:
            raise FaultMechanismError(
                f"failed to patch {scope} {name!r} to drop verb {verb!r} "
                f"on ({api_group!r}, {resource!r}): {exc!r}"
            ) from exc

        return _RbacRevokeAppliedFault(
            spec=spec,
            kubectl=self._kubectl,
            scope=scope,
            name=name,
            namespace=namespace,
            api_group=api_group,
            resource=resource,
            verb=verb,
            rule_index=rule_index,
        )

    @staticmethod
    def _safe_record(mutation: recovery.ClusterScopedMutation) -> None:
        try:
            recovery.record_mutation(mutation)
        except Exception as exc:
            logger.warning(
                lambda exc=exc, m=mutation: (
                    f"failed to record chaos-sweep mutation "
                    f"{m.kind}/{m.name} (ns={m.namespace!r}): {exc!r}"
                )
            )

    @staticmethod
    def _require(
        source: dict[str, Any],
        key: str,
        *,
        where: str,
        fault: str = "resource_quota",
    ) -> Any:
        if key not in source or source[key] is None:
            raise FaultPreconditionError(
                f"missing required field {where}[{key!r}] for cluster.{fault}"
            )
        return source[key]


def _find_rule_and_verb(
    rules: list[dict[str, Any]],
    *,
    api_group: str,
    resource: str,
    verb: str,
) -> tuple[int | None, int | None]:
    """Locate the (rule_index, verb_index) for an (api_group, resource, verb).

    Returns ``(None, None)`` when no rule matches both the api_group + resource
    AND contains the requested verb. RBAC rules are AND-ed across apiGroups,
    resources, and verbs lists, so the matching rule must list all three.
    """
    for r_idx, rule in enumerate(rules):
        groups = rule.get("apiGroups") or []
        resources = rule.get("resources") or []
        verbs = rule.get("verbs") or []
        if api_group not in groups:
            continue
        if resource not in resources:
            continue
        if verb not in verbs:
            continue
        return r_idx, verbs.index(verb)
    return None, None
