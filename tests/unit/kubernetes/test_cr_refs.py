# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.cr_refs.

``cr_refs`` is the canonical source for (group, version, plural) CRD triples.
Any divergence between the constants and the CRD manifests means CustomObjectsApi
calls fail. These tests pin the values against the CRD manifest and against
their format.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from aiperf.kubernetes import cr_refs

_CRD_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "deploy"
    / "helm"
    / "aiperf-operator"
    / "templates"
    / "crd.yaml"
)


class TestAIPerfJobRefs:
    """AIPerfJob CRD coordinates pin to the Helm chart manifest."""

    def test_group_version_plural_values(self) -> None:
        assert cr_refs.AIPERF_JOB_GROUP == "aiperf.nvidia.com"
        assert cr_refs.AIPERF_JOB_VERSION == "v1alpha1"
        assert cr_refs.AIPERF_JOB_PLURAL == "aiperfjobs"

    def test_api_version_is_group_slash_version(self) -> None:
        assert cr_refs.AIPERF_JOB_API_VERSION == "aiperf.nvidia.com/v1alpha1"
        expected = f"{cr_refs.AIPERF_JOB_GROUP}/{cr_refs.AIPERF_JOB_VERSION}"
        assert expected == cr_refs.AIPERF_JOB_API_VERSION

    def test_backward_compat_aliases_match_canonical(self) -> None:
        """Several modules still import the short names — they must point at
        the same values, otherwise upgrades silently call the wrong endpoint."""
        assert cr_refs.AIPERF_GROUP == cr_refs.AIPERF_JOB_GROUP
        assert cr_refs.AIPERF_VERSION == cr_refs.AIPERF_JOB_VERSION
        assert cr_refs.AIPERF_PLURAL == cr_refs.AIPERF_JOB_PLURAL
        assert cr_refs.AIPERF_API_VERSION == cr_refs.AIPERF_JOB_API_VERSION


class TestJobSetRefs:
    """JobSet CRD coordinates must match the upstream jobset-operator."""

    def test_group_version_plural_values(self) -> None:
        assert cr_refs.JOBSET_GROUP == "jobset.x-k8s.io"
        assert cr_refs.JOBSET_VERSION == "v1alpha2"
        assert cr_refs.JOBSET_PLURAL == "jobsets"

    def test_api_version_is_group_slash_version(self) -> None:
        assert cr_refs.JOBSET_API_VERSION == "jobset.x-k8s.io/v1alpha2"


class TestKueueRefs:
    """Kueue CRD coordinates (optional, used by operator preflight)."""

    def test_group_version_plural_values(self) -> None:
        assert cr_refs.KUEUE_GROUP == "kueue.x-k8s.io"
        assert cr_refs.KUEUE_VERSION == "v1beta1"
        assert cr_refs.KUEUE_LOCALQUEUE_PLURAL == "localqueues"


class TestFormatInvariants:
    """Every CRD triple must be shaped correctly to survive CustomObjectsApi."""

    @pytest.mark.parametrize(
        "group",
        [cr_refs.AIPERF_JOB_GROUP, cr_refs.JOBSET_GROUP, cr_refs.KUEUE_GROUP],
    )
    def test_group_is_dns_style_domain(self, group: str) -> None:
        """Kubernetes API groups are DNS-style — must have at least one dot and
        only lowercase letters, digits, dots, and hyphens."""
        assert "." in group
        assert re.fullmatch(r"[a-z0-9.\-]+", group)

    @pytest.mark.parametrize(
        "version",
        [cr_refs.AIPERF_JOB_VERSION, cr_refs.JOBSET_VERSION, cr_refs.KUEUE_VERSION],
    )
    def test_version_is_lowercase_v_prefixed(self, version: str) -> None:
        """API versions are always 'v<N>' optionally with 'alpha'/'beta' suffix."""
        assert re.fullmatch(r"v\d+((alpha|beta)\d+)?", version)

    @pytest.mark.parametrize(
        "plural",
        [
            cr_refs.AIPERF_JOB_PLURAL,
            cr_refs.JOBSET_PLURAL,
            cr_refs.KUEUE_LOCALQUEUE_PLURAL,
        ],
    )
    def test_plural_is_lowercase_ends_in_s(self, plural: str) -> None:
        """Plurals are routed into the REST URL path; kubectl convention is
        lowercase with a trailing 's'."""
        assert plural == plural.lower()
        assert plural.endswith("s")


@pytest.mark.skipif(
    not _CRD_MANIFEST.exists(),
    reason=f"CRD manifest not found at {_CRD_MANIFEST}",
)
class TestCRDManifestPinning:
    """cr_refs must stay in sync with the Helm CRD manifest."""

    def test_aiperf_group_matches_crd_manifest(self) -> None:
        manifest_text = _CRD_MANIFEST.read_text()
        # The Helm template wraps with Go template syntax; extract group via regex.
        match = re.search(r"group:\s*([a-z0-9.\-]+)", manifest_text)
        assert match, "CRD manifest missing group: field"
        assert match.group(1) == cr_refs.AIPERF_JOB_GROUP

    def test_aiperf_plural_matches_crd_manifest(self) -> None:
        manifest_text = _CRD_MANIFEST.read_text()
        match = re.search(r"plural:\s*([a-z0-9\-]+)", manifest_text)
        assert match, "CRD manifest missing plural: field"
        assert match.group(1) == cr_refs.AIPERF_JOB_PLURAL

    def test_aiperf_version_matches_crd_manifest(self) -> None:
        """The CRD declares its served versions under spec.versions[*].name."""
        manifest_text = _CRD_MANIFEST.read_text()
        # Strip Helm template braces so yaml.safe_load can parse.
        cleaned = re.sub(r"{{[^}]*}}", "PLACEHOLDER", manifest_text)
        try:
            doc = yaml.safe_load(cleaned)
        except yaml.YAMLError:
            pytest.skip("CRD manifest uses Helm templating that can't be stripped")
        versions = doc.get("spec", {}).get("versions", [])
        names = [v.get("name") for v in versions]
        assert cr_refs.AIPERF_JOB_VERSION in names
