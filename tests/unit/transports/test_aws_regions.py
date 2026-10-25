# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Region to DNS-suffix mapping used to derive AWS service hostnames.

Deliberately a pure lookup with no botocore dependency: it runs during config
validation, which must work whether or not the optional ``aiperf[aws]`` extra
is installed.
"""

import pytest
from pytest import param

from aiperf.transports.aws import regions
from aiperf.transports.aws.regions import dns_suffix, is_region_id


@pytest.mark.parametrize(
    "region,expected",
    [
        param("us-west-2", "amazonaws.com", id="commercial"),
        param("eu-central-1", "amazonaws.com", id="commercial-eu"),
        param("ap-southeast-1", "amazonaws.com", id="commercial-ap"),
        # China is the only partition with a different suffix.
        param("cn-north-1", "amazonaws.com.cn", id="china-north"),
        param("cn-northwest-1", "amazonaws.com.cn", id="china-northwest"),
        # GovCloud looks like a separate partition but keeps the commercial
        # suffix -- a natural place to guess wrong.
        param("us-gov-west-1", "amazonaws.com", id="govcloud"),
        param("us-gov-east-1", "amazonaws.com", id="govcloud-east"),
    ],
)
def test_dns_suffix_by_partition(region: str, expected: str) -> None:
    assert dns_suffix(region) == expected


def test_region_matching_is_case_insensitive() -> None:
    """Users paste regions from consoles and docs in mixed case."""
    assert dns_suffix("CN-NORTH-1") == "amazonaws.com.cn"


def test_unknown_region_falls_back_to_the_commercial_suffix() -> None:
    """New regions launch regularly. Guessing the common partition beats
    failing outright, and an explicit --url always overrides this anyway."""
    assert dns_suffix("xx-somewhere-9") == "amazonaws.com"


@pytest.mark.parametrize(
    "region",
    [
        param("us-west-2", id="commercial"),
        param("cn-northwest-1", id="china"),
        param("us-gov-west-1", id="govcloud"),
        param("us-isob-east-1", id="iso"),
        param("eusc-de-east-1", id="sovereign-cloud"),
        param("US-WEST-2", id="upper-case"),
        param("xx-somewhere-9", id="not-yet-launched"),
    ],
)  # fmt: skip
def test_region_shaped_ids_are_accepted(region: str) -> None:
    """Only the shape is checked, so regions launched after this release work."""
    assert is_region_id(region)


@pytest.mark.parametrize(
    "region",
    [
        param("evil.com/x", id="path"),
        param("evil.com#", id="fragment"),
        param("us-west-2.evil.com", id="extra-labels"),
        param("user@evil.com", id="userinfo"),
        param("us-west-2:8443", id="port"),
        param("us-west-2\n", id="trailing-newline"),
        param("us-west-\u0662", id="non-ascii-digit"),
        param("\u212a\u212a-west-2", id="kelvin-sign-folds-to-k"),
        param("auto", id="no-number"),
        param("", id="empty"),
    ],
)  # fmt: skip
def test_anything_that_could_leave_the_hostname_label_is_rejected(
    region: str,
) -> None:
    """The region is spliced between two DNS labels of a signed request's host,
    so a ``.``, ``/``, ``#``, ``@`` or ``:`` would move that request -- body and
    session token included -- to a host outside AWS."""
    assert not is_region_id(region)


@pytest.mark.parametrize(
    "region",
    [
        param("us-west-2", id="aws"),
        param("cn-northwest-1", id="aws-cn"),
        param("us-gov-west-1", id="aws-us-gov"),
        param("us-iso-east-1", id="aws-iso"),
        param("us-isob-east-1", id="aws-iso-b"),
        param("eu-isoe-west-1", id="aws-iso-e"),
        param("us-isof-south-1", id="aws-iso-f"),
        param("eusc-de-east-1", id="aws-eusc"),
        param("eusc-de-west-9", id="aws-eusc-unlisted-region"),
    ],
)  # fmt: skip
def test_the_derived_host_matches_botocore_in_every_partition(region: str) -> None:
    """The EU sovereign cloud and the four ISO partitions do not use
    amazonaws.com, so a suffix guessed from the region prefix sent requests to
    the wrong host. botocore's own resolver is the oracle."""
    botocore_loaders = pytest.importorskip("botocore.loaders")
    from botocore.regions import EndpointResolver

    resolver = EndpointResolver(botocore_loaders.create_loader().load_data("endpoints"))
    expected = resolver.construct_endpoint("runtime.sagemaker", region)["hostname"]

    assert f"runtime.sagemaker.{region}.{dns_suffix(region)}" == expected


def test_without_botocore_the_commercial_and_china_fallback_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config validation must work without the aiperf[aws] extra."""
    monkeypatch.setattr(regions, "_partitions", lambda: ())

    assert dns_suffix("us-west-2") == "amazonaws.com"
    assert dns_suffix("cn-north-1") == "amazonaws.com.cn"
