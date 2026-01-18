# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for plot package.

Provides shared utilities for metric parsing, data preparation, and other
cross-cutting concerns used by multiple plot modules.
"""

import re

import orjson

from aiperf.plot.metric_names import _format_server_metric_name


def parse_server_metric_spec(metric_spec: str) -> tuple[str, str | None, dict | None]:
    """
    Parse server metric specification with optional endpoint and label filters.

    Supports Prometheus-style metric filtering with flexible syntax for
    endpoint-specific and label-specific metric selection. Handles both
    endpoint and label filters simultaneously or independently.

    Formats:
        - "vllm:metric" → all endpoints and labels merged
        - "vllm:metric[http://localhost:8081/metrics]" → specific endpoint
        - "vllm:metric{label=value}" → specific labels
        - "vllm:metric[endpoint]{labels}" → both filters (NEW!)
        - "vllm:metric{label1=value1,label2=value2}" → multiple labels

    Quote handling:
        - Accepts both single quotes: {method='GET'}
        - And double quotes: {method="GET"}
        - Strips both for consistent matching

    Args:
        metric_spec: Metric specification string with optional filters

    Returns:
        Tuple of (metric_name, endpoint_url, labels_dict):
        - metric_name: Base metric name (e.g., "vllm:kv_cache_usage_perc")
        - endpoint_url: Endpoint filter or None
        - labels_dict: Labels filter dict or None

    Examples:
        >>> parse_server_metric_spec("vllm:cache_usage")
        ("vllm:cache_usage", None, None)

        >>> parse_server_metric_spec("vllm:cache[http://localhost:8081/metrics]")
        ("vllm:cache", "http://localhost:8081/metrics", None)

        >>> parse_server_metric_spec("vllm:requests{method='GET',status='200'}")
        ("vllm:requests", None, {"method": "GET", "status": "200"})

        >>> parse_server_metric_spec("vllm:cache[http://localhost:8081]{instance='worker-1'}")
        ("vllm:cache", "http://localhost:8081", {"instance": "worker-1"})
    """
    # Combined pattern: metric[endpoint]{labels}
    # Groups: (1) metric name, (2) endpoint (optional), (3) labels (optional)
    combined_match = re.match(
        r"^([^\[\{]+)(?:\[([^\]]+)\])?(?:\{([^\}]+)\})?$", metric_spec
    )

    if not combined_match:
        # Check for common syntax errors
        if "[" in metric_spec and "]" not in metric_spec:
            raise ValueError(
                f"Invalid metric specification: missing closing bracket ']' in '{metric_spec}'\n"
                f"Expected format: metric_name[endpoint]"
            )
        if "{" in metric_spec and "}" not in metric_spec:
            raise ValueError(
                f"Invalid metric specification: missing closing brace '}}' in '{metric_spec}'\n"
                f"Expected format: metric_name{{label1=value1,label2=value2}}"
            )
        # Fallback: return as-is if pattern doesn't match (may be simple metric name)
        return metric_spec.strip(), None, None

    metric_name = combined_match.group(1)
    endpoint = combined_match.group(2)  # None if not present
    labels_str = combined_match.group(3)  # None if not present

    # Parse labels if present
    labels = None
    if labels_str:
        labels = {}
        for pair in labels_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Invalid label syntax in '{metric_spec}': '{pair}' is missing '='\n"
                    f"Expected format: {{label1=value1,label2=value2}}"
                )
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Remove both single and double quotes from value
            value = value.strip("'").strip('"')
            if not key:
                raise ValueError(
                    f"Invalid label syntax in '{metric_spec}': empty label key\n"
                    f"Expected format: {{label1=value1,label2=value2}}"
                )
            labels[key] = value

    return metric_name.strip(), endpoint, labels


def filter_server_metrics_dataframe(
    df,
    metric_name: str,
    endpoint_filter: str | None = None,
    labels_filter: dict | None = None,
) -> tuple:
    """
    Filter server metrics DataFrame by metric name, endpoint, and labels.

    Applies filters in order: metric name → endpoint → labels. Converts
    timestamps to relative seconds and returns filtered DataFrame with
    metadata.

    Args:
        df: Server metrics DataFrame with columns:
            [timestamp_ns, endpoint_url, metric_name, metric_type,
             labels_json, unit, value, histogram_count, histogram_sum]
        metric_name: Base metric name to filter by
        endpoint_filter: Optional endpoint URL filter
        labels_filter: Optional labels dict filter

    Returns:
        Tuple of (filtered_df, unit, metric_type):
        - filtered_df: DataFrame with timestamp_s column added
        - unit: Unit from first row
        - metric_type: Metric type from first row

    Raises:
        ValueError: If no data remains after filtering
    """
    # Filter by metric name
    filtered = df[df["metric_name"] == metric_name].copy()

    if filtered.empty:
        available = df["metric_name"].unique().tolist()
        raise ValueError(
            f"Server metric '{metric_name}' not found. "
            f"Available: {available[:10]}{'...' if len(available) > 10 else ''}"
        )

    # Apply endpoint filter if specified
    if endpoint_filter:
        filtered = filtered[filtered["endpoint_url"] == endpoint_filter]
        if filtered.empty:
            available_endpoints = (
                df[df["metric_name"] == metric_name]["endpoint_url"].unique().tolist()
            )
            raise ValueError(
                f"No data for metric '{metric_name}' at endpoint '{endpoint_filter}'. "
                f"Available endpoints: {available_endpoints}"
            )

    # Apply labels filter if specified
    if labels_filter:
        labels_json = orjson.dumps(labels_filter, option=orjson.OPT_SORT_KEYS).decode()
        filtered = filtered[filtered["labels_json"] == labels_json]
        if filtered.empty:
            available_labels = (
                df[df["metric_name"] == metric_name]["labels_json"].unique().tolist()
            )
            raise ValueError(
                f"No data for metric '{metric_name}' with labels {labels_filter}. "
                f"Available label combinations: {available_labels[:5]}"
            )

    # Convert timestamp to relative seconds (from minimum timestamp)
    if not filtered.empty:
        filtered["timestamp_s"] = (
            filtered["timestamp_ns"] - filtered["timestamp_ns"].min()
        ) / 1e9

    # Extract metadata from first row
    unit = (
        filtered["unit"].iloc[0]
        if "unit" in filtered.columns and not filtered.empty
        else ""
    )
    metric_type = (
        filtered["metric_type"].iloc[0]
        if "metric_type" in filtered.columns and not filtered.empty
        else ""
    )

    return filtered, unit, metric_type


def detect_server_metric_series(df) -> list[tuple[str, str]]:
    """
    Detect unique endpoint+label combinations in server metrics DataFrame.

    Identifies all distinct series (endpoint + label combinations) for
    multi-series plotting. Each combination represents a separate time series
    that should be plotted as its own trace.

    Args:
        df: Filtered server metrics DataFrame

    Returns:
        List of (endpoint_url, labels_json) tuples, sorted for deterministic ordering

    Examples:
        >>> df = pd.DataFrame({
        ...     'endpoint_url': ['http://a', 'http://a', 'http://b'],
        ...     'labels_json': ['{"method":"GET"}', '{"method":"POST"}', '{}']
        ... })
        >>> detect_server_metric_series(df)
        [('http://a', '{"method":"GET"}'), ('http://a', '{"method":"POST"}'), ('http://b', '{}')]
    """
    if df.empty:
        return []

    # Get unique combinations and sort for deterministic ordering
    combinations = df[["endpoint_url", "labels_json"]].drop_duplicates()
    combinations = combinations.sort_values(by=["endpoint_url", "labels_json"])

    return [
        (row["endpoint_url"], row["labels_json"]) for _, row in combinations.iterrows()
    ]


def create_series_legend_label(
    metric_name: str,
    endpoint_url: str,
    labels_json: str,
    total_series: int,
    all_series_labels: list[dict] | None = None,
) -> str:
    """
    Create concise legend label showing only differentiating information.

    Generates short, readable legend entries by showing only the labels
    that actually vary between series. Drops common labels and uses
    compact formatting for clarity.

    Args:
        metric_name: Base metric name
        endpoint_url: Endpoint URL for this series
        labels_json: JSON-encoded labels for this series
        total_series: Total number of series (for label formatting decisions)
        all_series_labels: Optional list of all series label dicts for smart filtering

    Returns:
        Concise formatted legend label string

    Examples:
        >>> # Only component differs
        >>> create_series_legend_label("requests", "http://a", '{"component":"backend"}', 2,
        ...     all_series_labels=[{"component":"backend"}, {"component":"prefill"}])
        'backend'

        >>> # Component and endpoint differ
        >>> create_series_legend_label("requests", "http://a",
        ...     '{"component":"backend","endpoint":"generate"}', 4,
        ...     all_series_labels=[{"component":"backend","endpoint":"generate"},
        ...                        {"component":"backend","endpoint":"clear"},
        ...                        {"component":"prefill","endpoint":"generate"},
        ...                        {"component":"prefill","endpoint":"clear"}])
        'backend/generate'
    """
    # Parse labels
    labels_dict = orjson.loads(labels_json.encode()) if labels_json != "{}" else {}

    # Single series - just use metric name
    if total_series == 1:
        return metric_name

    # If we have all series labels, show only differentiating ones
    if all_series_labels and len(all_series_labels) > 1:
        # Find which label keys actually vary across series
        varying_keys = set()
        for key in labels_dict:
            values = set()
            for other_labels in all_series_labels:
                if key in other_labels:
                    values.add(other_labels[key])
            if len(values) > 1:  # This key has different values across series
                varying_keys.add(key)

        # Use only varying labels
        if varying_keys:
            # Special case: If only "engine" varies (common in vLLM histograms)
            if varying_keys == {"engine"}:
                return f"engine-{labels_dict['engine']}"

            # Special case: finished_reason (vLLM request completion)
            if varying_keys == {"finished_reason"} or (
                "finished_reason" in varying_keys and len(varying_keys) <= 2
            ):
                return labels_dict.get("finished_reason", "unknown")

            # Prioritize certain keys for cleaner display
            priority_keys = [
                "dynamo_component",
                "component",
                "dynamo_endpoint",
                "endpoint",
                "method",
                "status",
                "engine",
                "finished_reason",
            ]
            selected_keys = [k for k in priority_keys if k in varying_keys]
            # Add any remaining varying keys not in priority list
            selected_keys.extend(
                [k for k in sorted(varying_keys) if k not in selected_keys]
            )

            # Special formatting for engine
            if selected_keys and selected_keys[0] == "engine":
                return f"engine-{labels_dict['engine']}"

            # Build compact label (no key names, just values)
            values = [labels_dict[k] for k in selected_keys if k in labels_dict]
            return "/".join(values[:3])  # Max 3 components for readability

    # Fallback: show key labels in compact format
    if labels_dict:
        # Special handling for engine labels (vLLM histograms)
        if "engine" in labels_dict:
            return f"engine-{labels_dict['engine']}"

        # Try to use common meaningful keys
        if "dynamo_component" in labels_dict and "dynamo_endpoint" in labels_dict:
            return f"{labels_dict['dynamo_component']}/{labels_dict['dynamo_endpoint']}"
        elif "component" in labels_dict and "endpoint" in labels_dict:
            return f"{labels_dict['component']}/{labels_dict['endpoint']}"
        elif "method" in labels_dict:
            return labels_dict["method"]
        elif "finished_reason" in labels_dict:
            return labels_dict["finished_reason"]
        else:
            # Show first 2 label values
            values = list(labels_dict.values())[:2]
            return "/".join(values)

    # No labels but multiple series - show short endpoint
    endpoint_short = (
        endpoint_url.split("/")[-2] if "/" in endpoint_url else endpoint_url
    )
    endpoint_short = endpoint_short.replace(":9090", "").replace("proxy", "")
    return endpoint_short[:20]


def get_available_labels_for_metric(
    server_metrics_aggregated: dict,
    metric_name: str,
) -> dict[str, list[dict[str, str]]]:
    """
    Get all available label combinations for a server metric.

    Extracts unique label sets across all endpoints for a given metric.
    Useful for UI dropdowns and user guidance on available filtering options.

    Args:
        server_metrics_aggregated: Server metrics aggregated dict from RunData
        metric_name: Metric name to query

    Returns:
        Dict mapping endpoint_url to list of label dicts:
        {
            "http://localhost:8081/metrics": [
                {"method": "GET", "status": "200"},
                {"method": "POST", "status": "200"}
            ],
            ...
        }

    Examples:
        >>> labels = get_available_labels_for_metric(
        ...     server_metrics_aggregated,
        ...     "vllm:request_success"
        ... )
        >>> labels["http://localhost:8081/metrics"]
        [{"method": "GET"}, {"method": "POST"}]
    """
    if metric_name not in server_metrics_aggregated:
        return {}

    result = {}
    endpoint_data = server_metrics_aggregated[metric_name]

    for endpoint_url, labels_dict in endpoint_data.items():
        labels_list = []
        for labels_key in labels_dict:
            if labels_key != "{}":
                # Parse JSON to dict
                labels_obj = orjson.loads(labels_key.encode())
                labels_list.append(labels_obj)
            else:
                # Empty labels
                labels_list.append({})

        result[endpoint_url] = labels_list

    return result


def get_server_metrics_summary(run_data) -> dict[str, dict]:
    """
    Get comprehensive summary of server metrics with label information.

    Extends basic metric information with label cardinality and examples
    for better user understanding and UI presentation.

    Args:
        run_data: RunData object with server_metrics_aggregated

    Returns:
        Dict mapping metric_name → {
            "display_name": str,
            "unit": str,
            "type": str,
            "description": str,
            "endpoints": list[str],
            "label_combinations": int,  # Total unique label combos across endpoints
            "sample_labels": dict | None,  # Example label set
        }

    Examples:
        >>> summary = get_server_metrics_summary(run_data)
        >>> summary["vllm:request_success"]["label_combinations"]
        4  # 2 endpoints × 2 label combos each
    """
    if (
        not hasattr(run_data, "server_metrics_aggregated")
        or not run_data.server_metrics_aggregated
    ):
        return {}

    metrics = {}
    for metric_name, endpoint_data in run_data.server_metrics_aggregated.items():
        if not endpoint_data:
            continue

        # Get metadata from first series
        first_endpoint = next(iter(endpoint_data.values()))
        if not first_endpoint:
            continue

        first_series = next(iter(first_endpoint.values()))

        # Count total label combinations across all endpoints
        total_label_combos = sum(len(labels) for labels in endpoint_data.values())

        # Get sample labels (first non-empty label set)
        sample_labels = None
        for endpoint in endpoint_data.values():
            for labels_key in endpoint:
                if labels_key != "{}":
                    sample_labels = orjson.loads(labels_key.encode())
                    break
            if sample_labels:
                break

        metrics[metric_name] = {
            "display_name": _format_server_metric_name(metric_name),
            "unit": first_series.get("unit") or "",
            "type": first_series.get("type", ""),
            "description": first_series.get("description", ""),
            "endpoints": list(endpoint_data.keys()),
            "label_combinations": total_label_combos,
            "sample_labels": sample_labels,
        }

    return metrics
