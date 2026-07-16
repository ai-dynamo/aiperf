// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pure helpers for the Compare page job-picker filter chips. Kept in a
// sibling module (no preact/htm imports) so they can be unit-tested via
// raw Node — see tests/unit/ui/test_operator_compare_filters.py.

// Sentinel key for the "(none)" facet bucket — null model/endpoint values
// land here so users can still filter to/from missing-metadata jobs.
export const FILTER_NONE = '__none__';

// Pure filter — exported for unit tests at tests/unit/ui/test_operator_compare_filters.py.
// AND across (nsFilter, modelFilter, endpointFilter, search); OR within each Set.
// Empty Set on a dimension means "no filter on this dimension".
export function applyJobFilters(jobs, { nsFilter, modelFilter, endpointFilter, search }) {
  const q = (search || '').toLowerCase();
  return jobs.filter((job) => {
    if (nsFilter && nsFilter.size && !nsFilter.has(job.namespace ?? FILTER_NONE)) return false;
    if (modelFilter && modelFilter.size && !modelFilter.has(job.model ?? FILTER_NONE)) return false;
    if (endpointFilter && endpointFilter.size && !endpointFilter.has(job.endpoint ?? FILTER_NONE)) return false;
    if (!q) return true;
    return (
      (job.job_id ?? '').toLowerCase().includes(q) ||
      (job.namespace ?? '').toLowerCase().includes(q) ||
      (job.model ?? '').toLowerCase().includes(q) ||
      (job.endpoint ?? '').toLowerCase().includes(q)
    );
  });
}

// Distinct-value counts per chip-filter dimension. Returns three Maps keyed
// by value (or FILTER_NONE for nulls) -> absolute job count.
export function extractFacets(jobs) {
  const ns = new Map();
  const model = new Map();
  const endpoint = new Map();
  const bump = (m, k) => m.set(k, (m.get(k) ?? 0) + 1);
  for (const j of jobs) {
    bump(ns, j.namespace ?? FILTER_NONE);
    bump(model, j.model ?? FILTER_NONE);
    bump(endpoint, j.endpoint ?? FILTER_NONE);
  }
  return { ns, model, endpoint };
}

// Cross-filtered facets: each dimension's chip set is computed from jobs
// filtered by every OTHER dimension's active filter (and the search box),
// so clicking a chip narrows the remaining dimensions' chips to only what
// can still co-occur with the selection. The dimension's OWN selected
// chips are preserved (filterSet is not applied to its own dimension).
//
// Empty Sets behave the same as `applyJobFilters`: empty = no filter on
// that dimension. Search applies to every dimension's facet projection.
export function extractCrossFacets(jobs, { nsFilter, modelFilter, endpointFilter, search }) {
  const filterByOthers = (skip) => applyJobFilters(jobs, {
    nsFilter: skip === 'ns' ? new Set() : nsFilter,
    modelFilter: skip === 'model' ? new Set() : modelFilter,
    endpointFilter: skip === 'endpoint' ? new Set() : endpointFilter,
    search,
  });
  return {
    ns: extractFacets(filterByOthers('ns')).ns,
    model: extractFacets(filterByOthers('model')).model,
    endpoint: extractFacets(filterByOthers('endpoint')).endpoint,
  };
}
