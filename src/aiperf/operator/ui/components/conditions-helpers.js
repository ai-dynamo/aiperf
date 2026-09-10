// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const CONDITION_LABELS = {
  ConfigValid: 'Config',
  EndpointReachable: 'Endpoint',
  PreflightPassed: 'Preflight',
  ResourcesCreated: 'Resources',
  WorkersReady: 'Workers',
  BenchmarkRunning: 'Running',
  ResultsAvailable: 'Results',
};

export function conditionClass(condition) {
  const status = (condition.status ?? '').toLowerCase();
  const reason = (condition.reason ?? '').toLowerCase();
  const type = (condition.type ?? '').toLowerCase();

  if (type.includes('warning') || reason.includes('warning')) {
    return 'condition-badge--progress';
  }
  if (status === 'true') return 'condition-badge--true';
  if (reason.includes('progress') || reason.includes('waiting')) {
    return 'condition-badge--progress';
  }
  if (status === 'false') return 'condition-badge--false';
  return 'condition-badge--unknown';
}

export function conditionLabel(type) {
  if (CONDITION_LABELS[type]) return CONDITION_LABELS[type];
  if (type === 'PreflightHasWarnings') return 'Preflight warnings';
  return type
    .replace(/Has/g, ' ')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .trim();
}

function conditionBadge(condition) {
  return {
    type: condition.type,
    message: condition.message,
    label: conditionLabel(condition.type),
    className: conditionClass(condition),
  };
}

function falseTerminalConditionTypes(conditions) {
  const types = new Set();

  for (const condition of conditions ?? []) {
    if ((condition.status ?? '').toLowerCase() !== 'false') continue;
    if (condition.type === 'Complete' || condition.type === 'Failed') types.add(condition.type);
  }

  return types;
}

function shouldHideCondition(condition, terminalFalseTypes) {
  const status = (condition.status ?? '').toLowerCase();
  if (status === 'true') return true;

  return terminalFalseTypes.size === 2 && terminalFalseTypes.has(condition.type);
}

export function visibleConditionBadges(conditions) {
  const terminalFalseTypes = falseTerminalConditionTypes(conditions);
  return (conditions ?? [])
    .filter(condition => !shouldHideCondition(condition, terminalFalseTypes))
    .map(conditionBadge);
}

export function visibleConditionBadgeSummary(conditions, limit) {
  const badges = [];
  let total = 0;
  const terminalFalseTypes = falseTerminalConditionTypes(conditions);

  for (const condition of conditions ?? []) {
    if (shouldHideCondition(condition, terminalFalseTypes)) continue;
    total += 1;
    if (badges.length < limit) badges.push(conditionBadge(condition));
  }

  return { badges, overflow: Math.max(0, total - badges.length) };
}
