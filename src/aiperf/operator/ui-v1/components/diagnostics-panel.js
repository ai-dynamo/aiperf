// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tabbed diagnostics panel. Consolidates Events / Logs / Conditions / Pods
 * into one Panel.  Active tab is URL-backed via ``?diag=<id>`` query param.
 *
 * Tab availability depends on mode + archived flag:
 *   - mode='live'     : all four tabs
 *   - mode='completed': all four tabs (frozen)
 *   - archived=true   : Events + Conditions only (logs/pods irrelevant; pod CRs are gone)
 *
 * Default tab: Events for live, Conditions for archived/completed.
 */

import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { Panel } from './panel.js';
import { EventsTab } from './diagnostics-events-tab.js';
import { LogsTab } from './diagnostics-logs-tab.js';
import { ConditionsTab } from './diagnostics-conditions-tab.js';
import { PodsTab } from './diagnostics-pods-tab.js';

const ALL_TABS = ['events', 'logs', 'conditions', 'pods'];

function readTabFromUrl() {
  const url = new URL(window.location.href);
  const t = url.searchParams.get('diag');
  return ALL_TABS.includes(t) ? t : null;
}

function writeTabToUrl(tab) {
  const url = new URL(window.location.href);
  url.searchParams.set('diag', tab);
  window.history.replaceState(null, '', url.toString());
}

export function DiagnosticsPanel({
  ns, name, conditions, pods, mode, archived,
  eventCount, logSeverityCounts, conditionWarnCount, podCrashCount,
}) {
  const availableTabs = archived ? ['events', 'conditions'] : ALL_TABS;
  const defaultTab = (mode === 'live' && !archived) ? 'events' : 'conditions';
  const [active, setActive] = useState(() => readTabFromUrl() ?? defaultTab);

  useEffect(() => {
    if (!availableTabs.includes(active)) {
      setActive(availableTabs[0]);
    }
  }, [archived, mode]);

  useEffect(() => {
    function onPopState() {
      const fromUrl = readTabFromUrl();
      if (fromUrl && availableTabs.includes(fromUrl) && fromUrl !== active) {
        setActive(fromUrl);
      }
    }
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, [availableTabs, active]);

  const switchTo = (tab) => {
    setActive(tab);
    writeTabToUrl(tab);
  };

  const badgeWarn = (conditionWarnCount > 0 || podCrashCount > 0) ? (conditionWarnCount + podCrashCount) : null;

  return html`
    <${Panel} title="diagnostics" testId="panel-diagnostics"
              badge=${badgeWarn} badgeTone=${badgeWarn ? 'warn' : null}>
      <div class="diag-tabs" role="tablist">
        ${availableTabs.map((tab) => {
          const count = tab === 'events' ? eventCount
                      : tab === 'logs' ? null
                      : tab === 'conditions' ? (conditions?.length ?? null)
                      : (pods?.length ?? null);
          return html`
            <span class=${'diag-tab' + (active === tab ? ' diag-tab--active' : '')}
                  data-tab-id=${tab}
                  role="tab"
                  aria-selected=${active === tab}
                  onClick=${() => switchTo(tab)}
                  key=${tab}>
              ${tab}
              ${count != null && html`<span class="diag-tab-count">${count}</span>`}
            </span>
          `;
        })}
      </div>
      ${active === 'events' && html`<${EventsTab} ns=${ns} name=${name} active=${true} />`}
      ${active === 'logs' && html`<${LogsTab} ns=${ns} name=${name} pods=${pods} active=${true} />`}
      ${active === 'conditions' && html`<${ConditionsTab} conditions=${conditions} />`}
      ${active === 'pods' && html`<${PodsTab} pods=${pods} />`}
    <//>
  `;
}
