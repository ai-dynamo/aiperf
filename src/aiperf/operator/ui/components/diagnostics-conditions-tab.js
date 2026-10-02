// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Conditions tab for DiagnosticsPanel — thin wrapper that reuses the
 * existing ``components/conditions.js`` component (which sweep-detail
 * also uses, so it is retained at its original path). No data fetching;
 * conditions are passed in by the parent page from the live status.
 */

import { html } from 'htm/preact';
import { Conditions } from './conditions.js';

export function ConditionsTab({ conditions }) {
  return html`
    <div class="diag-tab-body diag-tab-body--conditions">
      <${Conditions} conditions=${conditions} />
    </div>
  `;
}
