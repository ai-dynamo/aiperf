// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const COMPLETED_PHASES = new Set(['completed', 'succeeded']);
const CANCELLED_PHASES = new Set(['cancelled', 'canceled']);
const TERMINAL_PHASES = new Set([
  ...COMPLETED_PHASES,
  ...CANCELLED_PHASES,
  'failed',
  'error',
  'partiallyfailed',
  'archived',
]);

export function deriveJobRunState({ phase, epoch, runEpoch }) {
  const phaseLower = (phase ?? 'Unknown').toLowerCase();
  const selectedRunEpoch = epoch != null ? String(epoch) : null;
  const liveRunEpoch = runEpoch != null ? String(runEpoch) : null;
  const viewingCurrentRun = epoch === undefined
    || (liveRunEpoch != null && selectedRunEpoch === liveRunEpoch);

  const isTerminal = TERMINAL_PHASES.has(phaseLower);
  return {
    phaseLower,
    isRunning: phaseLower === 'running',
    isCompleted: COMPLETED_PHASES.has(phaseLower),
    isCancelled: CANCELLED_PHASES.has(phaseLower),
    isPartiallyFailed: phaseLower === 'partiallyfailed',
    isArchived: phaseLower === 'archived',
    isTerminal,
    viewingCurrentRun,
    pollingDone: isTerminal,
    showLiveRunPanels: !isTerminal,
  };
}
