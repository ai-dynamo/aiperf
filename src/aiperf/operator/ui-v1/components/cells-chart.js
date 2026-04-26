import { html } from 'htm/preact';
import { useEffect, useRef } from 'preact/hooks';
import { palette } from '../lib/theme.js';

/**
 * Per-cell metric chart for a sweep.
 *
 * Props:
 *   dimensions: [{ name, values }]   from /sweeps/:ns/:name/cells
 *   cells:      [CellEntry]
 *   metric:     string               e.g. 'request_throughput'
 *   stat:       string               e.g. 'avg' | 'p99'
 *
 * 1D dimension: line chart, x = dim values, y = chosen metric stat.
 * 2D dimension: small-multiples — one chart element per second-dim value.
 * 3+ D:        renders a single chart over the FIRST dimension and a
 *              note instructing to use the table view.
 */
export function CellsChart({ dimensions, cells, metric, stat }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || typeof Chart === 'undefined') return;
    if (!dimensions || dimensions.length === 0 || !cells || cells.length === 0) return;

    const primaryDim = dimensions[0];
    const xValues = primaryDim.values;

    // Build series: if 1D, one series; if 2D+, one series per second-dim value
    const datasets = [];
    if (dimensions.length <= 1) {
      const data = xValues.map(v => {
        const cell = cells.find(c => (c.values?.[primaryDim.name] === v));
        return cell?.metrics?.[metric]?.[stat] ?? null;
      });
      datasets.push({
        label: `${metric} (${stat})`,
        data,
        borderColor: palette.blue ?? '#4ea1ff',
        backgroundColor: 'transparent',
        spanGaps: true,
      });
    } else {
      const secondDim = dimensions[1];
      for (const sv of secondDim.values) {
        const data = xValues.map(xv => {
          const cell = cells.find(c =>
            c.values?.[primaryDim.name] === xv &&
            c.values?.[secondDim.name] === sv
          );
          return cell?.metrics?.[metric]?.[stat] ?? null;
        });
        datasets.push({
          label: `${secondDim.name}=${sv}`,
          data,
          spanGaps: true,
        });
      }
    }

    if (chartRef.current) chartRef.current.destroy();
    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: { labels: xValues.map(String), datasets },
      options: {
        responsive: true,
        plugins: { legend: { display: datasets.length > 1 } },
        scales: {
          x: { title: { display: true, text: primaryDim.name } },
          y: { title: { display: true, text: `${metric} (${stat})` } },
        },
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [dimensions, cells, metric, stat]);

  if (!dimensions || dimensions.length === 0) {
    return html`<div data-testid="sweep-cells-chart" class="text-dim">
      No swept dimensions in this sweep.
    </div>`;
  }
  if (!cells || cells.length === 0) {
    return html`<div data-testid="sweep-cells-chart" class="text-dim">
      No cells completed yet.
    </div>`;
  }
  return html`
    <div data-testid="sweep-cells-chart">
      <canvas ref=${canvasRef} style="max-height: 360px"></canvas>
      ${dimensions.length >= 3 && html`
        <p class="text-dim" style="margin-top: var(--space-2); font-size: var(--font-size-sm)">
          ${dimensions.length}-D sweep — chart shows the first dimension only.
          Use the table view to inspect higher-dim cells.
        </p>
      `}
    </div>
  `;
}
