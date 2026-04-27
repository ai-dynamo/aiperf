import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { palette } from '../lib/theme.js';
import { ChartWrapper } from './chart-wrapper.js';

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
 * 2D dimension: small-multiples — one chart series per second-dim value.
 * 3+ D:        renders a single chart over the FIRST dimension and a
 *              note instructing to use the table view.
 */
const SERIES_COLORS = [
  palette.blue,
  palette.peach,
  palette.green,
  palette.mauve,
  palette.teal,
  palette.lavender,
  palette.red,
  palette.sapphire,
];

export function CellsChart({ dimensions, cells, metric, stat }) {
  const { data, options, hasData } = useMemo(() => {
    if (!dimensions || dimensions.length === 0 || !cells || cells.length === 0) {
      return { data: null, options: null, hasData: false };
    }

    const primaryDim = dimensions[0];
    const xValues = primaryDim.values;
    const datasets = [];

    if (dimensions.length <= 1) {
      const ys = xValues.map(v => {
        const cell = cells.find(c => (c.values?.[primaryDim.name] === v));
        return cell?.metrics?.[metric]?.[stat] ?? null;
      });
      const c = SERIES_COLORS[0];
      datasets.push({
        label: `${metric} (${stat})`,
        data: ys,
        borderColor: c,
        backgroundColor: c + '22',
        pointBackgroundColor: c,
        pointBorderColor: c,
        pointRadius: 4,
        tension: 0.1,
        spanGaps: true,
      });
    } else {
      const secondDim = dimensions[1];
      secondDim.values.forEach((sv, idx) => {
        const ys = xValues.map(xv => {
          const cell = cells.find(cc =>
            cc.values?.[primaryDim.name] === xv &&
            cc.values?.[secondDim.name] === sv
          );
          return cell?.metrics?.[metric]?.[stat] ?? null;
        });
        const c = SERIES_COLORS[idx % SERIES_COLORS.length];
        datasets.push({
          label: `${secondDim.name}=${sv}`,
          data: ys,
          borderColor: c,
          backgroundColor: c + '22',
          pointBackgroundColor: c,
          pointBorderColor: c,
          pointRadius: 3,
          tension: 0.1,
          spanGaps: true,
        });
      });
    }

    const chartData = {
      labels: xValues.map(String),
      datasets,
    };
    const chartOptions = {
      plugins: {
        legend: {
          display: datasets.length > 1,
          labels: { color: palette.text, font: { size: 11 } },
        },
        tooltip: {
          backgroundColor: palette.mantle,
          titleColor: palette.text,
          bodyColor: palette.text,
          borderColor: palette.surface0,
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          title: { display: true, text: primaryDim.name, color: palette.overlay1, font: { size: 11 } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: 10 } },
        },
        y: {
          title: { display: true, text: `${metric} (${stat})`, color: palette.overlay1, font: { size: 11 } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: 10 } },
        },
      },
    };
    return { data: chartData, options: chartOptions, hasData: true };
  }, [dimensions, cells, metric, stat]);

  if (!dimensions || dimensions.length === 0) {
    return html`<div data-testid="sweep-cells-chart" class="text-dim" style="padding:var(--space-3) 0">
      No swept dimensions in this sweep.
    </div>`;
  }
  if (!cells || cells.length === 0 || !hasData) {
    return html`<div data-testid="sweep-cells-chart" class="text-dim" style="padding:var(--space-3) 0">
      No cells completed yet.
    </div>`;
  }

  return html`
    <div data-testid="sweep-cells-chart">
      <${ChartWrapper} type="line" data=${data} options=${options} height=${360} />
      ${dimensions.length >= 3 && html`
        <p class="text-dim" style="margin-top:var(--space-2);font-size:var(--font-size-sm)">
          ${dimensions.length}-D sweep — chart shows the first dimension only.
          Use the table view to inspect higher-dim cells.
        </p>
      `}
    </div>
  `;
}
