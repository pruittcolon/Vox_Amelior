/**
 * Profit Margin Visualization
 * Renders gauge and treemap charts for profit margin engine results.
 *
 * @module nexus/visualizations/engines/financial/margin
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Profit Margin Analysis</h5>
            <div class="fin-grid fin-grid-2 fin-gap-md">
                <div class="fin-chart-container" id="margin-gauge-${vizId}" style="height: 250px;"></div>
                <div class="fin-chart-container" id="margin-treemap-${vizId}" style="height: 350px;"></div>
            </div>
        </div>
    `;
}

export function render(data, vizId) {
    renderMarginGauge(data, `margin-gauge-${vizId}`);
    renderMarginTreemap(data, `margin-treemap-${vizId}`);
}

function renderMarginGauge(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const margin = (data?.profit_margin ?? data?.margin ?? data?.net_margin ?? 0) * 100;

    Plotly.newPlot(container, [{
        type: 'indicator',
        mode: 'gauge+number',
        value: margin,
        number: { suffix: '%', font: { size: 24 } },
        gauge: {
            axis: { range: [-20, 50] },
            bar: { color: margin >= 0 ? VIZ_COLORS.success : VIZ_COLORS.error },
            steps: [
                { range: [-20, 0], color: 'rgba(239, 68, 68, 0.2)' },
                { range: [0, 15], color: 'rgba(251, 191, 36, 0.2)' },
                { range: [15, 50], color: 'rgba(16, 185, 129, 0.2)' }
            ]
        }
    }], {
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 20, r: 20, t: 30, b: 10 },
        title: { text: 'Net Profit Margin', font: { size: 12 } }
    }, { responsive: true, displayModeBar: false });
}

function renderMarginTreemap(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const breakdown = data?.margin_breakdown || data?.breakdown || [];

    if (!breakdown.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No margin breakdown available</p>';
        return;
    }

    Plotly.newPlot(container, [{
        type: 'treemap',
        labels: breakdown.map(b => b.name || b.category),
        values: breakdown.map(b => Math.abs(b.value || b.amount || 0)),
        parents: breakdown.map(() => ''),
        marker: {
            colors: breakdown.map(b => (b.value || 0) >= 0 ? VIZ_COLORS.success : VIZ_COLORS.error)
        },
        textinfo: 'label+percent entry'
    }], {
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 10, r: 10, t: 30, b: 10 },
        title: { text: 'Profit Breakdown', font: { size: 12 } }
    }, { responsive: true, displayModeBar: false });
}
