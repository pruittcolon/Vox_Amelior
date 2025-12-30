/**
 * Spend Patterns Visualization
 * Renders sankey and trend charts for spend pattern engine results.
 *
 * @module nexus/visualizations/engines/financial/spend
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Spend Pattern Analysis</h5>
            <div class="fin-grid fin-grid-2 fin-gap-md">
                <div class="fin-chart-container" id="spend-sankey-${vizId}" style="height: 350px;"></div>
                <div class="fin-chart-container" id="spend-trend-${vizId}" style="height: 350px;"></div>
            </div>
        </div>
    `;
}

export function render(data, vizId) {
    renderSpendSankey(data, `spend-sankey-${vizId}`);
    renderSpendTrend(data, `spend-trend-${vizId}`);
}

function renderSpendSankey(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const categories = data?.categories || data?.spend_categories || [];

    if (!categories.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No spend category data</p>';
        return;
    }

    const nodes = ['Total Spend', ...categories.map(c => c.name || c.category)];
    const nodeMap = Object.fromEntries(nodes.map((n, i) => [n, i]));

    Plotly.newPlot(container, [{
        type: 'sankey',
        node: {
            label: nodes,
            color: nodes.map((_, i) => i === 0 ? VIZ_COLORS.primary : `hsl(${i * 40}, 70%, 50%)`)
        },
        link: {
            source: categories.map(() => 0),
            target: categories.map((_, i) => i + 1),
            value: categories.map(c => c.amount || c.value || 0)
        }
    }], {
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 10, r: 10, t: 30, b: 10 },
        title: { text: 'Spend Distribution', font: { size: 12 } }
    }, { responsive: true, displayModeBar: false });
}

function renderSpendTrend(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const periods = data?.periods || data?.dates || [];
    const spending = data?.spending || data?.values || [];

    if (!periods.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No trend data</p>';
        return;
    }

    Plotly.newPlot(container, [{
        type: 'scatter',
        mode: 'lines+markers',
        x: periods,
        y: spending,
        fill: 'tozeroy',
        fillcolor: 'rgba(2, 85, 158, 0.1)',
        line: { color: VIZ_COLORS.primary, width: 2 },
        marker: { size: 6 }
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 50, r: 20, t: 30, b: 40 },
        title: { text: 'Spending Trend', font: { size: 12 } },
        xaxis: { gridcolor: VIZ_COLORS.border },
        yaxis: { title: 'Spend ($)', gridcolor: VIZ_COLORS.border }
    }, { responsive: true, displayModeBar: false });
}
