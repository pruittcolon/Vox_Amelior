/**
 * Cost Optimization Visualization
 * Renders treemap and pareto charts for cost engine results.
 *
 * @module nexus/visualizations/engines/financial/cost
 */

import { VIZ_COLORS, escapeHtml } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

/**
 * Build HTML section for Cost visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string
 */
export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Cost Optimization</h5>
            <div class="fin-grid fin-grid-2 fin-gap-md">
                <div class="fin-chart-container" id="cost-treemap-${vizId}" style="height: 350px;"></div>
                <div class="fin-chart-container" id="cost-pareto-${vizId}" style="height: 350px;"></div>
            </div>
        </div>
    `;
}

// ============================================================================
// Chart Renderer
// ============================================================================

/**
 * Render Cost visualization charts.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
export function render(data, vizId) {
    renderCostTreemap(data, `cost-treemap-${vizId}`);
    renderCostPareto(data, `cost-pareto-${vizId}`);
}

function renderCostTreemap(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    // Backend returns cost_categories as dict {name: value}, convert to array
    let categories = [];

    // Option 1: cost_categories dict format from backend
    if (data?.cost_categories && typeof data.cost_categories === 'object' && !Array.isArray(data.cost_categories)) {
        categories = Object.entries(data.cost_categories)
            .filter(([_, val]) => val > 0) // Filter out zero values
            .map(([name, value]) => ({ name, value }));
    }
    // Option 2: Array format (legacy)
    else if (Array.isArray(data?.cost_breakdown)) {
        categories = data.cost_breakdown;
    }
    else if (Array.isArray(data?.categories)) {
        categories = data.categories;
    }

    if (!categories.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No cost breakdown data available</p>';
        return;
    }

    const labels = categories.map(c => c.name || c.category);
    const values = categories.map(c => c.value || c.cost || c.amount || 0);
    const parents = categories.map(() => '');

    Plotly.newPlot(container, [{
        type: 'treemap',
        labels,
        values,
        parents,
        marker: {
            colors: values.map((_, i) => `hsl(${200 + i * 15}, 70%, ${50 + i * 3}%)`),
        },
        textinfo: 'label+percent entry',
        hovertemplate: '<b>%{label}</b><br>Cost: $%{value:,.0f}<br>%{percentEntry:.1%}<extra></extra>'
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 10, r: 10, t: 30, b: 10 },
        title: { text: 'Cost Breakdown', font: { size: 12 } }
    }, { responsive: true, displayModeBar: false });
}


function renderCostPareto(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    let labels = [];
    let values = [];
    let cumulative = [];

    // Option 1: Backend pareto.graph format with x_data, y_data, cumulative
    if (data?.pareto?.graph?.x_data && data?.pareto?.graph?.y_data) {
        labels = data.pareto.graph.x_data;
        values = data.pareto.graph.y_data;
        cumulative = data.pareto.graph.cumulative || [];
    }
    // Option 2: cost_categories dict - build pareto from it
    else if (data?.cost_categories && typeof data.cost_categories === 'object') {
        const entries = Object.entries(data.cost_categories)
            .filter(([_, val]) => val > 0)
            .sort((a, b) => b[1] - a[1]); // Sort desc by value

        labels = entries.map(([name]) => name);
        values = entries.map(([_, val]) => val);

        // Calculate cumulative
        const total = values.reduce((a, b) => a + b, 0);
        let cumSum = 0;
        cumulative = values.map(v => {
            cumSum += v;
            return (cumSum / total) * 100;
        });
    }
    // Option 3: Legacy array format
    else if (Array.isArray(data?.pareto_data) && data.pareto_data.length) {
        const sorted = [...data.pareto_data].sort((a, b) => (b.value || b.cost || 0) - (a.value || a.cost || 0));
        labels = sorted.map(i => i.name || i.category);
        values = sorted.map(i => i.value || i.cost || 0);

        const total = values.reduce((a, b) => a + b, 0);
        let cumSum = 0;
        cumulative = values.map(v => {
            cumSum += v;
            return (cumSum / total) * 100;
        });
    }

    if (!labels.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No pareto data available</p>';
        return;
    }

    Plotly.newPlot(container, [
        {
            type: 'bar',
            x: labels,
            y: values,
            name: 'Cost',
            marker: { color: VIZ_COLORS.primaryLight }
        },
        {
            type: 'scatter',
            mode: 'lines+markers',
            x: labels,
            y: cumulative,
            name: 'Cumulative %',
            yaxis: 'y2',
            line: { color: VIZ_COLORS.error, width: 2 },
            marker: { size: 6 }
        }
    ], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 50, r: 50, t: 30, b: 60 },
        title: { text: 'Cost Pareto Analysis', font: { size: 12 } },
        xaxis: { tickangle: -30, gridcolor: VIZ_COLORS.border },
        yaxis: { title: 'Cost ($)', gridcolor: VIZ_COLORS.border },
        yaxis2: { overlaying: 'y', side: 'right', title: 'Cumulative %', range: [0, 100] },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'transparent' }
    }, { responsive: true, displayModeBar: false });
}

