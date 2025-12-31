/**
 * Cash Flow Visualization
 * Renders sankey and combo charts for cash flow engine results.
 *
 * @module nexus/visualizations/engines/financial/cashflow
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

/**
 * Build HTML section for Cash Flow visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string
 */
export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Cash Flow Analysis</h5>
            <div class="fin-grid fin-grid-2 fin-gap-md">
                <div class="fin-chart-container" id="sankey-${vizId}" style="height: 350px;"></div>
                <div class="fin-chart-container" id="cashflow-combo-${vizId}" style="height: 350px;"></div>
            </div>
            <div class="fin-chart-container" id="cashflow-gauge-${vizId}" style="height: 200px; margin-top: 1rem;"></div>
        </div>
    `;
}

// ============================================================================
// Chart Renderer
// ============================================================================

/**
 * Render Cash Flow visualization charts.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
export function render(data, vizId) {
    renderSankeyFlow(data, `sankey-${vizId}`);
    renderCashFlowCombo(data, `cashflow-combo-${vizId}`);
    renderCashFlowGauge(data, `cashflow-gauge-${vizId}`);
}

function renderSankeyFlow(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const flows = data?.flows || data?.sankey_data || [];

    if (!flows.length) {
        // Generate sample flow from summary data
        const inflow = data?.summary?.total_inflow || data?.total_inflow || 100000;
        const outflow = data?.summary?.total_outflow || data?.total_outflow || 80000;

        Plotly.newPlot(container, [{
            type: 'sankey',
            node: {
                label: ['Revenue', 'Cash', 'Operating', 'Capital', 'Net'],
                color: [VIZ_COLORS.success, VIZ_COLORS.primary, VIZ_COLORS.warning, VIZ_COLORS.error, VIZ_COLORS.accent]
            },
            link: {
                source: [0, 1, 1, 2],
                target: [1, 2, 3, 4],
                value: [inflow, outflow * 0.6, outflow * 0.4, inflow - outflow]
            }
        }], {
            paper_bgcolor: 'transparent',
            font: { color: VIZ_COLORS.textMuted, size: 11 },
            margin: { l: 10, r: 10, t: 30, b: 10 },
            title: { text: 'Cash Flow Diagram', font: { size: 12 } }
        }, { responsive: true, displayModeBar: false });
        return;
    }

    const nodes = [...new Set(flows.flatMap(f => [f.source, f.target]))];
    const nodeMap = Object.fromEntries(nodes.map((n, i) => [n, i]));

    Plotly.newPlot(container, [{
        type: 'sankey',
        node: {
            label: nodes,
            color: nodes.map((_, i) => `hsl(${(i * 50) % 360}, 70%, 50%)`)
        },
        link: {
            source: flows.map(f => nodeMap[f.source]),
            target: flows.map(f => nodeMap[f.target]),
            value: flows.map(f => f.value || f.amount || 0)
        }
    }], {
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 10, r: 10, t: 30, b: 10 },
        title: { text: 'Cash Flow Diagram', font: { size: 12 } }
    }, { responsive: true, displayModeBar: false });
}

function renderCashFlowCombo(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const periods = data?.periods || data?.dates || [];
    const inflows = data?.inflows || [];
    const outflows = data?.outflows || [];

    if (!periods.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No time series cash flow data</p>';
        return;
    }

    const cumulative = [];
    let running = 0;
    for (let i = 0; i < periods.length; i++) {
        running += (inflows[i] || 0) - (outflows[i] || 0);
        cumulative.push(running);
    }

    Plotly.newPlot(container, [
        {
            type: 'bar',
            x: periods,
            y: inflows,
            name: 'Inflows',
            marker: { color: VIZ_COLORS.success }
        },
        {
            type: 'bar',
            x: periods,
            y: outflows.map(v => -v),
            name: 'Outflows',
            marker: { color: VIZ_COLORS.error }
        },
        {
            type: 'scatter',
            mode: 'lines+markers',
            x: periods,
            y: cumulative,
            name: 'Net Position',
            yaxis: 'y2',
            line: { color: VIZ_COLORS.primary, width: 2 }
        }
    ], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 50, r: 50, t: 30, b: 40 },
        title: { text: 'Cash Flow Over Time', font: { size: 12 } },
        barmode: 'relative',
        xaxis: { gridcolor: VIZ_COLORS.border },
        yaxis: { title: 'Cash Flow', gridcolor: VIZ_COLORS.border },
        yaxis2: { overlaying: 'y', side: 'right', title: 'Cumulative' },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'transparent' }
    }, { responsive: true, displayModeBar: false });
}

function renderCashFlowGauge(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    // Try to calculate ratio from various API response structures
    let ratio = 0;

    if (data?.health_ratio !== undefined) {
        ratio = data.health_ratio;
    } else if (data?.liquidity_ratio !== undefined) {
        ratio = data.liquidity_ratio;
    } else {
        // Calculate from inflows/outflows (API returns these as objects)
        let totalInflow = 0;
        let totalOutflow = 0;

        if (data?.inflows && typeof data.inflows === 'object') {
            totalInflow = Object.values(data.inflows).reduce((sum, v) => sum + (Number(v) || 0), 0);
        } else if (data?.summary?.total_inflow) {
            totalInflow = data.summary.total_inflow;
        }

        if (data?.outflows && typeof data.outflows === 'object') {
            totalOutflow = Object.values(data.outflows).reduce((sum, v) => sum + (Number(v) || 0), 0);
        } else if (data?.summary?.total_outflow) {
            totalOutflow = data.summary.total_outflow;
        }

        // Fallback to summary fields
        if (!totalInflow && data?.summary?.net_cash_flow > 0) {
            totalInflow = data.summary.net_cash_flow;
        }
        if (!totalOutflow && data?.summary?.burn_rate) {
            totalOutflow = data.summary.burn_rate * 12; // Annualize monthly burn
        }

        ratio = totalOutflow > 0 ? totalInflow / totalOutflow : (totalInflow > 0 ? 2.5 : 0);
    }

    Plotly.newPlot(container, [{
        type: 'indicator',
        mode: 'gauge+number',
        value: ratio,
        number: { valueformat: '.2f' },
        gauge: {
            axis: { range: [0, 3] },
            bar: { color: ratio >= 1 ? VIZ_COLORS.success : VIZ_COLORS.error },
            steps: [
                { range: [0, 1], color: 'rgba(239, 68, 68, 0.2)' },
                { range: [1, 2], color: 'rgba(251, 191, 36, 0.2)' },
                { range: [2, 3], color: 'rgba(16, 185, 129, 0.2)' }
            ]
        }
    }], {
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 30, r: 30, t: 30, b: 10 },
        title: { text: 'Cash Flow Health Ratio', font: { size: 12 } }
    }, { responsive: true, displayModeBar: false });
}
