/**
 * Resource Utilization Visualization
 * Renders gauge chart for resource utilization engine results.
 *
 * @module nexus/visualizations/engines/financial/resource
 */

import { VIZ_COLORS, formatNumber } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Resource Utilization</h5>
            <div class="fin-chart-container" id="resource-gauge-${vizId}" style="height: 250px;"></div>
        </div>
    `;
}

export function render(data, vizId) {
    renderResourceGauge(data, `resource-gauge-${vizId}`);
}

function renderResourceGauge(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    // Check if this is a fallback response (engine couldn't find required columns)
    if (data?.fallback_used || data?.summary?.status === 'fallback_summary') {
        const reason = data?.summary?.reason || 'Resource utilization analysis requires columns like resource_id, usage, capacity.';
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: #94a3b8;">
                <p style="margin-bottom: 0.5rem; font-weight: 500;">Data Requirements Not Met</p>
                <p style="font-size: 0.85rem; color: #64748b;">${reason}</p>
            </div>
        `;
        return;
    }

    // Extract utilization from various possible API response structures
    let utilization = 0;

    // Check summary.avg_utilization (actual API response)
    if (data?.summary?.avg_utilization !== undefined) {
        utilization = data.summary.avg_utilization;
    } else if (data?.utilization !== undefined) {
        utilization = data.utilization * 100;
    } else if (data?.average_utilization !== undefined) {
        utilization = data.average_utilization * 100;
    } else if (data?.efficiency !== undefined) {
        utilization = data.efficiency * 100;
    } else if (data?.summary?.peak_utilization !== undefined) {
        // Fallback to peak if avg not available
        utilization = data.summary.peak_utilization;
    }

    // Handle both decimal (0-1) and percentage (0-100) formats
    if (utilization > 0 && utilization < 1) {
        utilization = utilization * 100;
    }


    Plotly.newPlot(container, [{
        type: 'indicator',
        mode: 'gauge+number',
        value: utilization,
        number: { suffix: '%', font: { size: 24 } },
        gauge: {
            axis: { range: [0, 100] },
            bar: { color: utilization > 80 ? VIZ_COLORS.error : utilization > 60 ? VIZ_COLORS.warning : VIZ_COLORS.success },
            bgcolor: VIZ_COLORS.surface,
            steps: [
                { range: [0, 60], color: 'rgba(16, 185, 129, 0.2)' },
                { range: [60, 80], color: 'rgba(251, 191, 36, 0.2)' },
                { range: [80, 100], color: 'rgba(239, 68, 68, 0.2)' }
            ],
            threshold: {
                line: { color: VIZ_COLORS.primary, width: 3 },
                thickness: 0.75,
                value: data?.target_utilization ? data.target_utilization * 100 : 75
            }
        }
    }], {
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 20, r: 20, t: 30, b: 10 },
        title: { text: 'Average Resource Utilization', font: { size: 12 } }
    }, { responsive: true, displayModeBar: false });
}

