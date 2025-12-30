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

    const utilization = (data?.utilization || data?.average_utilization || data?.efficiency || 0) * 100;

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
