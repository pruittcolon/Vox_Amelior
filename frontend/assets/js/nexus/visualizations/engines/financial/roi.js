/**
 * ROI Prediction Visualization
 * Renders tornado and gauge charts for ROI engine results.
 *
 * @module nexus/visualizations/engines/financial/roi
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

/**
 * Build HTML section for ROI visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string
 */
export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>ROI Prediction</h5>
            <div class="fin-grid fin-grid-2 fin-gap-md">
                <div class="fin-chart-container" id="roi-tornado-${vizId}" style="height: 350px;"></div>
                <div class="fin-chart-container" id="roi-gauge-${vizId}" style="height: 250px;"></div>
            </div>
        </div>
    `;
}

// ============================================================================
// Chart Renderer
// ============================================================================

/**
 * Render ROI visualization charts.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
export function render(data, vizId) {
    renderROITornado(data, `roi-tornado-${vizId}`);
    renderROIGauge(data, `roi-gauge-${vizId}`);
}

function renderROITornado(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const factors = data?.sensitivity || data?.factors || data?.drivers || [];

    if (!factors.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No sensitivity data available</p>';
        return;
    }

    const sorted = [...factors].sort((a, b) => Math.abs(b.impact || b.value || 0) - Math.abs(a.impact || a.value || 0));
    const labels = sorted.map(f => f.name || f.factor);
    const positive = sorted.map(f => Math.max(0, f.impact || f.value || 0));
    const negative = sorted.map(f => Math.min(0, f.impact || f.value || 0));

    Plotly.newPlot(container, [
        {
            type: 'bar',
            orientation: 'h',
            y: labels,
            x: positive,
            name: 'Positive',
            marker: { color: VIZ_COLORS.success }
        },
        {
            type: 'bar',
            orientation: 'h',
            y: labels,
            x: negative,
            name: 'Negative',
            marker: { color: VIZ_COLORS.error }
        }
    ], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 120, r: 20, t: 30, b: 40 },
        title: { text: 'Sensitivity Analysis', font: { size: 12 } },
        barmode: 'relative',
        xaxis: { title: 'Impact on ROI (%)', gridcolor: VIZ_COLORS.border, zeroline: true },
        yaxis: { gridcolor: VIZ_COLORS.border },
        showlegend: false
    }, { responsive: true, displayModeBar: false });
}

function renderROIGauge(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const roi = data?.roi ?? data?.predicted_roi ?? data?.value ?? 0;
    const roiPercent = roi * 100;

    Plotly.newPlot(container, [{
        type: 'indicator',
        mode: 'gauge+number+delta',
        value: roiPercent,
        number: { suffix: '%', font: { size: 24 } },
        delta: { reference: data?.baseline_roi ? data.baseline_roi * 100 : 0 },
        gauge: {
            axis: { range: [-50, 200], ticksuffix: '%' },
            bar: { color: VIZ_COLORS.primary },
            bgcolor: VIZ_COLORS.surface,
            borderwidth: 1,
            bordercolor: VIZ_COLORS.border,
            steps: [
                { range: [-50, 0], color: 'rgba(239, 68, 68, 0.2)' },
                { range: [0, 50], color: 'rgba(251, 191, 36, 0.2)' },
                { range: [50, 200], color: 'rgba(16, 185, 129, 0.2)' }
            ],
            threshold: {
                line: { color: VIZ_COLORS.error, width: 3 },
                thickness: 0.75,
                value: data?.target_roi ? data.target_roi * 100 : 100
            }
        }
    }], {
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 20, r: 20, t: 30, b: 10 },
        title: { text: 'Predicted ROI', font: { size: 12 } }
    }, { responsive: true, displayModeBar: false });
}
