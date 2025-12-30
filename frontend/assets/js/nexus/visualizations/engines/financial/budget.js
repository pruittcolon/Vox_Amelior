/**
 * Budget Variance Visualization
 * Renders waterfall and bullet charts for budget variance engine results.
 *
 * @module nexus/visualizations/engines/financial/budget
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Budget Variance Analysis</h5>
            <div class="fin-grid fin-grid-2 fin-gap-md">
                <div class="fin-chart-container" id="waterfall-${vizId}" style="height: 350px;"></div>
                <div class="fin-chart-container" id="bullet-${vizId}" style="height: 350px;"></div>
            </div>
        </div>
    `;
}

export function render(data, vizId) {
    renderBudgetWaterfall(data, `waterfall-${vizId}`);
    renderBudgetBullet(data, `bullet-${vizId}`);
}

function renderBudgetWaterfall(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const items = data?.variances || data?.breakdown || [];

    if (!items.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No variance breakdown available</p>';
        return;
    }

    Plotly.newPlot(container, [{
        type: 'waterfall',
        x: items.map(i => i.category || i.name),
        y: items.map(i => i.variance || i.value || 0),
        measure: items.map(i => i.measure || 'relative'),
        connector: { line: { color: VIZ_COLORS.border } },
        increasing: { marker: { color: VIZ_COLORS.success } },
        decreasing: { marker: { color: VIZ_COLORS.error } },
        totals: { marker: { color: VIZ_COLORS.primary } }
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 50, r: 20, t: 30, b: 60 },
        title: { text: 'Budget Variance Waterfall', font: { size: 12 } },
        xaxis: { tickangle: -30, gridcolor: VIZ_COLORS.border },
        yaxis: { title: 'Variance ($)', gridcolor: VIZ_COLORS.border }
    }, { responsive: true, displayModeBar: false });
}

function renderBudgetBullet(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const actual = data?.actual || data?.summary?.actual || 0;
    const budget = data?.budget || data?.summary?.budget || actual * 1.1;
    const target = data?.target || budget;

    Plotly.newPlot(container, [{
        type: 'indicator',
        mode: 'number+gauge+delta',
        value: actual,
        delta: { reference: budget },
        number: { prefix: '$', valueformat: ',.0f' },
        gauge: {
            shape: 'bullet',
            axis: { range: [0, budget * 1.5] },
            bar: { color: actual > budget ? VIZ_COLORS.error : VIZ_COLORS.success },
            steps: [
                { range: [0, budget * 0.8], color: 'rgba(16, 185, 129, 0.2)' },
                { range: [budget * 0.8, budget], color: 'rgba(251, 191, 36, 0.2)' },
                { range: [budget, budget * 1.5], color: 'rgba(239, 68, 68, 0.2)' }
            ],
            threshold: {
                line: { color: VIZ_COLORS.primary, width: 3 },
                thickness: 0.75,
                value: target
            }
        }
    }], {
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 20, r: 20, t: 50, b: 20 },
        title: { text: 'Actual vs Budget', font: { size: 12 } }
    }, { responsive: true, displayModeBar: false });
}
