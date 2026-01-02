/**
 * Oracle Causality Visualization
 * Renders causal bar charts for oracle engine results.
 *
 * @module nexus/visualizations/engines/advanced/oracle
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section">
            <h5>Causal Analysis</h5>
            <div class="oracle-chart-container" id="oracle-chart-${vizId}" style="height: 350px;"></div>
        </div>
    `;
}

export function render(data, vizId) {
    renderOracleChart(data, `oracle-chart-${vizId}`);
}

function renderOracleChart(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    // Support multiple backend formats:
    // 1. data.causal_factors / data.causes / data.drivers (old format)
    // 2. data.causal_relationships (Oracle engine format: [{cause, effect, p_value, f_statistic, strength}])
    let causes = data?.causal_factors || data?.causes || data?.drivers || [];

    // Transform causal_relationships to compatible format
    if (!causes.length && data?.causal_relationships?.length) {
        causes = data.causal_relationships.map(rel => ({
            name: `${rel.cause} -> ${rel.effect}`,
            variable: rel.cause,
            effect: rel.effect,
            impact: rel.f_statistic || 0,
            coefficient: 1 - (rel.p_value || 0), // Higher is better (lower p-value)
            strength: rel.strength || 'Unknown'
        }));
    }

    if (!causes.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No causal factors identified</p>';
        return;
    }

    const sorted = [...causes].sort((a, b) => Math.abs(b.effect || b.impact || 0) - Math.abs(a.effect || a.impact || 0));
    const labels = sorted.map(c => c.name || c.variable || c.cause);
    const effects = sorted.map(c => c.effect_value || c.impact || c.coefficient || 0);
    const colors = effects.map(e => e >= 0 ? VIZ_COLORS.success : VIZ_COLORS.error);

    Plotly.newPlot(container, [{
        type: 'bar',
        orientation: 'h',
        y: labels,
        x: effects,
        marker: { color: colors },
        hovertemplate: '<b>%{y}</b><br>Effect: %{x:.3f}<extra></extra>'
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 120, r: 20, t: 30, b: 40 },
        title: { text: 'Causal Impact Analysis', font: { size: 12 } },
        xaxis: {
            title: 'Causal Effect',
            gridcolor: VIZ_COLORS.border,
            zeroline: true,
            zerolinecolor: VIZ_COLORS.textMuted
        },
        yaxis: { gridcolor: VIZ_COLORS.border }
    }, { responsive: true, displayModeBar: false });
}

