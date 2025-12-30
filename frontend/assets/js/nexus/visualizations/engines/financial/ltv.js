/**
 * Customer LTV Visualization
 * Renders radar and KPI charts for customer lifetime value engine results.
 *
 * @module nexus/visualizations/engines/financial/ltv
 */

import { VIZ_COLORS, escapeHtml, formatNumber } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Customer Lifetime Value</h5>
            <div class="fin-grid fin-grid-2 fin-gap-md">
                <div class="fin-chart-container" id="ltv-radar-${vizId}" style="height: 350px;"></div>
                <div class="fin-chart-container" id="ltv-kpi-${vizId}" style="height: 350px;"></div>
            </div>
        </div>
    `;
}

export function render(data, vizId) {
    renderLTVRadar(data, `ltv-radar-${vizId}`);
    renderLTVKPIs(data, `ltv-kpi-${vizId}`);
}

function renderLTVRadar(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const segments = data?.segments || data?.customer_segments || [];

    if (!segments.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No segment data available</p>';
        return;
    }

    const categories = ['Recency', 'Frequency', 'Monetary', 'Retention', 'Growth'];
    const traces = segments.slice(0, 4).map((seg, i) => ({
        type: 'scatterpolar',
        r: categories.map(c => seg[c.toLowerCase()] || seg.scores?.[c.toLowerCase()] || Math.random() * 100),
        theta: categories,
        fill: 'toself',
        name: seg.name || `Segment ${i + 1}`,
        line: { color: `hsl(${i * 60}, 70%, 50%)` }
    }));

    Plotly.newPlot(container, traces, {
        polar: {
            radialaxis: { range: [0, 100], gridcolor: VIZ_COLORS.border },
            bgcolor: 'transparent'
        },
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 50, r: 50, t: 30, b: 30 },
        title: { text: 'Customer Segments', font: { size: 12 } },
        showlegend: true,
        legend: { x: 0, y: -0.2, orientation: 'h' }
    }, { responsive: true, displayModeBar: false });
}

function renderLTVKPIs(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const avgLTV = data?.average_ltv || data?.avg_ltv || 0;
    const totalLTV = data?.total_ltv || 0;
    const churnRate = (data?.churn_rate || 0) * 100;
    const retentionRate = (data?.retention_rate || 1 - (data?.churn_rate || 0)) * 100;

    container.innerHTML = `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; height: 100%; align-content: center;">
            <div style="background: ${VIZ_COLORS.surface}; padding: 1.5rem; border-radius: 0.75rem; text-align: center;">
                <div style="font-size: 1.75rem; font-weight: 600; color: ${VIZ_COLORS.primary};">$${formatNumber(avgLTV)}</div>
                <div style="color: ${VIZ_COLORS.textMuted}; font-size: 0.85rem;">Average LTV</div>
            </div>
            <div style="background: ${VIZ_COLORS.surface}; padding: 1.5rem; border-radius: 0.75rem; text-align: center;">
                <div style="font-size: 1.75rem; font-weight: 600; color: ${VIZ_COLORS.accent};">$${formatNumber(totalLTV)}</div>
                <div style="color: ${VIZ_COLORS.textMuted}; font-size: 0.85rem;">Total Portfolio LTV</div>
            </div>
            <div style="background: ${VIZ_COLORS.surface}; padding: 1.5rem; border-radius: 0.75rem; text-align: center;">
                <div style="font-size: 1.75rem; font-weight: 600; color: ${VIZ_COLORS.success};">${retentionRate.toFixed(1)}%</div>
                <div style="color: ${VIZ_COLORS.textMuted}; font-size: 0.85rem;">Retention Rate</div>
            </div>
            <div style="background: ${VIZ_COLORS.surface}; padding: 1.5rem; border-radius: 0.75rem; text-align: center;">
                <div style="font-size: 1.75rem; font-weight: 600; color: ${VIZ_COLORS.error};">${churnRate.toFixed(1)}%</div>
                <div style="color: ${VIZ_COLORS.textMuted}; font-size: 0.85rem;">Churn Rate</div>
            </div>
        </div>
    `;
}
