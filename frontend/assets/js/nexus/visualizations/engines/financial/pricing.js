/**
 * Pricing Strategy Visualization
 * Renders 3D surface chart for pricing strategy engine results.
 *
 * @module nexus/visualizations/engines/financial/pricing
 */

import { VIZ_COLORS, isLowPowerMode } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Pricing Strategy</h5>
            <div class="fin-chart-container-xl" id="pricing-surface-${vizId}" style="height: 450px;"></div>
        </div>
    `;
}

export function render(data, vizId) {
    renderPricingSurface(data, `pricing-surface-${vizId}`);
}

function renderPricingSurface(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const surface = data?.price_elasticity || data?.surface || data?.matrix || null;

    if (!surface) {
        // Fallback to simpler visualization
        renderPricingSimple(data, container);
        return;
    }

    const z = Array.isArray(surface[0]) ? surface : [surface];
    const x = data?.price_points || Array.from({ length: z[0].length }, (_, i) => i * 10 + 50);
    const y = data?.demand_levels || Array.from({ length: z.length }, (_, i) => i * 100 + 100);

    if (isLowPowerMode()) {
        // 2D heatmap for low power
        Plotly.newPlot(container, [{
            type: 'heatmap',
            z,
            x,
            y,
            colorscale: 'Viridis'
        }], {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: VIZ_COLORS.textMuted, size: 11 },
            margin: { l: 50, r: 40, t: 30, b: 40 },
            title: { text: 'Price Sensitivity (2D)', font: { size: 12 } },
            xaxis: { title: 'Price' },
            yaxis: { title: 'Demand' }
        }, { responsive: true, displayModeBar: false });
        return;
    }

    Plotly.newPlot(container, [{
        type: 'surface',
        z,
        x,
        y,
        colorscale: 'Viridis',
        contours: {
            z: { show: true, usecolormap: true, highlightcolor: '#fff', project: { z: true } }
        }
    }], {
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 0, r: 0, t: 30, b: 0 },
        title: { text: 'Price-Demand Surface', font: { size: 12 } },
        scene: {
            xaxis: { title: 'Price', gridcolor: VIZ_COLORS.border },
            yaxis: { title: 'Demand', gridcolor: VIZ_COLORS.border },
            zaxis: { title: 'Revenue', gridcolor: VIZ_COLORS.border },
            bgcolor: 'transparent'
        }
    }, { responsive: true, displayModeBar: true });
}

function renderPricingSimple(data, container) {
    // The API returns recommendations and elasticity, not optimal_price
    const recommendations = data?.recommendations || [];
    const elasticities = data?.elasticity || [];
    const insights = data?.insights || [];

    // If we have recommendations, show them
    if (recommendations.length > 0 || elasticities.length > 0) {
        const rec = recommendations[0] || {};
        const elast = elasticities[0] || {};

        const avgPrice = elast.avg_price || 0;
        const elasticity = elast.elasticity || rec.elasticity || 0;
        const action = rec.action || 'Maintain';
        const reason = rec.reason || 'Analysis complete';

        container.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 1.5rem; padding: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 0.9rem; color: ${VIZ_COLORS.textMuted};">Recommendation</div>
                    <div style="font-size: 2rem; font-weight: 700; color: ${action === 'Raise Price' ? VIZ_COLORS.success : action === 'Lower Price' ? VIZ_COLORS.warning : VIZ_COLORS.primary};">${action}</div>
                </div>
                <div style="display: flex; gap: 2rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: ${VIZ_COLORS.textMuted};">Elasticity</div>
                        <div style="font-size: 1.25rem; color: ${VIZ_COLORS.primary};">${elasticity.toFixed(2)}</div>
                    </div>
                    ${avgPrice > 0 ? `
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: ${VIZ_COLORS.textMuted};">Avg Price</div>
                        <div style="font-size: 1.25rem; color: ${VIZ_COLORS.textMuted};">$${avgPrice.toLocaleString()}</div>
                    </div>
                    ` : ''}
                </div>
                <div style="font-size: 0.85rem; color: ${VIZ_COLORS.textMuted}; text-align: center; max-width: 400px;">${reason}</div>
            </div>
        `;
        return;
    }

    // Fallback for legacy data format
    const optimal = data?.optimal_price || data?.recommended_price || 0;
    const current = data?.current_price || 0;
    const range = data?.price_range || [current * 0.8, current * 1.2];

    container.innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 1.5rem;">
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; color: ${VIZ_COLORS.textMuted};">Optimal Price Point</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: ${VIZ_COLORS.success};">$${optimal.toFixed(2)}</div>
            </div>
            <div style="display: flex; gap: 2rem;">
                <div style="text-align: center;">
                    <div style="font-size: 0.8rem; color: ${VIZ_COLORS.textMuted};">Current</div>
                    <div style="font-size: 1.25rem; color: ${VIZ_COLORS.primary};">$${current.toFixed(2)}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.8rem; color: ${VIZ_COLORS.textMuted};">Range</div>
                    <div style="font-size: 1.25rem; color: ${VIZ_COLORS.textMuted};">$${range[0].toFixed(0)} - $${range[1].toFixed(0)}</div>
                </div>
            </div>
        </div>
    `;
}
