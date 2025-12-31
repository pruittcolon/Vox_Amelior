/**
 * Inventory Optimization Visualization
 * Renders treemap for inventory optimization engine results.
 *
 * @module nexus/visualizations/engines/financial/inventory
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Inventory Optimization</h5>
            <div class="fin-chart-container" id="inventory-treemap-${vizId}" style="height: 400px;"></div>
        </div>
    `;
}

export function render(data, vizId) {
    renderInventoryTreemap(data, `inventory-treemap-${vizId}`);
}

function renderInventoryTreemap(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    console.log('[Inventory] Data keys:', Object.keys(data || {}));

    // Check if this is a fallback response (engine couldn't find required columns)
    if (data?.fallback_used || data?.summary?.status === 'fallback_summary') {
        const reason = data?.summary?.reason || 'Inventory optimization requires columns like product_id, stock_level, reorder_point.';
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: #94a3b8;">
                <p style="margin-bottom: 0.5rem; font-weight: 500;">Data Requirements Not Met</p>
                <p style="font-size: 0.85rem; color: #64748b;">${reason}</p>
            </div>
        `;
        return;
    }

    // Try to extract items from various possible API response structures
    let items = data?.items || data?.inventory || [];


    // Check abc_analysis (actual API response structure)
    if (!items.length && data?.abc_analysis) {
        const abc = data.abc_analysis;
        items = [];
        // Extract items from each class
        ['classA', 'classB', 'classC'].forEach(cls => {
            if (abc[cls] && typeof abc[cls] === 'object') {
                Object.entries(abc[cls]).forEach(([name, value]) => {
                    items.push({
                        name: name,
                        value: value,
                        turnover: cls === 'classA' ? 3 : cls === 'classB' ? 1.5 : 0.3
                    });
                });
            }
        });
    }

    // Try recommendations as fallback
    if (!items.length && data?.recommendations?.length > 0) {
        items = data.recommendations.map(r => ({
            name: r.product || 'Unknown',
            value: 1000,
            turnover: r.action === 'Reorder Immediately' ? 0.2 : 1
        }));
    }

    if (!items.length) {
        // Show summary info if we have it
        const summary = data?.summary || {};
        if (summary.total_inventory_value || summary.total_items) {
            container.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 1rem; color: ${VIZ_COLORS.textMuted};">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: 700; color: ${VIZ_COLORS.success};">$${(summary.total_inventory_value || 0).toLocaleString()}</div>
                        <div style="font-size: 0.9rem;">Total Inventory Value</div>
                    </div>
                    <div style="display: flex; gap: 2rem;">
                        <div style="text-align: center;">
                            <div style="font-size: 1.25rem; color: ${VIZ_COLORS.primary};">${summary.total_items || 0}</div>
                            <div style="font-size: 0.8rem;">Items</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.25rem; color: ${VIZ_COLORS.primary};">${(summary.avg_turnover || 0).toFixed(1)}x</div>
                            <div style="font-size: 0.8rem;">Avg Turnover</div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No inventory data available</p>';
        }
        return;
    }

    // Color based on turnover rate or stock status
    const colors = items.map(i => {
        const turnover = i.turnover || i.turnover_rate || 1;
        if (turnover > 2) return VIZ_COLORS.success;
        if (turnover > 0.5) return VIZ_COLORS.warning;
        return VIZ_COLORS.error;
    });

    Plotly.newPlot(container, [{
        type: 'treemap',
        labels: items.map(i => i.name || i.sku || i.item),
        values: items.map(i => i.value || i.quantity || i.stock || 0),
        parents: items.map(() => ''),
        marker: { colors },
        textinfo: 'label+percent entry',
        hovertemplate: '<b>%{label}</b><br>Value: $%{value:,.0f}<extra></extra>'
    }], {
        paper_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 10, r: 10, t: 30, b: 10 },
        title: { text: 'Inventory Distribution', font: { size: 12 } }
    }, { responsive: true, displayModeBar: false });
}
