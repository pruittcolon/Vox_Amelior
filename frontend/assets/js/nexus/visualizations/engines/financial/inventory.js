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

    const items = data?.items || data?.inventory || [];

    if (!items.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No inventory data available</p>';
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
