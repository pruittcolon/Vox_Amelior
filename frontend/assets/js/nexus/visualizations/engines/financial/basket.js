/**
 * Market Basket Visualization
 * Renders network graph for market basket analysis engine results.
 *
 * @module nexus/visualizations/engines/financial/basket
 */

import { VIZ_COLORS, isLowPowerMode, PERFORMANCE_LIMITS, escapeHtml } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Market Basket Analysis</h5>
            <div class="fin-chart-container-lg" id="basket-network-${vizId}" style="height: 450px;"></div>
        </div>
    `;
}

export function render(data, vizId) {
    renderBasketNetwork(data, `basket-network-${vizId}`);
}

function renderBasketNetwork(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const rules = data?.rules || data?.associations || [];
    const nodes = new Set();
    const links = [];

    rules.forEach(rule => {
        const antecedent = rule.antecedent || rule.if || rule.from || '';
        const consequent = rule.consequent || rule.then || rule.to || '';
        const confidence = rule.confidence || rule.lift || rule.support || 0.5;

        if (antecedent && consequent) {
            nodes.add(antecedent);
            nodes.add(consequent);
            links.push({ source: antecedent, target: consequent, value: confidence });
        }
    });

    if (nodes.size === 0) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No association rules found</p>';
        return;
    }

    // Limit nodes for performance
    const maxNodes = isLowPowerMode() ? PERFORMANCE_LIMITS.networkNodesLow : PERFORMANCE_LIMITS.networkNodes;
    const nodeArray = [...nodes].slice(0, maxNodes);
    const nodeMap = Object.fromEntries(nodeArray.map((n, i) => [n, i]));

    // Create positions using force-directed layout simulation
    const positions = nodeArray.map((_, i) => {
        const angle = (i / nodeArray.length) * 2 * Math.PI;
        const radius = 1 + Math.random() * 0.5;
        return {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius
        };
    });

    // Create edge traces
    const edgeX = [];
    const edgeY = [];
    const filteredLinks = links.filter(l => nodeMap[l.source] !== undefined && nodeMap[l.target] !== undefined);

    filteredLinks.slice(0, isLowPowerMode() ? PERFORMANCE_LIMITS.networkLinksLow : PERFORMANCE_LIMITS.networkLinks).forEach(link => {
        const sourceIdx = nodeMap[link.source];
        const targetIdx = nodeMap[link.target];
        edgeX.push(positions[sourceIdx].x, positions[targetIdx].x, null);
        edgeY.push(positions[sourceIdx].y, positions[targetIdx].y, null);
    });

    Plotly.newPlot(container, [
        {
            type: 'scatter',
            mode: 'lines',
            x: edgeX,
            y: edgeY,
            line: { width: 0.5, color: VIZ_COLORS.border },
            hoverinfo: 'none'
        },
        {
            type: 'scatter',
            mode: 'markers+text',
            x: positions.map(p => p.x),
            y: positions.map(p => p.y),
            text: nodeArray.map(n => n.length > 12 ? n.substring(0, 10) + '...' : n),
            textposition: 'top center',
            marker: {
                size: 20,
                color: VIZ_COLORS.primary,
                line: { width: 2, color: '#fff' }
            },
            hovertext: nodeArray,
            hoverinfo: 'text'
        }
    ], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 10 },
        margin: { l: 20, r: 20, t: 30, b: 20 },
        title: { text: `Association Network (${nodeArray.length} items)`, font: { size: 12 } },
        xaxis: { visible: false },
        yaxis: { visible: false },
        showlegend: false
    }, { responsive: true, displayModeBar: false });
}
