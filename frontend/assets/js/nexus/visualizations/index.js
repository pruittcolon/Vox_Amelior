/**
 * NexusViz Aggregator
 * Imports all engine visualizations and exposes window.NexusViz for use by engine-results.js
 * 
 * This file serves as the single entry point for all visualization modules.
 * It maps engine names to their respective visualization modules and provides
 * a unified API for building HTML sections and rendering charts.
 *
 * @module nexus/visualizations/index
 */

// ============================================================================
// Core Utilities
// ============================================================================

import {
    VIZ_COLORS,
    escapeHtml,
    formatNumber,
    getClusterColor,
    isLowPowerMode,
    chartInstanceCache,
    echartInstanceCache
} from './core/viz-utils.js';

import { installPlotlyGuards } from './core/plotly-helpers.js';
import { resizeAllECharts } from './core/echart-helpers.js';

// ============================================================================
// ML Engine Visualizations
// ============================================================================

import * as titanViz from './engines/ml/titan.js';
import * as clusteringViz from './engines/ml/clustering.js';
import * as anomalyViz from './engines/ml/anomaly.js';
import * as statisticalViz from './engines/ml/statistical.js';
import * as trendViz from './engines/ml/trend.js';
import * as predictiveViz from './engines/ml/predictive.js';
import * as graphsViz from './engines/ml/graphs.js';

// ============================================================================
// Financial Engine Visualizations
// ============================================================================

import * as costViz from './engines/financial/cost.js';
import * as roiViz from './engines/financial/roi.js';
import * as cashflowViz from './engines/financial/cashflow.js';
import * as budgetViz from './engines/financial/budget.js';
import * as marginViz from './engines/financial/margin.js';
import * as forecastViz from './engines/financial/forecast.js';
import * as ltvViz from './engines/financial/ltv.js';
import * as spendViz from './engines/financial/spend.js';
import * as inventoryViz from './engines/financial/inventory.js';
import * as pricingViz from './engines/financial/pricing.js';
import * as basketViz from './engines/financial/basket.js';
import * as resourceViz from './engines/financial/resource.js';

// ============================================================================
// Advanced Engine Visualizations
// ============================================================================

import * as ragViz from './engines/advanced/rag.js';
import * as chaosViz from './engines/advanced/chaos.js';
import * as oracleViz from './engines/advanced/oracle.js';

// ============================================================================
// Engine-to-Visualization Mapping
// ============================================================================

/**
 * Maps engine names to their visualization modules.
 * Each module must export buildSection(data, vizId) and render(data, vizId).
 */
const engineVizMap = {
    // ML Engines
    titan: titanViz,
    clustering: clusteringViz,
    anomaly: anomalyViz,
    statistical: statisticalViz,
    trend: trendViz,
    predictive: predictiveViz,
    graphs: graphsViz,

    // Financial Engines
    cost: costViz,
    roi: roiViz,
    cash_flow: cashflowViz,
    budget_variance: budgetViz,
    profit_margins: marginViz,
    revenue_forecasting: forecastViz,
    customer_ltv: ltvViz,
    spend_patterns: spendViz,
    inventory_optimization: inventoryViz,
    pricing_strategy: pricingViz,
    market_basket: basketViz,
    resource_utilization: resourceViz,

    // Advanced Engines
    rag_evaluation: ragViz,
    chaos: chaosViz,
    oracle: oracleViz
};

// ============================================================================
// Public API
// ============================================================================

/**
 * Build HTML section for engine visualization.
 * Called by engine-results.js to generate visualization container HTML.
 * 
 * @param {string} engineName - Engine identifier
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string for visualization section
 */
function buildVizSection(engineName, data, vizId) {
    const viz = engineVizMap[engineName];
    if (!viz || typeof viz.buildSection !== 'function') {
        console.log(`[NexusViz] No visualization module for engine: ${engineName}`);
        return '';
    }

    try {
        return viz.buildSection(data, vizId) || '';
    } catch (err) {
        console.error(`[NexusViz] Error building section for ${engineName}:`, err);
        return '';
    }
}

/**
 * Render engine-specific visualizations into containers.
 * Called by engine-results.js after HTML is inserted into DOM.
 * 
 * @param {string} engineName - Engine identifier
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
function renderEngineVisualizations(engineName, data, vizId) {
    console.log(`[NexusViz] renderEngineVisualizations called for: ${engineName}`, 'data keys:', data ? Object.keys(data) : 'null');

    const viz = engineVizMap[engineName];
    if (!viz || typeof viz.render !== 'function') {
        console.log(`[NexusViz] No render function for engine: ${engineName}`);
        return;
    }

    try {
        // Delay rendering slightly to ensure DOM is ready
        requestAnimationFrame(() => {
            console.log(`[NexusViz] Calling ${engineName}.render() with vizId: ${vizId}`);
            viz.render(data, vizId);
        });
    } catch (err) {
        console.error(`[NexusViz] Error rendering ${engineName}:`, err);
    }
}

/**
 * Resize all visible charts (called on window resize or tab change).
 */
function resizeVisibleCharts() {
    // Resize Plotly charts
    chartInstanceCache.forEach(instance => {
        if (instance && typeof instance.resize === 'function') {
            try {
                instance.resize();
            } catch (e) {
                // Ignore resize errors
            }
        }
    });

    // Resize ECharts
    resizeAllECharts();

    // Resize any Plotly charts in the DOM
    if (typeof Plotly !== 'undefined') {
        document.querySelectorAll('.js-plotly-plot').forEach(plot => {
            try {
                Plotly.Plots.resize(plot);
            } catch (e) {
                // Ignore resize errors
            }
        });
    }

    // Resize financial chart factory if present
    if (window.financialChartFactory?.resizeAll) {
        window.financialChartFactory.resizeAll();
    }
}

/**
 * Highlight a specific cluster in clustering visualization.
 * @param {number} clusterId - Cluster ID to highlight
 * @param {string} vizId - Visualization ID
 */
function highlightCluster(clusterId, vizId) {
    if (clusteringViz.highlightCluster) {
        clusteringViz.highlightCluster(clusterId, vizId);
    }
}

// ============================================================================
// Initialize and Export
// ============================================================================

// Install Plotly guards on load
installPlotlyGuards();

// Handle window resize
let resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resizeVisibleCharts, 250);
});

// Export to window for use by engine-results.js (which uses window.NexusViz)
window.NexusViz = {
    buildVizSection,
    renderEngineVisualizations,
    resizeVisibleCharts,
    highlightCluster,
    getClusterColor,
    VIZ_COLORS
};

// Signal that NexusViz is ready for use
window.dispatchEvent(new CustomEvent('nexusviz-ready'));

console.log('[NexusViz] Modular visualization system loaded - 22 engines supported');
