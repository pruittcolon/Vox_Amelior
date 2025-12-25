/**
 * NexusAI Engine Definitions
 * Complete registry of all 22 analysis engines with metadata.
 *
 * @module nexus/engines/engine-definitions
 */

// ============================================================================
// Engine Registry
// ============================================================================

/**
 * @typedef {Object} EngineDefinition
 * @property {string} name - Unique engine identifier (API endpoint key)
 * @property {string} display - Human-readable display name
 * @property {string} icon - Emoji icon for the engine
 * @property {'ml'|'financial'|'advanced'} category - Engine category
 * @property {string} [description] - Brief description of what the engine does
 */

/**
 * All 22 analysis engines organized by category.
 * @type {EngineDefinition[]}
 */
export const ALL_ENGINES = [
    // =========================================================================
    // ML & Analytics Engines (7 total)
    // =========================================================================
    {
        name: 'titan',
        display: 'Titan AutoML',
        icon: '🤖',
        category: 'ml',
        description: 'Automatic machine learning model selection and training'
    },
    {
        name: 'predictive',
        display: 'Predictive Forecasting',
        icon: '📈',
        category: 'ml',
        description: 'Time-series predictions and trend forecasting'
    },
    {
        name: 'clustering',
        display: 'Clustering',
        icon: '🎯',
        category: 'ml',
        description: 'Automatic data segmentation and cluster analysis'
    },
    {
        name: 'anomaly',
        display: 'Anomaly Detection',
        icon: '🔍',
        category: 'ml',
        description: 'Identify outliers and unusual patterns in data'
    },
    {
        name: 'statistical',
        display: 'Statistical Analysis',
        icon: '📊',
        category: 'ml',
        description: 'Comprehensive statistical metrics and distributions'
    },
    {
        name: 'trend',
        display: 'Trend Analysis',
        icon: '📉',
        category: 'ml',
        description: 'Detect and quantify data trends over time'
    },
    {
        name: 'graphs',
        display: 'Universal Graph',
        icon: '📈',
        category: 'ml',
        description: 'Automatic visualization generation for any dataset'
    },

    // =========================================================================
    // Financial Intelligence Engines (12 total)
    // =========================================================================
    {
        name: 'cost',
        display: 'Cost Optimization',
        icon: '💰',
        category: 'financial',
        description: 'Identify cost reduction opportunities'
    },
    {
        name: 'roi',
        display: 'ROI Prediction',
        icon: '💵',
        category: 'financial',
        description: 'Calculate and predict return on investment'
    },
    {
        name: 'spend_patterns',
        display: 'Spend Pattern Analysis',
        icon: '💳',
        category: 'financial',
        description: 'Analyze spending behaviors and patterns'
    },
    {
        name: 'budget_variance',
        display: 'Budget Variance',
        icon: '📋',
        category: 'financial',
        description: 'Track actual vs. budgeted amounts'
    },
    {
        name: 'profit_margins',
        display: 'Profit Margin Analysis',
        icon: '📊',
        category: 'financial',
        description: 'Analyze profit margins across products/services'
    },
    {
        name: 'revenue_forecasting',
        display: 'Revenue Forecasting',
        icon: '💹',
        category: 'financial',
        description: 'Predict future revenue based on historical data'
    },
    {
        name: 'customer_ltv',
        display: 'Customer LTV',
        icon: '👥',
        category: 'financial',
        description: 'Calculate customer lifetime value'
    },
    {
        name: 'cash_flow',
        display: 'Cash Flow Analysis',
        icon: '💸',
        category: 'financial',
        description: 'Analyze cash flow patterns and projections'
    },
    {
        name: 'inventory_optimization',
        display: 'Inventory Optimization',
        icon: '📦',
        category: 'financial',
        description: 'Optimize inventory levels and turnover'
    },
    {
        name: 'pricing_strategy',
        display: 'Pricing Strategy',
        icon: '🏷️',
        category: 'financial',
        description: 'Analyze pricing effectiveness and recommendations'
    },
    {
        name: 'market_basket',
        display: 'Market Basket Analysis',
        icon: '🛒',
        category: 'financial',
        description: 'Discover product associations and cross-sell opportunities'
    },
    {
        name: 'resource_utilization',
        display: 'Resource Utilization',
        icon: '⚙️',
        category: 'financial',
        description: 'Analyze resource usage and efficiency'
    },

    // =========================================================================
    // Advanced AI Lab Engines (3 total)
    // =========================================================================
    {
        name: 'rag_evaluation',
        display: 'RAG Evaluation',
        icon: '🔬',
        category: 'advanced',
        description: 'Evaluate retrieval-augmented generation quality'
    },
    {
        name: 'chaos',
        display: 'Chaos Engine',
        icon: '🌀',
        category: 'advanced',
        description: 'Chaos analysis and sensitivity testing'
    },
    {
        name: 'oracle',
        display: 'Oracle Causality',
        icon: '🔮',
        category: 'advanced',
        description: 'Causal inference and relationship discovery'
    }
];

// ============================================================================
// Engine Helpers
// ============================================================================

/**
 * Get engines filtered by category.
 * @param {'ml'|'financial'|'advanced'} category
 * @returns {EngineDefinition[]}
 */
export function getEnginesByCategory(category) {
    return ALL_ENGINES.filter(engine => engine.category === category);
}

/**
 * Get an engine by its name.
 * @param {string} name
 * @returns {EngineDefinition|undefined}
 */
export function getEngineByName(name) {
    return ALL_ENGINES.find(engine => engine.name === name);
}

/**
 * Get all engine names.
 * @returns {string[]}
 */
export function getAllEngineNames() {
    return ALL_ENGINES.map(engine => engine.name);
}

/**
 * Get category counts.
 * @returns {Object<string, number>}
 */
export function getCategoryCounts() {
    const counts = { ml: 0, financial: 0, advanced: 0 };
    ALL_ENGINES.forEach(engine => {
        counts[engine.category]++;
    });
    return counts;
}

/**
 * Total number of engines.
 * @type {number}
 */
export const ENGINE_COUNT = ALL_ENGINES.length;
