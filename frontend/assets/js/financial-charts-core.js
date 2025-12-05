/**
 * Financial Charts Core Module
 * 
 * Enterprise-grade financial visualization foundation.
 * Provides unified theme, color palettes, chart factory pattern,
 * and utility functions for all financial chart modules.
 * 
 * @module FinancialChartsCore
 * @requires Plotly.js 2.27+
 * @requires ECharts 5.4+
 */

'use strict';

// =============================================================================
// ENTERPRISE THEME CONFIGURATION
// =============================================================================

const FinancialTheme = {
    // Primary brand colors
    colors: {
        primary: '#6366f1',      // Indigo - main accent
        secondary: '#8b5cf6',    // Violet - secondary accent
        success: '#10b981',      // Emerald - positive/profit
        danger: '#ef4444',       // Red - negative/loss
        warning: '#f59e0b',      // Amber - caution
        info: '#06b6d4',         // Cyan - informational
        neutral: '#64748b',      // Slate - neutral
        
        // Financial-specific
        revenue: '#22c55e',      // Green - revenue
        cost: '#f43f5e',         // Rose - costs
        profit: '#10b981',       // Emerald - profit
        loss: '#dc2626',         // Red - loss
        forecast: '#a855f7',     // Purple - predictions
        actual: '#3b82f6',       // Blue - actuals
        budget: '#94a3b8',       // Gray - budgets
        variance: '#f97316',     // Orange - variance
        
        // Gradient stops
        gradientStart: '#6366f1',
        gradientEnd: '#8b5cf6',
    },
    
    // Color palettes for multi-series charts
    palettes: {
        // Premium financial palette
        financial: [
            '#6366f1', '#8b5cf6', '#a855f7', '#d946ef',
            '#ec4899', '#f43f5e', '#f97316', '#eab308',
            '#22c55e', '#14b8a6', '#06b6d4', '#3b82f6'
        ],
        
        // Sequential for heatmaps
        sequential: [
            '#f0fdf4', '#bbf7d0', '#86efac', '#4ade80',
            '#22c55e', '#16a34a', '#15803d', '#166534'
        ],
        
        // Diverging for variance
        diverging: [
            '#dc2626', '#f87171', '#fca5a5', '#fecaca',
            '#e5e7eb', '#bbf7d0', '#86efac', '#4ade80', '#22c55e'
        ],
        
        // Categorical
        categorical: [
            '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
            '#8b5cf6', '#06b6d4', '#f97316', '#84cc16'
        ]
    },
    
    // Typography
    fonts: {
        primary: '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
        mono: '"JetBrains Mono", "Fira Code", monospace',
        sizes: {
            xs: 10,
            sm: 11,
            md: 12,
            lg: 14,
            xl: 16,
            xxl: 20,
            hero: 32
        }
    },
    
    // Chart backgrounds
    backgrounds: {
        chart: 'rgba(15, 23, 42, 0.6)',
        card: 'rgba(30, 41, 59, 0.8)',
        hover: 'rgba(99, 102, 241, 0.1)',
        grid: 'rgba(148, 163, 184, 0.1)'
    },
    
    // Borders
    borders: {
        color: 'rgba(148, 163, 184, 0.2)',
        radius: 12,
        width: 1
    },
    
    // Shadows
    shadows: {
        sm: '0 1px 2px rgba(0, 0, 0, 0.3)',
        md: '0 4px 6px rgba(0, 0, 0, 0.4)',
        lg: '0 10px 15px rgba(0, 0, 0, 0.5)',
        glow: '0 0 20px rgba(99, 102, 241, 0.3)'
    },
    
    // Animation timings
    animations: {
        fast: 200,
        normal: 400,
        slow: 800,
        reveal: 1200
    }
};


// =============================================================================
// PLOTLY BASE CONFIGURATION
// =============================================================================

const PlotlyBaseConfig = {
    // Disable mode bar for cleaner look (or customize)
    displayModeBar: false,
    responsive: true,
    scrollZoom: false,
    
    // Clean up defaults
    staticPlot: false,
    
    // Locale
    locale: 'en-US'
};

const PlotlyBaseLayout = {
    // Dark theme
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    
    // Typography
    font: {
        family: FinancialTheme.fonts.primary,
        size: FinancialTheme.fonts.sizes.md,
        color: '#e2e8f0'
    },
    
    // Margins
    margin: { l: 60, r: 30, t: 40, b: 60, pad: 4 },
    
    // Grid styling
    xaxis: {
        gridcolor: FinancialTheme.backgrounds.grid,
        linecolor: FinancialTheme.borders.color,
        zerolinecolor: FinancialTheme.borders.color,
        tickfont: { color: '#94a3b8', size: 11 },
        titlefont: { color: '#cbd5e1', size: 12 }
    },
    
    yaxis: {
        gridcolor: FinancialTheme.backgrounds.grid,
        linecolor: FinancialTheme.borders.color,
        zerolinecolor: FinancialTheme.borders.color,
        tickfont: { color: '#94a3b8', size: 11 },
        titlefont: { color: '#cbd5e1', size: 12 }
    },
    
    // Legend
    legend: {
        font: { color: '#e2e8f0', size: 11 },
        bgcolor: 'rgba(0,0,0,0)',
        bordercolor: 'rgba(0,0,0,0)'
    },
    
    // Hover
    hoverlabel: {
        bgcolor: '#1e293b',
        bordercolor: '#475569',
        font: { family: FinancialTheme.fonts.primary, color: '#f1f5f9', size: 12 }
    }
};


// =============================================================================
// ECHARTS BASE CONFIGURATION
// =============================================================================

const EChartsBaseTheme = {
    // Background
    backgroundColor: 'transparent',
    
    // Text styles
    textStyle: {
        fontFamily: FinancialTheme.fonts.primary,
        color: '#e2e8f0'
    },
    
    // Title
    title: {
        textStyle: {
            color: '#f8fafc',
            fontWeight: 600,
            fontSize: 16
        },
        subtextStyle: {
            color: '#94a3b8',
            fontSize: 12
        }
    },
    
    // Legend
    legend: {
        textStyle: {
            color: '#cbd5e1'
        }
    },
    
    // Tooltip
    tooltip: {
        backgroundColor: 'rgba(30, 41, 59, 0.95)',
        borderColor: '#475569',
        borderWidth: 1,
        textStyle: {
            color: '#f1f5f9'
        },
        extraCssText: 'box-shadow: 0 4px 12px rgba(0,0,0,0.4); border-radius: 8px;'
    },
    
    // Color palette
    color: FinancialTheme.palettes.financial
};


// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

const FinancialUtils = {
    /**
     * Format currency value
     * @param {number} value - The value to format
     * @param {string} currency - Currency code (USD, EUR, etc.)
     * @param {boolean} compact - Use compact notation for large numbers
     */
    formatCurrency(value, currency = 'USD', compact = false) {
        if (value == null || isNaN(value)) return '$0';
        
        const options = {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 0,
            maximumFractionDigits: compact ? 1 : 2
        };
        
        if (compact && Math.abs(value) >= 1000) {
            options.notation = 'compact';
            options.compactDisplay = 'short';
        }
        
        return new Intl.NumberFormat('en-US', options).format(value);
    },
    
    /**
     * Format percentage
     * @param {number} value - The value (0.15 = 15%)
     * @param {number} decimals - Decimal places
     */
    formatPercent(value, decimals = 1) {
        if (value == null || isNaN(value)) return '0%';
        return `${(value * 100).toFixed(decimals)}%`;
    },
    
    /**
     * Format large numbers with K/M/B suffix
     */
    formatCompact(value) {
        if (value == null || isNaN(value)) return '0';
        
        const absValue = Math.abs(value);
        const sign = value < 0 ? '-' : '';
        
        if (absValue >= 1e9) return sign + (absValue / 1e9).toFixed(1) + 'B';
        if (absValue >= 1e6) return sign + (absValue / 1e6).toFixed(1) + 'M';
        if (absValue >= 1e3) return sign + (absValue / 1e3).toFixed(1) + 'K';
        return sign + absValue.toFixed(0);
    },
    
    /**
     * Get color based on value (positive = green, negative = red)
     */
    getValueColor(value, invert = false) {
        const positive = invert ? FinancialTheme.colors.danger : FinancialTheme.colors.success;
        const negative = invert ? FinancialTheme.colors.success : FinancialTheme.colors.danger;
        return value >= 0 ? positive : negative;
    },
    
    /**
     * Generate gradient color array
     */
    generateGradient(startColor, endColor, steps) {
        const start = this.hexToRgb(startColor);
        const end = this.hexToRgb(endColor);
        const gradient = [];
        
        for (let i = 0; i < steps; i++) {
            const t = i / (steps - 1);
            const r = Math.round(start.r + (end.r - start.r) * t);
            const g = Math.round(start.g + (end.g - start.g) * t);
            const b = Math.round(start.b + (end.b - start.b) * t);
            gradient.push(`rgb(${r},${g},${b})`);
        }
        
        return gradient;
    },
    
    /**
     * Convert hex to RGB
     */
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    },
    
    /**
     * Deep merge objects
     */
    deepMerge(target, source) {
        const result = { ...target };
        for (const key in source) {
            if (source[key] instanceof Object && key in target) {
                result[key] = this.deepMerge(target[key], source[key]);
            } else {
                result[key] = source[key];
            }
        }
        return result;
    },
    
    /**
     * Debounce function for resize handlers
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    /**
     * Create animated number counter
     */
    animateNumber(element, endValue, duration = 1000, format = 'currency') {
        const startValue = 0;
        const startTime = performance.now();
        
        const updateNumber = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function (ease-out-cubic)
            const eased = 1 - Math.pow(1 - progress, 3);
            const currentValue = startValue + (endValue - startValue) * eased;
            
            if (format === 'currency') {
                element.textContent = this.formatCurrency(currentValue, 'USD', true);
            } else if (format === 'percent') {
                element.textContent = this.formatPercent(currentValue);
            } else {
                element.textContent = this.formatCompact(currentValue);
            }
            
            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            }
        };
        
        requestAnimationFrame(updateNumber);
    }
};


// =============================================================================
// CHART FACTORY
// =============================================================================

class FinancialChartFactory {
    constructor() {
        this.charts = new Map();
        this.theme = FinancialTheme;
        this.utils = FinancialUtils;
        
        // Register resize handler
        window.addEventListener('resize', FinancialUtils.debounce(() => {
            this.resizeAll();
        }, 250));
    }
    
    /**
     * Create a new chart instance
     * @param {string} containerId - DOM element ID
     * @param {string} type - Chart type
     * @param {object} data - Chart data
     * @param {object} options - Additional options
     */
    create(containerId, type, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container #${containerId} not found`);
            return null;
        }
        
        // Clear existing chart
        if (this.charts.has(containerId)) {
            this.destroy(containerId);
        }
        
        let chart;
        
        switch (type) {
            // Plotly charts
            case 'waterfall':
            case 'sankey':
            case 'treemap':
            case 'bullet':
            case 'tornado':
            case 'scatter3d':
            case 'surface':
                chart = this._createPlotlyChart(container, type, data, options);
                break;
            
            // ECharts
            case 'gauge':
            case 'gaugeCluster':
            case 'radar':
            case 'network':
                chart = this._createEChartsChart(container, type, data, options);
                break;
            
            default:
                console.error(`Unknown chart type: ${type}`);
                return null;
        }
        
        this.charts.set(containerId, { type, chart, library: this._getLibrary(type) });
        return chart;
    }
    
    /**
     * Get chart library for type
     */
    _getLibrary(type) {
        const echartsTypes = ['gauge', 'gaugeCluster', 'radar', 'network'];
        return echartsTypes.includes(type) ? 'echarts' : 'plotly';
    }
    
    /**
     * Create Plotly chart - delegates to specific modules
     */
    _createPlotlyChart(container, type, data, options) {
        // Will be populated by other modules
        console.log(`Plotly chart type '${type}' - use specific module`);
        return null;
    }
    
    /**
     * Create ECharts chart - delegates to specific modules
     */
    _createEChartsChart(container, type, data, options) {
        // Will be populated by other modules
        console.log(`ECharts type '${type}' - use specific module`);
        return null;
    }
    
    /**
     * Update existing chart
     */
    update(containerId, data, options = {}) {
        const chartInfo = this.charts.get(containerId);
        if (!chartInfo) return;
        
        if (chartInfo.library === 'plotly') {
            Plotly.react(containerId, data.traces, data.layout, PlotlyBaseConfig);
        } else if (chartInfo.library === 'echarts') {
            chartInfo.chart.setOption(data, options.notMerge || false);
        }
    }
    
    /**
     * Destroy chart
     */
    destroy(containerId) {
        const chartInfo = this.charts.get(containerId);
        if (!chartInfo) return;
        
        if (chartInfo.library === 'plotly') {
            Plotly.purge(containerId);
        } else if (chartInfo.library === 'echarts') {
            chartInfo.chart.dispose();
        }
        
        this.charts.delete(containerId);
    }
    
    /**
     * Resize all charts
     */
    resizeAll() {
        this.charts.forEach((chartInfo, containerId) => {
            if (chartInfo.library === 'plotly') {
                Plotly.Plots.resize(containerId);
            } else if (chartInfo.library === 'echarts') {
                chartInfo.chart.resize();
            }
        });
    }
    
    /**
     * Get theme
     */
    getTheme() {
        return this.theme;
    }
    
    /**
     * Get utilities
     */
    getUtils() {
        return this.utils;
    }
}


// =============================================================================
// EXPORTS
// =============================================================================

// Create singleton instance
const financialChartFactory = new FinancialChartFactory();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        FinancialTheme,
        FinancialUtils,
        FinancialChartFactory,
        financialChartFactory,
        PlotlyBaseConfig,
        PlotlyBaseLayout,
        EChartsBaseTheme
    };
}

// Export for browser
if (typeof window !== 'undefined') {
    window.FinancialTheme = FinancialTheme;
    window.FinancialUtils = FinancialUtils;
    window.FinancialChartFactory = FinancialChartFactory;
    window.financialChartFactory = financialChartFactory;
    window.PlotlyBaseConfig = PlotlyBaseConfig;
    window.PlotlyBaseLayout = PlotlyBaseLayout;
    window.EChartsBaseTheme = EChartsBaseTheme;
}
