/**
 * Financial Flow Charts Module
 * 
 * Premium Waterfall and Sankey diagrams for financial analysis.
 * Waterfall: Budget variance, NPV breakdown, P&L analysis
 * Sankey: Cash flow visualization, spend allocation, revenue streams
 * 
 * @module FinancialChartsFlow
 * @requires FinancialChartsCore
 * @requires Plotly.js 2.27+
 */

'use strict';

// Safe getters for core module dependencies
const getPlotlyBaseConfig = () => window.PlotlyBaseConfig || PlotlyBaseConfig || { responsive: true };
const getPlotlyBaseLayout = () => window.PlotlyBaseLayout || PlotlyBaseLayout || {};
const getTheme = () => window.FinancialTheme || FinancialTheme || { colors: {}, palettes: { financial: ['#6366f1'] } };
const getUtils = () => window.FinancialUtils || FinancialUtils || { formatCurrency: (v) => `$${v}` };

// =============================================================================
// WATERFALL CHART - Premium Styled
// =============================================================================

const WaterfallChart = {
    // Premium gradient colors
    gradients: {
        increase: ['#059669', '#10b981', '#34d399'],  // Emerald gradient
        decrease: ['#dc2626', '#ef4444', '#f87171'],  // Red gradient
        total: ['#6366f1', '#818cf8', '#a5b4fc']      // Indigo gradient
    },
    
    /**
     * Create a waterfall chart for variance analysis
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('WaterfallChart: Container not found:', containerId);
            return null;
        }
        
        const theme = window.FinancialTheme || { 
            fonts: { primary: 'Inter, sans-serif' },
            colors: { success: '#10b981', danger: '#ef4444', primary: '#6366f1' }
        };
        const utils = window.FinancialUtils || { 
            formatCurrency: (v) => `$${v?.toLocaleString() || 0}` 
        };
        
        // Calculate cumulative values for waterfall
        const { measures, hoverText } = this._processWaterfallData(data, theme, utils);
        
        const trace = {
            type: 'waterfall',
            orientation: 'v',
            measure: measures,
            x: data.labels,
            y: data.values,
            text: data.values.map((v, i) => {
                if (measures[i] === 'total') return '$' + Math.abs(v).toLocaleString(undefined, {maximumFractionDigits: 0});
                const sign = v >= 0 ? '+' : '';
                return sign + '$' + v.toLocaleString(undefined, {maximumFractionDigits: 0});
            }),
            textposition: 'outside',
            textfont: {
                family: theme.fonts?.primary || 'Inter, sans-serif',
                size: 12,
                color: '#e2e8f0'
            },
            connector: {
                line: {
                    color: 'rgba(148, 163, 184, 0.4)',
                    width: 2,
                    dash: 'dot'
                },
                visible: true
            },
            increasing: {
                marker: {
                    color: this.gradients.increase[1],
                    line: { color: this.gradients.increase[0], width: 2 }
                }
            },
            decreasing: {
                marker: {
                    color: this.gradients.decrease[1],
                    line: { color: this.gradients.decrease[0], width: 2 }
                }
            },
            totals: {
                marker: {
                    color: this.gradients.total[1],
                    line: { color: this.gradients.total[0], width: 2 }
                }
            },
            hoverinfo: 'text',
            hovertext: hoverText,
            hoverlabel: {
                bgcolor: 'rgba(30, 41, 59, 0.95)',
                bordercolor: '#6366f1',
                font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
            }
        };
        
        const layout = {
            title: '',
            showlegend: false,
            font: {
                family: theme.fonts?.primary || 'Inter, sans-serif',
                color: '#e2e8f0'
            },
            xaxis: {
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                linecolor: 'rgba(148, 163, 184, 0.2)',
                tickfont: { color: '#94a3b8', size: 11 },
                tickangle: data.labels.length > 6 ? -45 : 0
            },
            yaxis: {
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                linecolor: 'rgba(148, 163, 184, 0.2)',
                tickfont: { color: '#94a3b8', size: 11 },
                tickformat: '$,.0f',
                zeroline: true,
                zerolinecolor: 'rgba(148, 163, 184, 0.3)',
                zerolinewidth: 2
            },
            height: options.height || 380,
            margin: { l: 70, r: 30, t: 30, b: data.labels.length > 6 ? 90 : 50 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            bargap: 0.3
        };
        
        const config = { displayModeBar: false, responsive: true };
        
        try {
            Plotly.newPlot(containerId, [trace], layout, config);
            console.log('WaterfallChart rendered successfully');
        } catch (e) {
            console.error('WaterfallChart failed:', e);
        }
        
        return { containerId, type: 'waterfall' };
    },
    
    /**
     * Process data for waterfall chart
     */
    _processWaterfallData(data, theme, utils) {
        const measures = [];
        const hoverText = [];
        
        let cumulative = 0;
        
        data.values.forEach((value, index) => {
            const label = data.labels[index];
            const isTotal = label.toLowerCase().includes('total') || 
                           label.toLowerCase().includes('net') ||
                           label.toLowerCase().includes('actual') ||
                           label.toLowerCase().includes('budget') ||
                           (index === data.values.length - 1 && data.showTotal !== false) ||
                           (data.totals && data.totals.includes(index));
            
            if (isTotal) {
                measures.push('total');
                hoverText.push(
                    `<b style="font-size:14px">${label}</b><br>` +
                    `<span style="color:#8b5cf6">📊 Total:</span> $${Math.abs(value).toLocaleString()}`
                );
            } else if (value >= 0) {
                measures.push('relative');
                cumulative += value;
                hoverText.push(
                    `<b style="font-size:14px">${label}</b><br>` +
                    `<span style="color:#10b981">📈 Increase:</span> +$${value.toLocaleString()}<br>` +
                    `<span style="color:#94a3b8">Running Total:</span> $${cumulative.toLocaleString()}`
                );
            } else {
                measures.push('relative');
                cumulative += value;
                hoverText.push(
                    `<b style="font-size:14px">${label}</b><br>` +
                    `<span style="color:#ef4444">📉 Decrease:</span> -$${Math.abs(value).toLocaleString()}<br>` +
                    `<span style="color:#94a3b8">Running Total:</span> $${cumulative.toLocaleString()}`
                );
            }
        });
        
        return { measures, hoverText };
    },
    
    /**
     * Create budget variance waterfall
     */
    createBudgetVariance(containerId, budgetData, actualData, categories, options = {}) {
        const labels = ['Budget', ...categories, 'Actual'];
        const budgetTotal = budgetData.reduce((a, b) => a + b, 0);
        const actualTotal = actualData.reduce((a, b) => a + b, 0);
        
        // Calculate variances
        const variances = categories.map((_, i) => actualData[i] - budgetData[i]);
        const values = [budgetTotal, ...variances, actualTotal];
        
        return this.create(containerId, {
            labels,
            values,
            title: options.title || 'Budget vs Actual Variance',
            totals: [0, labels.length - 1]
        }, options);
    },
    
    /**
     * Create P&L waterfall
     */
    createPnL(containerId, data, options = {}) {
        const labels = ['Revenue', 'COGS', 'Gross Profit', 'OpEx', 'EBITDA', 'D&A', 'EBIT', 'Interest', 'Taxes', 'Net Income'];
        
        // Default P&L structure if only totals provided
        const values = data.values || [
            data.revenue || 1000000,
            -(data.cogs || 400000),
            data.grossProfit || 600000,
            -(data.opex || 200000),
            data.ebitda || 400000,
            -(data.depreciation || 50000),
            data.ebit || 350000,
            -(data.interest || 30000),
            -(data.taxes || 80000),
            data.netIncome || 240000
        ];
        
        return this.create(containerId, {
            labels: data.labels || labels,
            values,
            title: options.title || 'Profit & Loss Breakdown',
            totals: [2, 4, 6, 9]
        }, options);
    }
};


// =============================================================================
// SANKEY DIAGRAM - Premium Styled
// =============================================================================

const SankeyChart = {
    // Premium color scales for different flow types
    colorScales: {
        inflow: ['#22c55e', '#4ade80', '#86efac', '#bbf7d0'],   // Greens
        outflow: ['#ef4444', '#f87171', '#fca5a5', '#fecaca'],  // Reds
        neutral: ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe'],  // Indigos
        categories: [
            ['#0ea5e9', '#38bdf8', '#7dd3fc', '#bae6fd'],       // Sky blues
            ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe'],       // Violets
            ['#f97316', '#fb923c', '#fdba74', '#fed7aa'],       // Oranges
            ['#14b8a6', '#2dd4bf', '#5eead4', '#99f6e4'],       // Teals
            ['#ec4899', '#f472b6', '#f9a8d4', '#fbcfe8'],       // Pinks
            ['#eab308', '#facc15', '#fde047', '#fef08a']        // Yellows
        ]
    },
    
    /**
     * Create a Sankey diagram for flow visualization
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('SankeyChart: Container not found:', containerId);
            return null;
        }
        
        const theme = window.FinancialTheme || { 
            fonts: { primary: 'Inter, sans-serif' },
            colors: { revenue: '#22c55e', cost: '#ef4444' }
        };
        
        // Process nodes and links with premium coloring
        const { nodeColors, linkColors, nodeLabels } = this._processDataPremium(data, options);
        
        const trace = {
            type: 'sankey',
            orientation: 'h',
            arrangement: 'snap',
            node: {
                pad: 25,
                thickness: 30,
                line: {
                    color: 'rgba(15, 23, 42, 0.8)',
                    width: 2
                },
                label: nodeLabels,
                color: nodeColors,
                hovertemplate: 
                    '<b style="font-size:14px">%{label}</b><br>' +
                    '<span style="color:#10b981">💰 Total Flow:</span> $%{value:,.2f}<br>' +
                    '<extra></extra>',
                hoverlabel: {
                    bgcolor: 'rgba(30, 41, 59, 0.95)',
                    bordercolor: '#6366f1',
                    font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
                }
            },
            link: {
                source: data.links.map(l => l.source),
                target: data.links.map(l => l.target),
                value: data.links.map(l => l.value),
                color: linkColors,
                hovertemplate: 
                    '<b>%{source.label}</b> → <b>%{target.label}</b><br>' +
                    '<span style="color:#8b5cf6">💸 Amount:</span> $%{value:,.2f}<br>' +
                    '<extra></extra>',
                hoverlabel: {
                    bgcolor: 'rgba(30, 41, 59, 0.95)',
                    bordercolor: '#8b5cf6',
                    font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
                }
            }
        };
        
        const layout = {
            title: '',
            font: { 
                family: theme.fonts?.primary || 'Inter, sans-serif',
                size: 12,
                color: '#e2e8f0'
            },
            height: options.height || 450,
            margin: { l: 15, r: 15, t: 25, b: 15 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };
        
        const config = { 
            displayModeBar: false,
            responsive: true 
        };
        
        try {
            Plotly.newPlot(containerId, [trace], layout, config);
            console.log('SankeyChart rendered successfully');
        } catch (e) {
            console.error('SankeyChart failed:', e);
        }
        
        return { containerId, type: 'sankey' };
    },
    
    /**
     * Process Sankey data with premium gradient coloring
     */
    _processDataPremium(data, options) {
        const nodes = data.nodes || [];
        const links = data.links || [];
        const maxValue = Math.max(...links.map(l => l.value), 1);
        
        // Track which category index each node belongs to
        const nodeCategories = {};
        let categoryCounter = 0;
        
        // Assign colors based on node type/position
        const nodeColors = nodes.map((node, i) => {
            const nodeLower = (node || '').toLowerCase();
            
            // Revenue/Income sources - green gradient
            if (nodeLower.includes('revenue') || nodeLower.includes('income') || 
                nodeLower.includes('sales') || nodeLower.includes('inflow') ||
                nodeLower.includes('total inflow')) {
                nodeCategories[i] = 'inflow';
                return this.colorScales.inflow[0];
            }
            
            // Costs/Expenses - red gradient
            if (nodeLower.includes('cost') || nodeLower.includes('expense') || 
                nodeLower.includes('outflow') || nodeLower.includes('payment') ||
                nodeLower.includes('salaries') || nodeLower.includes('operations') ||
                nodeLower.includes('marketing')) {
                nodeCategories[i] = 'outflow';
                return this.colorScales.outflow[0];
            }
            
            // Central nodes - indigo
            if (nodeLower.includes('operating') || nodeLower.includes('total') ||
                nodeLower.includes('balance') || nodeLower.includes('net')) {
                nodeCategories[i] = 'neutral';
                return this.colorScales.neutral[0];
            }
            
            // Categories - cycle through color scales
            const catIndex = categoryCounter % this.colorScales.categories.length;
            nodeCategories[i] = catIndex;
            categoryCounter++;
            return this.colorScales.categories[catIndex][0];
        });
        
        // Link colors - gradient based on value intensity
        const linkColors = links.map(link => {
            const sourceCategory = nodeCategories[link.source];
            const intensity = Math.min(0.7, 0.3 + (link.value / maxValue) * 0.4);
            
            let baseColor;
            if (sourceCategory === 'inflow') {
                baseColor = this.colorScales.inflow[1];
            } else if (sourceCategory === 'outflow') {
                baseColor = this.colorScales.outflow[1];
            } else if (sourceCategory === 'neutral') {
                baseColor = this.colorScales.neutral[1];
            } else if (typeof sourceCategory === 'number') {
                baseColor = this.colorScales.categories[sourceCategory][1];
            } else {
                baseColor = '#6366f1';
            }
            
            // Convert to rgba with intensity
            const rgb = this._hexToRgb(baseColor);
            return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${intensity})`;
        });
        
        // Format node labels
        const nodeLabels = nodes.map(node => node || '');
        
        return { nodeColors, linkColors, nodeLabels };
    },
    
    /**
     * Helper: Hex to RGB
     */
    _hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 99, g: 102, b: 241 };
    },
    
    /**
     * Create cash flow Sankey with inflow/outflow differentiation
     */
    createCashFlow(containerId, cashFlowData, options = {}) {
        // Build nodes from inflows and outflows
        const nodes = ['Total Inflows'];
        const links = [];
        
        // Add inflow sources
        let inflowIndex = 1;
        const inflowStart = 1;
        if (cashFlowData.inflows) {
            Object.entries(cashFlowData.inflows).forEach(([source, value]) => {
                nodes.push(source);
                links.push({ source: inflowIndex, target: 0, value: Math.abs(value) });
                inflowIndex++;
            });
        }
        
        // Add central node
        nodes.push('Operating Cash');
        const centerIndex = nodes.length - 1;
        links.push({ source: 0, target: centerIndex, value: cashFlowData.totalInflows || 100000 });
        
        // Add outflow destinations
        if (cashFlowData.outflows) {
            Object.entries(cashFlowData.outflows).forEach(([dest, value]) => {
                const destIndex = nodes.length;
                nodes.push(dest);
                links.push({ source: centerIndex, target: destIndex, value: Math.abs(value) });
            });
        }
        
        // Add ending balance
        nodes.push('Ending Balance');
        const netFlow = (cashFlowData.totalInflows || 100000) - (cashFlowData.totalOutflows || 80000);
        if (netFlow > 0) {
            links.push({ source: centerIndex, target: nodes.length - 1, value: netFlow });
        }
        
        return this.create(containerId, {
            nodes,
            links,
            title: options.title || 'Cash Flow Analysis'
        }, options);
    },
    
    /**
     * Create spend allocation Sankey
     */
    createSpendAllocation(containerId, spendData, options = {}) {
        const nodes = ['Total Spend'];
        const links = [];
        
        // First level - categories
        let categoryIndex = 1;
        Object.entries(spendData.categories || {}).forEach(([category, categoryData]) => {
            nodes.push(category);
            links.push({ 
                source: 0, 
                target: categoryIndex, 
                value: categoryData.total || categoryData 
            });
            
            // Second level - subcategories if available
            if (categoryData.subcategories) {
                Object.entries(categoryData.subcategories).forEach(([sub, value]) => {
                    const subIndex = nodes.length;
                    nodes.push(sub);
                    links.push({ source: categoryIndex, target: subIndex, value });
                });
            }
            
            categoryIndex = nodes.length;
        });
        
        return this.create(containerId, {
            nodes,
            links,
            title: options.title || 'Spend Allocation'
        }, options);
    }
};


// =============================================================================
// COMBO FLOW CHART (Bar + Line)
// =============================================================================

const ComboFlowChart = {
    /**
     * Create bar + line combo for cash flow over time
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return null;
        
        const theme = window.FinancialTheme || FinancialTheme;
        const utils = window.FinancialUtils || FinancialUtils;
        
        // Bar trace for net flow
        const barTrace = {
            type: 'bar',
            name: data.barLabel || 'Net Cash Flow',
            x: data.labels,
            y: data.barValues,
            marker: {
                color: data.barValues.map(v => v >= 0 ? theme.colors.success : theme.colors.danger),
                line: {
                    color: data.barValues.map(v => v >= 0 ? theme.colors.success : theme.colors.danger),
                    width: 1
                }
            },
            hovertemplate: '<b>%{x}</b><br>Net Flow: %{y:$,.0f}<extra></extra>'
        };
        
        // Line trace for cumulative balance
        const lineTrace = {
            type: 'scatter',
            mode: 'lines+markers',
            name: data.lineLabel || 'Cash Balance',
            x: data.labels,
            y: data.lineValues,
            yaxis: 'y2',
            line: {
                color: theme.colors.primary,
                width: 3,
                shape: 'spline'
            },
            marker: {
                color: theme.colors.primary,
                size: 8,
                line: { color: '#1e293b', width: 2 }
            },
            hovertemplate: '<b>%{x}</b><br>Balance: %{y:$,.0f}<extra></extra>'
        };
        
        const layout = {
            ...PlotlyBaseLayout,
            title: {
                text: data.title || 'Cash Flow Over Time',
                font: { color: '#f8fafc', size: 16, family: theme.fonts.primary },
                x: 0.02,
                xanchor: 'left'
            },
            barmode: 'relative',
            showlegend: true,
            legend: {
                ...PlotlyBaseLayout.legend,
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.15
            },
            xaxis: {
                ...PlotlyBaseLayout.xaxis,
                title: { text: options.xLabel || '' }
            },
            yaxis: {
                ...PlotlyBaseLayout.yaxis,
                title: { text: 'Net Flow ($)', font: { color: '#94a3b8' } },
                tickformat: ',.0f',
                tickprefix: '$',
                side: 'left'
            },
            yaxis2: {
                ...PlotlyBaseLayout.yaxis,
                title: { text: 'Balance ($)', font: { color: theme.colors.primary } },
                tickformat: ',.0f',
                tickprefix: '$',
                side: 'right',
                overlaying: 'y',
                showgrid: false
            },
            height: options.height || 400,
            margin: { l: 80, r: 80, t: 60, b: 80 }
        };
        
        const config = { ...PlotlyBaseConfig };
        
        Plotly.newPlot(containerId, [barTrace, lineTrace], layout, config);
        
        return { containerId, type: 'combo' };
    },
    
    /**
     * Create from cash flow engine data
     */
    createFromCashFlowEngine(containerId, engineResult, options = {}) {
        // Extract graph data from engine result
        const graphData = engineResult.graphs?.find(g => g.type === 'bar_line_combo') || {};
        
        return this.create(containerId, {
            labels: graphData.x_data || engineResult.periods || [],
            barValues: graphData.bar_data || engineResult.net_flows || [],
            lineValues: graphData.line_data || engineResult.cumulative_balance || [],
            barLabel: graphData.bar_label || 'Net Flow',
            lineLabel: graphData.line_label || 'Cash Balance',
            title: options.title || 'Cash Flow Analysis'
        }, options);
    }
};


// =============================================================================
// AREA CHART FOR FORECASTING
// =============================================================================

const ForecastAreaChart = {
    // Premium gradient colors
    gradients: {
        historical: ['#3b82f6', '#2563eb'],      // Blue for historical data
        forecast: ['#a855f7', '#9333ea'],         // Purple for forecast
        confidence: {
            upper: 'rgba(168, 85, 247, 0.25)',
            lower: 'rgba(168, 85, 247, 0.08)'
        }
    },
    
    /**
     * Create forecast area chart with confidence bands
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('ForecastAreaChart: Container not found:', containerId);
            return null;
        }
        
        const traces = [];
        
        // Historical line with gradient effect
        if (data.historical) {
            traces.push({
                type: 'scatter',
                mode: 'lines+markers',
                name: '📊 Historical',
                x: data.historical.x,
                y: data.historical.y,
                line: {
                    color: this.gradients.historical[0],
                    width: 2.5,
                    shape: 'spline'
                },
                marker: {
                    color: this.gradients.historical[1],
                    size: 7,
                    line: { color: '#1e293b', width: 2 }
                },
                hovertemplate: '<b>%{x}</b><br>' +
                    '<span style="color:#3b82f6">📊 Actual:</span> $%{y:,.0f}<extra></extra>',
                hoverlabel: {
                    bgcolor: 'rgba(30, 41, 59, 0.95)',
                    bordercolor: '#3b82f6',
                    font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
                }
            });
        }
        
        // Confidence band (upper) - invisible line for fill reference
        if (data.forecast?.upper) {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                name: 'Upper Bound',
                x: data.forecast.x,
                y: data.forecast.upper,
                line: { color: 'rgba(168, 85, 247, 0.4)', width: 1 },
                showlegend: false,
                hoverinfo: 'skip'
            });
        }
        
        // Confidence band (lower) with gradient fill
        if (data.forecast?.lower) {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                name: '🎯 Confidence',
                x: data.forecast.x,
                y: data.forecast.lower,
                fill: 'tonexty',
                fillcolor: this.gradients.confidence.upper,
                line: { color: 'rgba(168, 85, 247, 0.4)', width: 1 },
                hovertemplate: '<b>%{x}</b><br>' +
                    '<span style="color:#a855f7">📈 Range:</span> $%{y:,.0f}<extra></extra>',
                hoverlabel: {
                    bgcolor: 'rgba(30, 41, 59, 0.95)',
                    bordercolor: '#a855f7',
                    font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
                }
            });
        }
        
        // Forecast line with premium styling
        if (data.forecast) {
            traces.push({
                type: 'scatter',
                mode: 'lines+markers',
                name: '🔮 Forecast',
                x: data.forecast.x,
                y: data.forecast.y,
                line: {
                    color: this.gradients.forecast[0],
                    width: 3,
                    dash: 'dash',
                    shape: 'spline'
                },
                marker: {
                    color: this.gradients.forecast[1],
                    size: 9,
                    symbol: 'diamond',
                    line: { color: '#1e293b', width: 2 }
                },
                hovertemplate: '<b>%{x}</b><br>' +
                    '<span style="color:#a855f7">🔮 Forecast:</span> $%{y:,.0f}<extra></extra>',
                hoverlabel: {
                    bgcolor: 'rgba(30, 41, 59, 0.95)',
                    bordercolor: '#a855f7',
                    font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
                }
            });
        }
        
        const layout = {
            title: '',
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.12,
                font: { color: '#e2e8f0', size: 11 },
                bgcolor: 'rgba(0,0,0,0)'
            },
            font: {
                family: 'Inter, sans-serif',
                color: '#e2e8f0'
            },
            xaxis: {
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                linecolor: 'rgba(148, 163, 184, 0.2)',
                tickfont: { color: '#94a3b8', size: 11 },
                title: { text: options.xLabel || '', font: { color: '#94a3b8' } }
            },
            yaxis: {
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                linecolor: 'rgba(148, 163, 184, 0.2)',
                tickfont: { color: '#94a3b8', size: 11 },
                title: { text: options.yLabel || '', font: { color: '#94a3b8' } },
                tickformat: '$,.0f'
            },
            height: options.height || 400,
            margin: { l: 75, r: 40, t: 30, b: 70 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            // Add forecast start annotation with glow
            shapes: data.forecastStart ? [{
                type: 'line',
                x0: data.forecastStart,
                x1: data.forecastStart,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {
                    color: 'rgba(168, 85, 247, 0.7)',
                    width: 2,
                    dash: 'dot'
                }
            }] : [],
            annotations: data.forecastStart ? [{
                x: data.forecastStart,
                y: 1,
                yref: 'paper',
                text: '🔮 Forecast Start',
                showarrow: false,
                font: { size: 10, color: '#a855f7' },
                yshift: 10
            }] : []
        };
        
        const config = { displayModeBar: false, responsive: true };
        
        try {
            Plotly.newPlot(containerId, traces, layout, config);
            console.log('ForecastAreaChart rendered with premium styling');
        } catch (e) {
            console.error('ForecastAreaChart failed:', e);
        }
        
        return { containerId, type: 'forecastArea' };
    }
};


// =============================================================================
// EXPORTS
// =============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        WaterfallChart,
        SankeyChart,
        ComboFlowChart,
        ForecastAreaChart
    };
}

if (typeof window !== 'undefined') {
    window.WaterfallChart = WaterfallChart;
    window.SankeyChart = SankeyChart;
    window.ComboFlowChart = ComboFlowChart;
    window.ForecastAreaChart = ForecastAreaChart;
}
