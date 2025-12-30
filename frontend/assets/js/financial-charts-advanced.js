/**
 * Financial Advanced Charts Module
 * 
 * Premium Treemaps, Tornado charts, Network graphs, and 3D surfaces.
 * Treemaps: Cost composition, inventory ABC, market share
 * Tornado: Sensitivity analysis, risk factors
 * Network: Market basket associations, supplier relationships
 * 3D Surface: Pricing optimization, multi-dimensional analysis
 * 
 * @module FinancialChartsAdvanced
 * @requires FinancialChartsCore
 * @requires Plotly.js 2.27+
 * @requires ECharts 5.4+
 */

'use strict';

// Safe references to core dependencies (handle load order)
const _getTheme = () => window.FinancialTheme || (typeof FinancialTheme !== 'undefined' ? FinancialTheme : {
    colors: { primary: '#6366f1', secondary: '#8b5cf6', success: '#10b981', danger: '#ef4444', neutral: '#64748b' },
    palettes: { financial: ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e', '#f97316', '#eab308', '#22c55e', '#14b8a6', '#06b6d4', '#3b82f6'] },
    fonts: { primary: 'Inter, sans-serif' }
});

const _getPlotlyLayout = () => window.PlotlyBaseLayout || (typeof PlotlyBaseLayout !== 'undefined' ? PlotlyBaseLayout : {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', size: 12, color: '#e2e8f0' },
    margin: { l: 60, r: 30, t: 40, b: 60, pad: 4 }
});

const _getPlotlyConfig = () => window.PlotlyBaseConfig || (typeof PlotlyBaseConfig !== 'undefined' ? PlotlyBaseConfig : {
    displayModeBar: false,
    responsive: true
});

const _isLowPower = () => Boolean(window.NexusPerformance?.lowPower);

// =============================================================================
// TREEMAP CHART (Plotly)
// =============================================================================

const TreemapChart = {
    /**
     * Create treemap for hierarchical data visualization
     * 
     * @param {string} containerId - DOM container ID
     * @param {object} data - Chart data
     * @param {string[]} data.labels - All labels including parent paths
     * @param {string[]} data.parents - Parent for each label
     * @param {number[]} data.values - Values for sizing
     * @param {string} data.title - Chart title
     * @param {object} options - Additional options
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('TreemapChart: Container not found:', containerId);
            return null;
        }
        
        const theme = _getTheme();
        const baseLayout = _getPlotlyLayout();
        const baseConfig = _getPlotlyConfig();
        
        // Generate colors based on hierarchy level and value
        const colors = this._generateColors(data, theme);
        
        console.log('TreemapChart creating with:', { containerId, labels: data.labels, values: data.values });
        
        const trace = {
            type: 'treemap',
            labels: data.labels,
            parents: data.parents,
            values: data.values,
            textinfo: 'label+percent parent+value',
            texttemplate: '<b>%{label}</b><br>%{percentParent:.1%}<br>$%{value:,.0f}',
            textfont: {
                family: theme.fonts?.primary || 'Inter, sans-serif',
                size: [14, 13, 12, 11],  // Larger text for bigger boxes
                color: '#ffffff'
            },
            insidetextfont: {
                family: theme.fonts?.primary || 'Inter, sans-serif',
                size: [14, 13, 12, 11],
                color: '#ffffff'
            },
            outsidetextfont: {
                family: theme.fonts?.primary || 'Inter, sans-serif',
                size: 11,
                color: '#94a3b8'
            },
            marker: {
                colors: colors,
                line: {
                    width: 2,
                    color: 'rgba(15, 23, 42, 0.8)'
                },
                pad: { t: 30, l: 6, r: 6, b: 6 },
                cornerradius: 5
            },
            hovertemplate: '<b style="font-size:14px">%{label}</b><br>' +
                          '<span style="color:#10b981">💰 Value:</span> $%{value:,.2f}<br>' +
                          '<span style="color:#8b5cf6">📊 Share:</span> %{percentRoot:.2%} of total<br>' +
                          '<span style="color:#f59e0b">📁 Parent:</span> %{parent}<extra></extra>',
            hoverlabel: {
                bgcolor: 'rgba(30, 41, 59, 0.95)',
                bordercolor: '#6366f1',
                font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
            },
            pathbar: {
                visible: true,
                side: 'top',
                edgeshape: '>',
                thickness: 28,
                textfont: {
                    family: theme.fonts?.primary || 'Inter, sans-serif',
                    size: 13,
                    color: '#f1f5f9'
                }
            },
            branchvalues: 'total',
            maxdepth: options.maxDepth || 3,
            tiling: {
                packing: 'squarify',
                squarifyratio: 1,
                pad: 3
            }
        };
        
        const layout = {
            ...baseLayout,
            title: {
                text: '',  // Remove title - it's in the card header
                font: { color: '#f8fafc', size: 16, family: theme.fonts?.primary || 'Inter, sans-serif' },
                x: 0.02,
                xanchor: 'left'
            },
            height: options.height || 400,
            margin: { l: 5, r: 5, t: 35, b: 5 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };
        
        const config = { 
            ...baseConfig,
            displayModeBar: false,
            responsive: true
        };
        
        try {
            Plotly.newPlot(containerId, [trace], layout, config);
            console.log('TreemapChart rendered successfully');
        } catch (e) {
            console.error('Plotly.newPlot failed:', e);
        }
        
        return { containerId, type: 'treemap' };
    },
    
    /**
     * Generate vibrant colors for treemap nodes based on value
     */
    _generateColors(data, theme) {
        // Premium color palette - vibrant gradients
        const colorScales = [
            // Blues to Cyans
            ['#0ea5e9', '#06b6d4', '#14b8a6', '#10b981'],
            // Purples to Pinks  
            ['#8b5cf6', '#a855f7', '#d946ef', '#ec4899'],
            // Oranges to Yellows
            ['#f97316', '#fb923c', '#fbbf24', '#facc15'],
            // Greens
            ['#22c55e', '#4ade80', '#86efac', '#a7f3d0'],
            // Reds to Roses
            ['#ef4444', '#f87171', '#fb7185', '#fda4af'],
            // Indigos
            ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe']
        ];
        
        const values = data.values;
        const maxValue = Math.max(...values.filter(v => v > 0));
        const minValue = Math.min(...values.filter(v => v > 0));
        
        // Track parent colors for children
        const parentColorMap = {};
        let categoryIndex = 0;
        
        return values.map((value, i) => {
            const label = data.labels[i];
            const parent = data.parents[i];
            
            // Root node - dark background
            if (parent === '') {
                return 'rgba(30, 41, 59, 0.8)';
            }
            
            // Category nodes (direct children of root)
            const isCategory = data.parents.indexOf(label) !== -1 || 
                              data.labels.filter((_, j) => data.parents[j] === label).length > 0;
            
            if (data.parents[i] === data.labels[0]) {
                // This is a top-level category
                const scale = colorScales[categoryIndex % colorScales.length];
                parentColorMap[label] = scale;
                categoryIndex++;
                return scale[0]; // Base color for category
            }
            
            // Leaf nodes - color based on value intensity within parent's scale
            const parentScale = parentColorMap[parent] || colorScales[0];
            const normalizedValue = maxValue > minValue 
                ? (value - minValue) / (maxValue - minValue) 
                : 0.5;
            
            // Pick color from scale based on value (higher value = more intense)
            const colorIndex = Math.min(
                Math.floor(normalizedValue * (parentScale.length - 1)),
                parentScale.length - 1
            );
            
            return parentScale[colorIndex];
        });
    },
    
    /**
     * Create from flat data with categories
     */
    createFromCategories(containerId, categories, options = {}) {
        const labels = [options.rootLabel || 'Total'];
        const parents = [''];
        const values = [0];
        
        let totalValue = 0;
        
        Object.entries(categories).forEach(([category, categoryData]) => {
            if (typeof categoryData === 'number') {
                // Simple category -> value
                labels.push(category);
                parents.push(options.rootLabel || 'Total');
                values.push(categoryData);
                totalValue += categoryData;
            } else if (typeof categoryData === 'object' && categoryData !== null) {
                // Category with subcategories - handle both formats:
                // { subcategories: {...} } OR direct { subcat: value, ... }
                const subcats = categoryData.subcategories || categoryData;
                
                // First, calculate the category total
                let categoryTotal = 0;
                const childEntries = [];
                Object.entries(subcats).forEach(([sub, value]) => {
                    if (typeof value === 'number') {
                        childEntries.push({ name: sub, value: value });
                        categoryTotal += value;
                    }
                });
                
                // Add category with its total value FIRST
                labels.push(category);
                parents.push(options.rootLabel || 'Total');
                values.push(categoryTotal);  // Parent value = sum of children
                
                // Then add all children
                childEntries.forEach(({ name, value }) => {
                    labels.push(name);
                    parents.push(category);
                    values.push(value);
                });
                
                totalValue += categoryTotal;
            }
        });
        
        // Update root value
        values[0] = totalValue;
        
        console.log('Treemap data built:', { 
            labels: labels.slice(0, 5), 
            parents: parents.slice(0, 5), 
            values: values.slice(0, 5),
            totalValue 
        });
        
        return this.create(containerId, {
            labels,
            parents,
            values,
            title: options.title
        }, options);
    },
    
    /**
     * Create cost composition treemap
     */
    createCostComposition(containerId, costData, options = {}) {
        return this.createFromCategories(containerId, costData, {
            rootLabel: 'Total Costs',
            title: options.title || 'Cost Composition',
            ...options
        });
    },
    
    /**
     * Create inventory ABC treemap
     */
    createInventoryABC(containerId, inventoryData, options = {}) {
        const categories = {
            'Class A (80%)': { subcategories: inventoryData.classA || {} },
            'Class B (15%)': { subcategories: inventoryData.classB || {} },
            'Class C (5%)': { subcategories: inventoryData.classC || {} }
        };
        
        return this.createFromCategories(containerId, categories, {
            rootLabel: 'Inventory',
            title: options.title || 'Inventory ABC Analysis',
            ...options
        });
    }
};


// =============================================================================
// TORNADO CHART (Plotly)
// =============================================================================

const TornadoChart = {
    // Premium gradient colors for impact bars
    gradients: {
        negative: ['#991b1b', '#dc2626', '#ef4444', '#f87171'],  // Red gradient for low scenario
        positive: ['#065f46', '#059669', '#10b981', '#34d399'],  // Emerald gradient for high scenario
        neutral: ['#3730a3', '#4f46e5', '#6366f1', '#818cf8']    // Indigo for emphasis
    },
    
    /**
     * Create tornado chart for sensitivity analysis
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('TornadoChart: Container not found:', containerId);
            return null;
        }
        
        const theme = window.FinancialTheme || { 
            fonts: { primary: 'Inter, sans-serif' },
            colors: { success: '#10b981', danger: '#ef4444' }
        };
        
        // Sort by total impact (largest first)
        const sorted = this._sortByImpact(data);
        
        // Calculate intensity for each bar
        const maxRange = Math.max(...sorted.lowImpact.map((l, i) => 
            Math.abs(l) + Math.abs(sorted.highImpact[i])
        ));
        
        // Generate intensity-based colors
        const lowColors = sorted.lowImpact.map((val, i) => {
            const intensity = Math.abs(val) / Math.max(...sorted.lowImpact.map(Math.abs));
            return this._getGradientColor(this.gradients.negative, intensity);
        });
        
        const highColors = sorted.highImpact.map((val, i) => {
            const intensity = Math.abs(val) / Math.max(...sorted.highImpact.map(Math.abs));
            return this._getGradientColor(this.gradients.positive, intensity);
        });
        
        // Low impact bars (left side, negative)
        const lowTrace = {
            type: 'bar',
            orientation: 'h',
            name: '📉 Low Scenario',
            y: sorted.factors,
            x: sorted.lowImpact,
            marker: {
                color: lowColors,
                line: { color: this.gradients.negative[0], width: 1.5 }
            },
            hovertemplate: '<b>%{y}</b><br>' +
                '<span style="color:#ef4444">📉 Low Impact:</span> %{x:$,.0f}<extra></extra>',
            hoverlabel: {
                bgcolor: 'rgba(30, 41, 59, 0.95)',
                bordercolor: '#ef4444',
                font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
            }
        };
        
        // High impact bars (right side, positive)
        const highTrace = {
            type: 'bar',
            orientation: 'h',
            name: '📈 High Scenario',
            y: sorted.factors,
            x: sorted.highImpact,
            marker: {
                color: highColors,
                line: { color: this.gradients.positive[0], width: 1.5 }
            },
            hovertemplate: '<b>%{y}</b><br>' +
                '<span style="color:#10b981">📈 High Impact:</span> +%{x:$,.0f}<extra></extra>',
            hoverlabel: {
                bgcolor: 'rgba(30, 41, 59, 0.95)',
                bordercolor: '#10b981',
                font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
            }
        };
        
        // Add range annotations
        const annotations = sorted.factors.map((factor, i) => {
            const range = Math.abs(sorted.highImpact[i]) + Math.abs(sorted.lowImpact[i]);
            return {
                x: sorted.highImpact[i] + Math.abs(sorted.highImpact[i]) * 0.15,
                y: factor,
                text: `Δ$${(range / 1000).toFixed(0)}K`,
                showarrow: false,
                font: { size: 10, color: '#94a3b8' },
                xanchor: 'left'
            };
        });
        
        const layout = {
            title: '',
            barmode: 'overlay',
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.12,
                font: { color: '#e2e8f0', size: 12 },
                bgcolor: 'rgba(0,0,0,0)'
            },
            font: {
                family: theme.fonts?.primary || 'Inter, sans-serif',
                color: '#e2e8f0'
            },
            xaxis: {
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                linecolor: 'rgba(148, 163, 184, 0.2)',
                tickfont: { color: '#94a3b8', size: 11 },
                zeroline: true,
                zerolinecolor: '#6366f1',
                zerolinewidth: 3,
                tickformat: '$,.0f'
            },
            yaxis: {
                gridcolor: 'rgba(148, 163, 184, 0.05)',
                linecolor: 'rgba(148, 163, 184, 0.2)',
                tickfont: { color: '#e2e8f0', size: 11 },
                automargin: true,
                categoryorder: 'array',
                categoryarray: sorted.factors.slice().reverse()
            },
            height: options.height || Math.max(320, sorted.factors.length * 55 + 100),
            margin: { l: 140, r: 60, t: 30, b: 70 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            bargap: 0.25,
            annotations: options.showRange !== false ? annotations : []
        };
        
        const config = { displayModeBar: false, responsive: true };
        
        try {
            Plotly.newPlot(containerId, [lowTrace, highTrace], layout, config);
            console.log('TornadoChart rendered with premium styling');
        } catch (e) {
            console.error('TornadoChart failed:', e);
        }
        
        return { containerId, type: 'tornado' };
    },
    
    /**
     * Get gradient color based on intensity
     */
    _getGradientColor(gradient, intensity) {
        const idx = Math.min(Math.floor(intensity * (gradient.length - 1)), gradient.length - 1);
        return gradient[idx];
    },
    
    /**
     * Sort factors by total impact
     */
    _sortByImpact(data) {
        const combined = data.factors.map((factor, i) => ({
            factor,
            low: data.lowImpact[i],
            high: data.highImpact[i],
            range: Math.abs(data.highImpact[i]) + Math.abs(data.lowImpact[i])
        }));
        
        combined.sort((a, b) => b.range - a.range);
        
        return {
            factors: combined.map(c => c.factor),
            lowImpact: combined.map(c => c.low),
            highImpact: combined.map(c => c.high)
        };
    },
    
    /**
     * Create ROI sensitivity analysis
     */
    createROISensitivity(containerId, factors, baselineROI, options = {}) {
        // Calculate impacts based on ±10% change
        const sensitivity = factors.map(f => ({
            name: f.name,
            lowImpact: baselineROI * (f.elasticity * -0.1),
            highImpact: baselineROI * (f.elasticity * 0.1)
        }));
        
        return this.create(containerId, {
            factors: sensitivity.map(s => s.name),
            lowImpact: sensitivity.map(s => s.lowImpact),
            highImpact: sensitivity.map(s => s.highImpact),
            title: options.title || 'ROI Sensitivity (±10% Change)'
        }, options);
    },
    
    /**
     * Create pricing elasticity tornado
     */
    createPricingImpact(containerId, products, options = {}) {
        return this.create(containerId, {
            factors: products.map(p => p.name),
            lowImpact: products.map(p => p.revenueAt10PercentLower - p.currentRevenue),
            highImpact: products.map(p => p.revenueAt10PercentHigher - p.currentRevenue),
            title: options.title || 'Price Change Impact (±10%)'
        }, options);
    }
};


// =============================================================================
// NETWORK GRAPH (ECharts) - Premium Styled
// =============================================================================

const NetworkGraph = {
    // Premium color palettes for nodes
    nodeColors: {
        primary: ['#3b82f6', '#2563eb', '#1d4ed8', '#1e40af'],    // Blue
        secondary: ['#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6'],  // Violet
        success: ['#10b981', '#059669', '#047857', '#065f46'],    // Emerald
        warning: ['#f59e0b', '#d97706', '#b45309', '#92400e'],    // Amber
        info: ['#06b6d4', '#0891b2', '#0e7490', '#155e75']        // Cyan
    },
    
    // Link gradient colors
    linkGradients: [
        ['rgba(99, 102, 241, 0.7)', 'rgba(139, 92, 246, 0.7)'],   // Indigo → Violet
        ['rgba(59, 130, 246, 0.7)', 'rgba(99, 102, 241, 0.7)'],   // Blue → Indigo
        ['rgba(16, 185, 129, 0.6)', 'rgba(6, 182, 212, 0.6)']     // Emerald → Cyan
    ],
    
    /**
     * Create network graph for association visualization
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('NetworkGraph: Container not found:', containerId);
            return null;
        }

        const lowPower = _isLowPower();
        let nodesInput = Array.isArray(data.nodes) ? data.nodes : [];
        let linksInput = Array.isArray(data.links) ? data.links : [];

        if (lowPower && nodesInput.length > 80) {
            nodesInput = [...nodesInput]
                .sort((a, b) => (b.value || 1) - (a.value || 1))
                .slice(0, 80);
            const nodeIds = new Set(nodesInput.map(node => node.id || node.name));
            linksInput = linksInput.filter(link => nodeIds.has(link.source) && nodeIds.has(link.target));
            if (linksInput.length > 120) {
                linksInput = linksInput.slice(0, 120);
            }
        }

        // Initialize ECharts
        const chart = echarts.init(container, null, { renderer: 'canvas' });
        
        // Process nodes with premium styling
        const { nodes, maxNodeValue } = this._processNodes(nodesInput, options);
        
        // Process links with premium gradient styling
        const links = this._processLinks(linksInput);
        
        const option = {
            backgroundColor: 'transparent',
            title: {
                text: '',
                left: 'left',
                textStyle: {
                    color: '#f8fafc',
                    fontSize: 16,
                    fontFamily: 'Inter, sans-serif'
                }
            },
            tooltip: {
                backgroundColor: 'rgba(30, 41, 59, 0.95)',
                borderColor: '#6366f1',
                borderWidth: 1,
                textStyle: {
                    color: '#f1f5f9',
                    fontFamily: 'Inter, sans-serif'
                },
                formatter: (params) => {
                    if (params.dataType === 'node') {
                        const intensity = (params.value / maxNodeValue * 100).toFixed(0);
                        return `<b style="font-size:14px">${params.name}</b><br>` +
                            `<span style="color:#8b5cf6">💎 Value:</span> ${params.value.toLocaleString()}<br>` +
                            `<span style="color:#94a3b8">Intensity:</span> ${intensity}%`;
                    } else if (params.dataType === 'edge') {
                        return `<b style="font-size:13px">${params.data.source}</b> → <b>${params.data.target}</b><br>` +
                            `<span style="color:#6366f1">🔗 Strength:</span> ${params.data.value.toFixed(2)}`;
                    }
                    return '';
                }
            },
            animationDuration: lowPower ? 0 : 1500,
            animationEasingUpdate: lowPower ? 'linear' : 'quinticInOut',
            series: [{
                type: 'graph',
                layout: lowPower ? 'circular' : 'force',
                data: nodes,
                links: links,
                roam: !lowPower,
                draggable: !lowPower,
                force: lowPower ? undefined : {
                    repulsion: 350,
                    gravity: 0.08,
                    edgeLength: [60, 220],
                    layoutAnimation: true
                },
                emphasis: {
                    focus: 'adjacency',
                    itemStyle: {
                        shadowBlur: 20,
                        shadowColor: 'rgba(99, 102, 241, 0.6)'
                    },
                    lineStyle: {
                        width: 8
                    }
                },
                lineStyle: {
                    opacity: 0.8
                }
            }]
        };
        
        chart.setOption(option);
        
        window.addEventListener('resize', () => chart.resize());
        
        console.log('NetworkGraph rendered with premium styling');
        return chart;
    },
    
    /**
     * Process nodes with premium styling
     */
    _processNodes(nodesData, options) {
        const maxNodeValue = Math.max(...nodesData.map(n => n.value || 1));
        const colorPalettes = Object.values(this.nodeColors);
        
        const nodes = nodesData.map((node, i) => {
            const intensity = (node.value || 1) / maxNodeValue;
            const palette = colorPalettes[i % colorPalettes.length];
            const colorIdx = Math.min(Math.floor(intensity * (palette.length - 1)), palette.length - 1);
            
            return {
                id: node.id || node.name,
                name: node.name,
                value: node.value || 1,
                symbolSize: 25 + intensity * 45,
                category: node.category || 0,
                itemStyle: {
                    color: {
                        type: 'radial',
                        x: 0.5, y: 0.5, r: 0.5,
                        colorStops: [
                            { offset: 0, color: palette[colorIdx] },
                            { offset: 0.7, color: palette[Math.max(0, colorIdx - 1)] || palette[0] },
                            { offset: 1, color: 'rgba(30, 41, 59, 0.8)' }
                        ]
                    },
                    borderColor: palette[0],
                    borderWidth: intensity > 0.5 ? 3 : 2,
                    shadowColor: palette[colorIdx],
                    shadowBlur: intensity > 0.7 ? 15 : 8
                },
                label: {
                    show: true,
                    position: 'right',
                    color: '#e2e8f0',
                    fontSize: 11,
                    fontFamily: 'Inter, sans-serif',
                    formatter: '{b}'
                }
            };
        });
        
        return { nodes, maxNodeValue };
    },
    
    /**
     * Process links with premium gradient styling
     */
    _processLinks(linksData) {
        const maxLinkValue = Math.max(...linksData.map(l => l.value || 1));
        
        return linksData.map((link, i) => {
            const intensity = (link.value || 1) / maxLinkValue;
            const gradient = this.linkGradients[i % this.linkGradients.length];
            
            return {
                source: link.source,
                target: link.target,
                value: link.value || 1,
                lineStyle: {
                    width: 1.5 + intensity * 6,
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 1, y2: 0,
                        colorStops: [
                            { offset: 0, color: gradient[0] },
                            { offset: 1, color: gradient[1] }
                        ]
                    },
                    curveness: 0.25,
                    opacity: 0.4 + intensity * 0.5
                }
            };
        });
    },
    
    /**
     * Create market basket association network
     */
    createMarketBasket(containerId, rules, options = {}) {
        // Extract unique items
        const itemSet = new Set();
        rules.forEach(rule => {
            if (Array.isArray(rule.antecedent)) {
                rule.antecedent.forEach(item => itemSet.add(item));
            } else {
                itemSet.add(rule.antecedent);
            }
            if (Array.isArray(rule.consequent)) {
                rule.consequent.forEach(item => itemSet.add(item));
            } else {
                itemSet.add(rule.consequent);
            }
        });
        
        const items = Array.from(itemSet);
        
        // Build nodes with support as value
        const nodes = items.map(item => ({
            id: item,
            name: item,
            value: rules.filter(r => 
                (Array.isArray(r.antecedent) ? r.antecedent.includes(item) : r.antecedent === item) ||
                (Array.isArray(r.consequent) ? r.consequent.includes(item) : r.consequent === item)
            ).reduce((sum, r) => sum + (r.support || 0.1), 0)
        }));
        
        // Build links from rules
        const links = rules.map(rule => {
            const source = Array.isArray(rule.antecedent) ? rule.antecedent[0] : rule.antecedent;
            const target = Array.isArray(rule.consequent) ? rule.consequent[0] : rule.consequent;
            return {
                source,
                target,
                value: rule.lift || rule.confidence || 1
            };
        });
        
        return this.create(containerId, {
            nodes,
            links,
            title: options.title || 'Product Associations'
        }, options);
    },
    
    /**
     * Create supplier relationship network
     */
    createSupplierNetwork(containerId, suppliers, options = {}) {
        const nodes = [
            { id: 'company', name: 'Your Company', value: 100, category: 0 }
        ];
        
        const links = [];
        
        suppliers.forEach(supplier => {
            nodes.push({
                id: supplier.id || supplier.name,
                name: supplier.name,
                value: supplier.spend || supplier.value || 10,
                category: 1
            });
            
            links.push({
                source: supplier.id || supplier.name,
                target: 'company',
                value: supplier.spend || supplier.value || 10
            });
        });
        
        return this.create(containerId, {
            nodes,
            links,
            title: options.title || 'Supplier Network'
        }, options);
    }
};


// =============================================================================
// 3D SURFACE CHART (Plotly)
// =============================================================================

const SurfaceChart = {
    /**
     * Create 3D surface for multi-dimensional analysis
     * 
     * @param {string} containerId - DOM container ID
     * @param {object} data - Chart data
     * @param {number[][]} data.z - 2D array of z values
     * @param {number[]} data.x - X axis values
     * @param {number[]} data.y - Y axis values
     * @param {string} data.title - Chart title
     * @param {object} options - Additional options
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return null;
        
        const theme = window.FinancialTheme || FinancialTheme;
        const baseLayout = _getPlotlyLayout();
        const baseConfig = _getPlotlyConfig();
        const lowPower = _isLowPower();

        if (lowPower) {
            const heatmapTrace = {
                type: 'heatmap',
                z: data.z,
                x: data.x,
                y: data.y,
                colorscale: [
                    [0, theme.colors.danger],
                    [0.25, theme.colors.warning],
                    [0.5, theme.colors.info],
                    [0.75, theme.colors.primary],
                    [1, theme.colors.success]
                ],
                hovertemplate: `${options.xLabel || 'X'}: %{x}<br>${options.yLabel || 'Y'}: %{y}<br>${options.zLabel || 'Z'}: %{z:$,.0f}<extra></extra>`
            };

            const heatmapLayout = {
                ...baseLayout,
                title: {
                    text: `${data.title || 'Pricing Surface'} • Low-Graphics Mode`,
                    font: { color: '#f8fafc', size: 16, family: theme.fonts.primary },
                    x: 0.02,
                    xanchor: 'left'
                },
                xaxis: {
                    ...baseLayout.xaxis,
                    title: { text: options.xLabel || 'X Axis', font: { color: '#94a3b8' } }
                },
                yaxis: {
                    ...baseLayout.yaxis,
                    title: { text: options.yLabel || 'Y Axis', font: { color: '#94a3b8' } }
                },
                height: options.height || 400,
                margin: { l: 60, r: 30, t: 60, b: 60 }
            };

            Plotly.newPlot(containerId, [heatmapTrace], heatmapLayout, baseConfig);
            return { containerId, type: 'heatmap' };
        }

        const trace = {
            type: 'surface',
            z: data.z,
            x: data.x,
            y: data.y,
            colorscale: [
                [0, theme.colors.danger],
                [0.25, theme.colors.warning],
                [0.5, theme.colors.info],
                [0.75, theme.colors.primary],
                [1, theme.colors.success]
            ],
            contours: {
                z: {
                    show: true,
                    usecolormap: true,
                    highlightcolor: '#ffffff',
                    project: { z: true }
                }
            },
            hovertemplate: `${options.xLabel || 'X'}: %{x}<br>${options.yLabel || 'Y'}: %{y}<br>${options.zLabel || 'Z'}: %{z:$,.0f}<extra></extra>`
        };
        
        const layout = {
            ...baseLayout,
            title: {
                text: data.title || '3D Analysis',
                font: { color: '#f8fafc', size: 16, family: theme.fonts.primary },
                x: 0.02,
                xanchor: 'left'
            },
            scene: {
                xaxis: {
                    title: { text: options.xLabel || 'X Axis', font: { color: '#94a3b8' } },
                    gridcolor: 'rgba(148, 163, 184, 0.2)',
                    tickfont: { color: '#94a3b8' },
                    backgroundcolor: 'rgba(0,0,0,0)'
                },
                yaxis: {
                    title: { text: options.yLabel || 'Y Axis', font: { color: '#94a3b8' } },
                    gridcolor: 'rgba(148, 163, 184, 0.2)',
                    tickfont: { color: '#94a3b8' },
                    backgroundcolor: 'rgba(0,0,0,0)'
                },
                zaxis: {
                    title: { text: options.zLabel || 'Z Axis', font: { color: '#94a3b8' } },
                    gridcolor: 'rgba(148, 163, 184, 0.2)',
                    tickfont: { color: '#94a3b8' },
                    backgroundcolor: 'rgba(0,0,0,0)'
                },
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.2 }
                },
                bgcolor: 'rgba(0,0,0,0)'
            },
            height: options.height || 500,
            margin: { l: 20, r: 20, t: 60, b: 20 }
        };
        
        const config = { 
            ...baseConfig,
            displayModeBar: true,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud']
        };
        
        Plotly.newPlot(containerId, [trace], layout, config);
        
        return { containerId, type: 'surface' };
    },
    
    /**
     * Create pricing optimization surface
     */
    createPricingOptimization(containerId, priceRange, quantityRange, revenueMatrix, options = {}) {
        return this.create(containerId, {
            z: revenueMatrix,
            x: priceRange,
            y: quantityRange,
            title: options.title || 'Pricing Optimization Surface'
        }, {
            xLabel: 'Price ($)',
            yLabel: 'Quantity',
            zLabel: 'Revenue ($)',
            ...options
        });
    },
    
    /**
     * Generate sample pricing surface from elasticity
     */
    createFromElasticity(containerId, basePrice, baseQuantity, elasticity, options = {}) {
        const priceSteps = 20;
        const quantitySteps = 20;
        
        const priceMin = basePrice * 0.5;
        const priceMax = basePrice * 1.5;
        const priceRange = [];
        for (let i = 0; i < priceSteps; i++) {
            priceRange.push(priceMin + (priceMax - priceMin) * (i / (priceSteps - 1)));
        }
        
        const quantityMin = baseQuantity * 0.5;
        const quantityMax = baseQuantity * 1.5;
        const quantityRange = [];
        for (let i = 0; i < quantitySteps; i++) {
            quantityRange.push(quantityMin + (quantityMax - quantityMin) * (i / (quantitySteps - 1)));
        }
        
        // Calculate revenue matrix
        const revenueMatrix = [];
        for (let qi = 0; qi < quantitySteps; qi++) {
            const row = [];
            for (let pi = 0; pi < priceSteps; pi++) {
                const price = priceRange[pi];
                const priceChange = (price - basePrice) / basePrice;
                const demandChange = priceChange * elasticity;
                const adjustedQuantity = quantityRange[qi] * (1 + demandChange);
                const revenue = price * adjustedQuantity;
                row.push(revenue);
            }
            revenueMatrix.push(row);
        }
        
        return this.createPricingOptimization(containerId, priceRange, quantityRange, revenueMatrix, options);
    }
};


// =============================================================================
// RADAR CHART (ECharts) - Premium Styled Multi-Metric Analysis
// =============================================================================

const RadarChart = {
    // Premium color palettes for radar series
    seriesColors: [
        { line: '#6366f1', area: 'rgba(99, 102, 241, 0.25)', glow: 'rgba(99, 102, 241, 0.6)' },    // Indigo
        { line: '#10b981', area: 'rgba(16, 185, 129, 0.25)', glow: 'rgba(16, 185, 129, 0.6)' },   // Emerald
        { line: '#f59e0b', area: 'rgba(245, 158, 11, 0.25)', glow: 'rgba(245, 158, 11, 0.6)' },   // Amber
        { line: '#8b5cf6', area: 'rgba(139, 92, 246, 0.25)', glow: 'rgba(139, 92, 246, 0.6)' },   // Violet
        { line: '#06b6d4', area: 'rgba(6, 182, 212, 0.25)', glow: 'rgba(6, 182, 212, 0.6)' }      // Cyan
    ],
    
    /**
     * Create radar chart for multi-dimensional comparison
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('RadarChart: Container not found:', containerId);
            return null;
        }
        
        const chart = echarts.init(container, null, { renderer: 'canvas' });
        
        const option = {
            backgroundColor: 'transparent',
            title: {
                text: '',
                left: 'left',
                textStyle: {
                    color: '#f8fafc',
                    fontSize: 16,
                    fontFamily: 'Inter, sans-serif'
                }
            },
            tooltip: {
                backgroundColor: 'rgba(30, 41, 59, 0.95)',
                borderColor: '#6366f1',
                borderWidth: 1,
                textStyle: {
                    color: '#f1f5f9',
                    fontFamily: 'Inter, sans-serif'
                },
                formatter: (params) => {
                    const series = params.data;
                    let html = `<b style="font-size:14px">${series.name}</b><br>`;
                    data.indicators.forEach((ind, i) => {
                        const value = series.value[i];
                        const max = ind.max || 100;
                        const pct = (value / max * 100).toFixed(0);
                        html += `<span style="color:#94a3b8">${ind.name}:</span> ${value} <span style="color:#6366f1">(${pct}%)</span><br>`;
                    });
                    return html;
                }
            },
            legend: {
                data: data.series.map(s => s.name),
                bottom: 10,
                textStyle: { 
                    color: '#e2e8f0',
                    fontFamily: 'Inter, sans-serif'
                }
            },
            radar: {
                indicator: data.indicators.map(ind => ({
                    name: ind.name,
                    max: ind.max || 100
                })),
                center: ['50%', '50%'],
                radius: '62%',
                axisName: {
                    color: '#e2e8f0',
                    fontSize: 11,
                    fontFamily: 'Inter, sans-serif'
                },
                splitNumber: 5,
                splitLine: {
                    lineStyle: { 
                        color: [
                            'rgba(99, 102, 241, 0.1)',
                            'rgba(99, 102, 241, 0.15)',
                            'rgba(99, 102, 241, 0.2)',
                            'rgba(99, 102, 241, 0.25)',
                            'rgba(99, 102, 241, 0.3)'
                        ]
                    }
                },
                splitArea: {
                    areaStyle: {
                        color: [
                            'rgba(99, 102, 241, 0.02)', 
                            'rgba(99, 102, 241, 0.04)',
                            'rgba(99, 102, 241, 0.06)', 
                            'rgba(99, 102, 241, 0.08)',
                            'rgba(99, 102, 241, 0.1)'
                        ]
                    }
                },
                axisLine: {
                    lineStyle: { color: 'rgba(148, 163, 184, 0.3)' }
                }
            },
            series: [{
                type: 'radar',
                data: data.series.map((s, i) => {
                    const colors = this.seriesColors[i % this.seriesColors.length];
                    // Calculate average for glow intensity
                    const avgValue = s.values.reduce((a, b) => a + b, 0) / s.values.length;
                    const maxPossible = data.indicators.reduce((sum, ind) => sum + (ind.max || 100), 0) / data.indicators.length;
                    const intensity = avgValue / maxPossible;
                    
                    return {
                        name: s.name,
                        value: s.values,
                        symbol: 'circle',
                        symbolSize: 8,
                        lineStyle: {
                            color: colors.line,
                            width: 2.5,
                            shadowColor: colors.glow,
                            shadowBlur: intensity > 0.6 ? 10 : 5
                        },
                        areaStyle: {
                            color: {
                                type: 'radial',
                                x: 0.5, y: 0.5, r: 0.5,
                                colorStops: [
                                    { offset: 0, color: 'rgba(99, 102, 241, 0.05)' },
                                    { offset: 0.7, color: colors.area },
                                    { offset: 1, color: colors.area }
                                ]
                            },
                            opacity: 0.8
                        },
                        itemStyle: {
                            color: colors.line,
                            borderColor: '#1e293b',
                            borderWidth: 2,
                            shadowColor: colors.glow,
                            shadowBlur: 8
                        },
                        emphasis: {
                            lineStyle: {
                                width: 4,
                                shadowBlur: 15
                            },
                            itemStyle: {
                                shadowBlur: 15
                            },
                            areaStyle: {
                                opacity: 0.4
                            }
                        }
                    };
                })
            }]
        };
        
        chart.setOption(option);
        
        window.addEventListener('resize', () => chart.resize());
        
        console.log('RadarChart rendered with premium styling');
        return chart;
    },
    
    /**
     * Create RFM analysis radar
     */
    createRFM(containerId, segments, options = {}) {
        const indicators = [
            { name: 'Recency', max: 100 },
            { name: 'Frequency', max: 100 },
            { name: 'Monetary', max: 100 }
        ];
        
        const series = segments.map(segment => ({
            name: segment.name,
            values: [segment.recency, segment.frequency, segment.monetary]
        }));
        
        return this.create(containerId, {
            indicators,
            series,
            title: options.title || 'Customer RFM Analysis'
        }, options);
    },
    
    /**
     * Create financial health radar
     */
    createFinancialHealth(containerId, metrics, options = {}) {
        const indicators = [
            { name: 'Profitability', max: 100 },
            { name: 'Liquidity', max: 100 },
            { name: 'Efficiency', max: 100 },
            { name: 'Growth', max: 100 },
            { name: 'Stability', max: 100 }
        ];
        
        const series = [{
            name: 'Current Period',
            values: [
                metrics.profitability || 0,
                metrics.liquidity || 0,
                metrics.efficiency || 0,
                metrics.growth || 0,
                metrics.stability || 0
            ]
        }];
        
        if (metrics.previous) {
            series.push({
                name: 'Previous Period',
                values: [
                    metrics.previous.profitability || 0,
                    metrics.previous.liquidity || 0,
                    metrics.previous.efficiency || 0,
                    metrics.previous.growth || 0,
                    metrics.previous.stability || 0
                ]
            });
        }
        
        return this.create(containerId, {
            indicators,
            series,
            title: options.title || 'Financial Health Score'
        }, options);
    }
};


// =============================================================================
// EXPORTS
// =============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TreemapChart,
        TornadoChart,
        NetworkGraph,
        SurfaceChart,
        RadarChart
    };
}

if (typeof window !== 'undefined') {
    window.TreemapChart = TreemapChart;
    window.TornadoChart = TornadoChart;
    window.NetworkGraph = NetworkGraph;
    window.SurfaceChart = SurfaceChart;
    window.RadarChart = RadarChart;
}
