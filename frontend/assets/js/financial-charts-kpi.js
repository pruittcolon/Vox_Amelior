/**
 * Financial KPI Charts Module
 * 
 * Premium Gauge clusters, Bullet charts, and KPI card visualizations.
 * Gauges: Cash runway, margins, utilization rates
 * Bullets: Budget vs actual, targets
 * KPI Cards: Animated number displays with trends
 * 
 * @module FinancialChartsKPI
 * @requires FinancialChartsCore
 * @requires ECharts 5.4+
 * @requires Plotly.js 2.27+
 */

'use strict';

// =============================================================================
// GAUGE CHART (ECharts)
// =============================================================================

const GaugeChart = {
    /**
     * Create a single gauge chart
     * 
     * @param {string} containerId - DOM container ID
     * @param {object} data - Chart data
     * @param {number} data.value - Current value
     * @param {number} data.max - Maximum value
     * @param {string} data.title - Gauge title
     * @param {string} data.unit - Unit label (%, $, etc.)
     * @param {object} options - Additional options
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return null;
        
        const theme = window.FinancialTheme || FinancialTheme;
        const utils = window.FinancialUtils || FinancialUtils;
        const baseTheme = window.EChartsBaseTheme || EChartsBaseTheme || {};
        
        // Initialize ECharts
        const chart = echarts.init(container, null, { renderer: 'canvas' });
        
        // Determine color based on thresholds
        const color = this._getGaugeColor(data.value, data.max, data.thresholds, theme);
        
        const option = {
            ...baseTheme,
            series: [{
                type: 'gauge',
                startAngle: 200,
                endAngle: -20,
                min: 0,
                max: data.max || 100,
                splitNumber: 5,
                radius: '90%',
                center: ['50%', '55%'],
                
                // Progress arc
                progress: {
                    show: true,
                    width: 18,
                    itemStyle: {
                        color: {
                            type: 'linear',
                            x: 0, y: 0, x2: 1, y2: 0,
                            colorStops: [
                                { offset: 0, color: color.start },
                                { offset: 1, color: color.end }
                            ]
                        },
                        shadowColor: color.glow,
                        shadowBlur: 10
                    }
                },
                
                // Pointer
                pointer: {
                    show: true,
                    length: '60%',
                    width: 6,
                    itemStyle: {
                        color: '#e2e8f0'
                    }
                },
                
                // Axis line (background track)
                axisLine: {
                    lineStyle: {
                        width: 18,
                        color: [[1, 'rgba(148, 163, 184, 0.15)']]
                    }
                },
                
                // Ticks
                axisTick: {
                    show: true,
                    distance: -30,
                    length: 6,
                    lineStyle: {
                        color: 'rgba(148, 163, 184, 0.4)',
                        width: 1
                    }
                },
                
                // Split lines
                splitLine: {
                    show: true,
                    distance: -35,
                    length: 10,
                    lineStyle: {
                        color: 'rgba(148, 163, 184, 0.5)',
                        width: 2
                    }
                },
                
                // Axis labels
                axisLabel: {
                    show: true,
                    distance: 25,
                    color: '#94a3b8',
                    fontSize: 10,
                    formatter: (value) => {
                        if (data.unit === '%') return value + '%';
                        if (data.unit === '$') return utils.formatCompact(value);
                        return value;
                    }
                },
                
                // Title
                title: {
                    show: true,
                    offsetCenter: [0, '75%'],
                    color: '#cbd5e1',
                    fontSize: 12,
                    fontWeight: 500
                },
                
                // Value detail
                detail: {
                    show: true,
                    offsetCenter: [0, '35%'],
                    fontSize: 28,
                    fontWeight: 700,
                    fontFamily: theme.fonts.primary,
                    color: color.text,
                    formatter: (value) => {
                        if (data.unit === '%') return value.toFixed(1) + '%';
                        if (data.unit === '$') return utils.formatCurrency(value, 'USD', true);
                        if (data.unit === 'months') return value.toFixed(1) + ' mo';
                        return value.toFixed(1);
                    }
                },
                
                data: [{
                    value: data.value,
                    name: data.title || ''
                }]
            }]
        };
        
        chart.setOption(option);
        
        // Animate on load
        this._animateValue(chart, 0, data.value, 1500);
        
        // Handle resize
        window.addEventListener('resize', () => chart.resize());
        
        return chart;
    },
    
    /**
     * Get gauge color based on value and thresholds
     */
    _getGaugeColor(value, max, thresholds, theme) {
        const percent = (value / max) * 100;
        
        // Default thresholds: good > 66%, warning 33-66%, danger < 33%
        const t = thresholds || { danger: 33, warning: 66 };
        
        if (percent <= t.danger) {
            return {
                start: '#dc2626',
                end: '#f87171',
                glow: 'rgba(220, 38, 38, 0.4)',
                text: theme.colors.danger
            };
        } else if (percent <= t.warning) {
            return {
                start: '#d97706',
                end: '#fbbf24',
                glow: 'rgba(217, 119, 6, 0.4)',
                text: theme.colors.warning
            };
        } else {
            return {
                start: '#059669',
                end: '#34d399',
                glow: 'rgba(5, 150, 105, 0.4)',
                text: theme.colors.success
            };
        }
    },
    
    /**
     * Animate gauge value
     */
    _animateValue(chart, from, to, duration) {
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const currentValue = from + (to - from) * eased;
            
            chart.setOption({
                series: [{ data: [{ value: currentValue }] }]
            });
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    },
    
    /**
     * Create cash runway gauge
     */
    createCashRunway(containerId, months, options = {}) {
        return this.create(containerId, {
            value: months,
            max: 24,
            title: 'Cash Runway',
            unit: 'months',
            thresholds: { danger: 25, warning: 50 }
        }, options);
    },
    
    /**
     * Create margin gauge
     */
    createMargin(containerId, marginPercent, options = {}) {
        return this.create(containerId, {
            value: marginPercent,
            max: 100,
            title: options.title || 'Profit Margin',
            unit: '%',
            thresholds: { danger: 10, warning: 25 }
        }, options);
    },
    
    /**
     * Create utilization gauge
     */
    createUtilization(containerId, utilPercent, options = {}) {
        return this.create(containerId, {
            value: utilPercent,
            max: 100,
            title: options.title || 'Utilization',
            unit: '%',
            thresholds: { danger: 33, warning: 66 }
        }, options);
    }
};


// =============================================================================
// GAUGE CLUSTER (Multiple Gauges)
// =============================================================================

const GaugeCluster = {
    /**
     * Create a cluster of mini gauges for KPI overview
     * 
     * @param {string} containerId - DOM container ID
     * @param {object[]} gauges - Array of gauge configs
     */
    create(containerId, gauges, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return null;
        
        const theme = window.FinancialTheme || FinancialTheme;
        
        // Initialize ECharts
        const chart = echarts.init(container, null, { renderer: 'canvas' });
        
        const series = gauges.map((gauge, index) => {
            const cols = Math.min(gauges.length, 4);
            const row = Math.floor(index / cols);
            const col = index % cols;
            
            const xCenter = (100 / cols) * (col + 0.5);
            const yCenter = gauges.length > 4 ? (row === 0 ? 35 : 75) : 50;
            
            const color = this._getColor(gauge.value, gauge.max, gauge.thresholds, theme);
            
            return {
                type: 'gauge',
                startAngle: 200,
                endAngle: -20,
                min: 0,
                max: gauge.max || 100,
                radius: gauges.length > 4 ? '35%' : '40%',
                center: [`${xCenter}%`, `${yCenter}%`],
                
                progress: {
                    show: true,
                    width: 10,
                    itemStyle: { color: color.main }
                },
                
                pointer: { show: false },
                
                axisLine: {
                    lineStyle: {
                        width: 10,
                        color: [[1, 'rgba(148, 163, 184, 0.1)']]
                    }
                },
                
                axisTick: { show: false },
                splitLine: { show: false },
                axisLabel: { show: false },
                
                title: {
                    show: true,
                    offsetCenter: [0, '85%'],
                    color: '#94a3b8',
                    fontSize: 11,
                    fontWeight: 500
                },
                
                detail: {
                    show: true,
                    offsetCenter: [0, '20%'],
                    fontSize: 20,
                    fontWeight: 700,
                    fontFamily: theme.fonts.primary,
                    color: color.text,
                    formatter: gauge.formatter || ((v) => v.toFixed(0) + (gauge.unit || ''))
                },
                
                data: [{
                    value: gauge.value,
                    name: gauge.title
                }]
            };
        });
        
        const baseTheme = window.EChartsBaseTheme || EChartsBaseTheme || {};
        const option = {
            ...baseTheme,
            series
        };
        
        chart.setOption(option);
        
        // Animate all gauges
        gauges.forEach((gauge, i) => {
            this._animateGauge(chart, i, 0, gauge.value, 1200 + i * 200);
        });
        
        window.addEventListener('resize', () => chart.resize());
        
        return chart;
    },
    
    _getColor(value, max, thresholds, theme) {
        const percent = (value / max) * 100;
        const t = thresholds || { danger: 33, warning: 66 };
        
        if (percent <= t.danger) {
            return { main: theme.colors.danger, text: theme.colors.danger };
        } else if (percent <= t.warning) {
            return { main: theme.colors.warning, text: theme.colors.warning };
        } else {
            return { main: theme.colors.success, text: theme.colors.success };
        }
    },
    
    _animateGauge(chart, seriesIndex, from, to, duration) {
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const currentValue = from + (to - from) * eased;
            
            const series = chart.getOption().series;
            series[seriesIndex].data[0].value = currentValue;
            chart.setOption({ series });
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    },
    
    /**
     * Create financial health cluster
     */
    createFinancialHealth(containerId, metrics, options = {}) {
        const gauges = [
            {
                title: 'Cash Runway',
                value: metrics.cashRunway || 0,
                max: 24,
                unit: ' mo',
                thresholds: { danger: 25, warning: 50 }
            },
            {
                title: 'Profit Margin',
                value: metrics.profitMargin || 0,
                max: 100,
                unit: '%',
                thresholds: { danger: 10, warning: 25 }
            },
            {
                title: 'Revenue Growth',
                value: Math.min(metrics.revenueGrowth || 0, 100),
                max: 100,
                unit: '%',
                thresholds: { danger: 0, warning: 15 }
            },
            {
                title: 'Churn Rate',
                value: metrics.churnRate || 0,
                max: 100,
                unit: '%',
                thresholds: { danger: 80, warning: 50 }
            }
        ];
        
        return this.create(containerId, gauges, options);
    }
};


// =============================================================================
// BULLET CHART (Plotly)
// =============================================================================

const BulletChart = {
    // Premium gradient colors
    gradients: {
        success: ['#047857', '#059669', '#10b981'],     // Emerald (exceeded target)
        danger: ['#991b1b', '#dc2626', '#ef4444'],       // Red (below target)
        ranges: {
            poor: 'rgba(239, 68, 68, 0.15)',
            fair: 'rgba(245, 158, 11, 0.15)',
            good: 'rgba(16, 185, 129, 0.15)'
        }
    },
    
    /**
     * Create bullet chart for target comparison
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('BulletChart: Container not found:', containerId);
            return null;
        }
        
        const theme = window.FinancialTheme || { 
            colors: { success: '#10b981', danger: '#ef4444' }
        };
        
        const traces = [];
        const pctOfTarget = (data.actual / data.target * 100).toFixed(0);
        const isAboveTarget = data.actual >= data.target;
        
        // Background ranges (poor, fair, good)
        const ranges = data.ranges || [
            { label: 'Poor', value: data.max * 0.33, color: this.gradients.ranges.poor },
            { label: 'Fair', value: data.max * 0.66, color: this.gradients.ranges.fair },
            { label: 'Good', value: data.max, color: this.gradients.ranges.good }
        ];
        
        // Build range bars (stacked)
        ranges.forEach((range, i) => {
            const prevValue = i > 0 ? ranges[i - 1].value : 0;
            traces.push({
                type: 'bar',
                orientation: 'h',
                name: range.label,
                y: [data.title || 'Performance'],
                x: [range.value - prevValue],
                marker: {
                    color: range.color,
                    line: { color: 'rgba(148, 163, 184, 0.1)', width: 1 }
                },
                hoverinfo: 'skip',
                showlegend: false
            });
        });
        
        // Actual value bar with gradient effect
        const barGradient = isAboveTarget ? this.gradients.success : this.gradients.danger;
        traces.push({
            type: 'bar',
            orientation: 'h',
            name: 'Actual',
            y: [data.title || 'Performance'],
            x: [data.actual],
            marker: {
                color: barGradient[1],
                line: {
                    color: barGradient[0],
                    width: 2
                }
            },
            width: 0.35,
            hovertemplate: `<b style="font-size:14px">${data.title || 'Actual'}</b><br>` +
                `<span style="color:${isAboveTarget ? '#10b981' : '#ef4444'}">` +
                `${isAboveTarget ? '✅' : '⚠️'} Actual:</span> $${data.actual.toLocaleString()}<br>` +
                `<span style="color:#94a3b8">vs Target:</span> ${pctOfTarget}%<extra></extra>`,
            hoverlabel: {
                bgcolor: 'rgba(30, 41, 59, 0.95)',
                bordercolor: isAboveTarget ? '#10b981' : '#ef4444',
                font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
            }
        });
        
        // Target marker with glow effect
        traces.push({
            type: 'scatter',
            mode: 'markers',
            name: 'Target',
            y: [data.title || 'Performance'],
            x: [data.target],
            marker: {
                symbol: 'line-ns',
                size: 32,
                color: '#f8fafc',
                line: {
                    width: 4,
                    color: '#f8fafc'
                }
            },
            hovertemplate: `<b style="font-size:14px">🎯 Target</b><br>` +
                `<span style="color:#6366f1">Goal:</span> $${data.target.toLocaleString()}<extra></extra>`,
            hoverlabel: {
                bgcolor: 'rgba(30, 41, 59, 0.95)',
                bordercolor: '#6366f1',
                font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
            }
        });
        
        const layout = {
            barmode: 'stack',
            showlegend: false,
            font: {
                family: 'Inter, sans-serif',
                color: '#e2e8f0'
            },
            xaxis: {
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                linecolor: 'rgba(148, 163, 184, 0.2)',
                tickfont: { color: '#94a3b8', size: 11 },
                range: [0, data.max * 1.05],
                tickformat: ',.0f',
                tickprefix: data.prefix || '$'
            },
            yaxis: {
                gridcolor: 'rgba(148, 163, 184, 0.05)',
                linecolor: 'rgba(148, 163, 184, 0.2)',
                tickfont: { color: '#e2e8f0', size: 12 },
                showticklabels: true
            },
            height: options.height || 100,
            margin: { l: 120, r: 40, t: 20, b: 40 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };
        
        const config = { displayModeBar: false, responsive: true };
        
        try {
            Plotly.newPlot(containerId, traces, layout, config);
            console.log('BulletChart rendered with premium styling');
        } catch (e) {
            console.error('BulletChart failed:', e);
        }
        
        return { containerId, type: 'bullet' };
    },
    
    /**
     * Create multiple bullet charts
     */
    createMultiple(containerId, items, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('BulletChart: Container not found:', containerId);
            return null;
        }
        
        const traces = [];
        const categories = items.map(i => i.title);
        
        // Background ranges for each
        items.forEach((item, idx) => {
            const max = item.max || Math.max(item.actual, item.target) * 1.2;
            
            // Add range background
            traces.push({
                type: 'bar',
                orientation: 'h',
                name: 'Range',
                y: [item.title],
                x: [max],
                marker: { 
                    color: 'rgba(148, 163, 184, 0.08)',
                    line: { color: 'rgba(148, 163, 184, 0.1)', width: 1 }
                },
                hoverinfo: 'skip',
                showlegend: false
            });
        });
        
        // Actual bars with gradient coloring
        const actualColors = items.map(i => {
            const pct = i.actual / i.target;
            if (pct >= 1) return this.gradients.success[1];
            if (pct >= 0.8) return '#f59e0b';  // Amber for close
            return this.gradients.danger[1];
        });
        
        const borderColors = items.map(i => {
            const pct = i.actual / i.target;
            if (pct >= 1) return this.gradients.success[0];
            if (pct >= 0.8) return '#d97706';
            return this.gradients.danger[0];
        });
        
        traces.push({
            type: 'bar',
            orientation: 'h',
            name: 'Actual',
            y: categories,
            x: items.map(i => i.actual),
            marker: {
                color: actualColors,
                line: { color: borderColors, width: 2 }
            },
            width: 0.45,
            hovertemplate: '<b>%{y}</b><br>' +
                '<span style="color:#8b5cf6">💰 Actual:</span> $%{x:,.0f}<extra></extra>',
            hoverlabel: {
                bgcolor: 'rgba(30, 41, 59, 0.95)',
                bordercolor: '#6366f1',
                font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
            }
        });
        
        // Target markers
        traces.push({
            type: 'scatter',
            mode: 'markers',
            name: 'Target',
            y: categories,
            x: items.map(i => i.target),
            marker: {
                symbol: 'line-ns',
                size: 28,
                color: '#f8fafc',
                line: { width: 3, color: '#f8fafc' }
            },
            hovertemplate: '<b>%{y}</b><br>🎯 Target: $%{x:,.0f}<extra></extra>',
            hoverlabel: {
                bgcolor: 'rgba(30, 41, 59, 0.95)',
                bordercolor: '#f8fafc',
                font: { family: 'Inter, sans-serif', size: 13, color: '#f1f5f9' }
            }
        });
        
        const layout = {
            barmode: 'overlay',
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
                tickformat: ',.0f',
                tickprefix: '$'
            },
            yaxis: {
                gridcolor: 'rgba(148, 163, 184, 0.05)',
                linecolor: 'rgba(148, 163, 184, 0.2)',
                tickfont: { color: '#e2e8f0', size: 11 },
                automargin: true
            },
            height: options.height || (items.length * 65 + 80),
            margin: { l: 120, r: 40, t: 20, b: 60 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };
        
        const config = { displayModeBar: false, responsive: true };
        
        try {
            Plotly.newPlot(containerId, traces, layout, config);
            console.log('BulletChart (multiple) rendered with premium styling');
        } catch (e) {
            console.error('BulletChart multiple failed:', e);
        }
        
        return { containerId, type: 'bulletMultiple' };
    },
    
    /**
     * Create budget comparison bullets
     */
    createBudgetComparison(containerId, departments, options = {}) {
        const items = departments.map(d => ({
            title: d.name,
            actual: d.actual,
            target: d.budget,
            max: Math.max(d.actual, d.budget) * 1.1
        }));
        
        return this.createMultiple(containerId, items, options);
    }
};


// =============================================================================
// KPI CARD COMPONENT
// =============================================================================

const KPICard = {
    /**
     * Create animated KPI card
     * 
     * @param {string} containerId - DOM container ID
     * @param {object} data - KPI data
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return null;
        
        const theme = window.FinancialTheme || FinancialTheme;
        const utils = window.FinancialUtils || FinancialUtils;
        
        // Determine trend direction and color
        const trend = data.trend || 0;
        const trendUp = trend > 0;
        const trendColor = data.invertTrend 
            ? (trendUp ? theme.colors.danger : theme.colors.success)
            : (trendUp ? theme.colors.success : theme.colors.danger);
        const trendIcon = trendUp ? '↑' : (trend < 0 ? '↓' : '→');
        
        // Build HTML
        container.innerHTML = `
            <div class="kpi-card" style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 16px;
                padding: 20px 24px;
                position: relative;
                overflow: hidden;
            ">
                <div class="kpi-glow" style="
                    position: absolute;
                    top: -50%;
                    right: -50%;
                    width: 100%;
                    height: 100%;
                    background: radial-gradient(circle, ${data.glowColor || 'rgba(99, 102, 241, 0.1)'} 0%, transparent 70%);
                    pointer-events: none;
                "></div>
                
                <div class="kpi-icon" style="
                    font-size: 24px;
                    margin-bottom: 8px;
                ">${data.icon || '📊'}</div>
                
                <div class="kpi-label" style="
                    color: #94a3b8;
                    font-size: 12px;
                    font-weight: 500;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 4px;
                ">${data.label}</div>
                
                <div class="kpi-value" style="
                    color: #f8fafc;
                    font-size: 32px;
                    font-weight: 700;
                    font-family: ${theme.fonts.primary};
                    line-height: 1.2;
                " data-value="${data.value}" data-format="${data.format || 'currency'}">
                    ${data.format === 'percent' ? '0%' : '$0'}
                </div>
                
                <div class="kpi-trend" style="
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    margin-top: 8px;
                ">
                    <span style="
                        color: ${trendColor};
                        font-weight: 600;
                        font-size: 14px;
                    ">${trendIcon} ${Math.abs(trend).toFixed(1)}%</span>
                    <span style="
                        color: #64748b;
                        font-size: 12px;
                    ">${data.trendLabel || 'vs last period'}</span>
                </div>
                
                ${data.sparkline ? `
                    <div class="kpi-sparkline" id="${containerId}-spark" style="
                        height: 40px;
                        margin-top: 12px;
                    "></div>
                ` : ''}
            </div>
        `;
        
        // Animate the value
        const valueEl = container.querySelector('.kpi-value');
        utils.animateNumber(valueEl, data.value, 1500, data.format || 'currency');
        
        // Add sparkline if data provided
        if (data.sparkline) {
            this._createSparkline(`${containerId}-spark`, data.sparkline, theme);
        }
        
        return { containerId, type: 'kpiCard' };
    },
    
    /**
     * Create mini sparkline
     */
    _createSparkline(containerId, values, theme) {
        const trace = {
            type: 'scatter',
            mode: 'lines',
            x: values.map((_, i) => i),
            y: values,
            line: {
                color: theme.colors.primary,
                width: 2,
                shape: 'spline'
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(99, 102, 241, 0.1)',
            hoverinfo: 'skip'
        };
        
        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { l: 0, r: 0, t: 0, b: 0 },
            xaxis: { visible: false },
            yaxis: { visible: false },
            showlegend: false,
            height: 40
        };
        
        Plotly.newPlot(containerId, [trace], layout, { 
            displayModeBar: false, 
            staticPlot: true 
        });
    },
    
    /**
     * Create KPI card grid
     */
    createGrid(containerId, kpis, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return null;
        
        const cols = options.columns || 4;
        
        container.innerHTML = `
            <div class="kpi-grid" style="
                display: grid;
                grid-template-columns: repeat(${cols}, 1fr);
                gap: 20px;
            ">
                ${kpis.map((kpi, i) => `
                    <div id="${containerId}-kpi-${i}"></div>
                `).join('')}
            </div>
        `;
        
        // Create each KPI card
        kpis.forEach((kpi, i) => {
            this.create(`${containerId}-kpi-${i}`, kpi, options);
        });
        
        return { containerId, type: 'kpiGrid' };
    },
    
    /**
     * Create financial metrics grid
     */
    createFinancialMetrics(containerId, metrics, options = {}) {
        const kpis = [
            {
                icon: '💰',
                label: 'Total Revenue',
                value: metrics.revenue || 0,
                format: 'currency',
                trend: metrics.revenueTrend || 0,
                glowColor: 'rgba(34, 197, 94, 0.1)'
            },
            {
                icon: '📊',
                label: 'Profit Margin',
                value: metrics.profitMargin || 0,
                format: 'percent',
                trend: metrics.marginTrend || 0,
                glowColor: 'rgba(99, 102, 241, 0.1)'
            },
            {
                icon: '👥',
                label: 'Customer LTV',
                value: metrics.customerLTV || 0,
                format: 'currency',
                trend: metrics.ltvTrend || 0,
                glowColor: 'rgba(6, 182, 212, 0.1)'
            },
            {
                icon: '📉',
                label: 'Churn Rate',
                value: metrics.churnRate || 0,
                format: 'percent',
                trend: metrics.churnTrend || 0,
                invertTrend: true,
                glowColor: 'rgba(239, 68, 68, 0.1)'
            }
        ];
        
        return this.createGrid(containerId, kpis, options);
    }
};


// =============================================================================
// EXPORTS
// =============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        GaugeChart,
        GaugeCluster,
        BulletChart,
        KPICard
    };
}

if (typeof window !== 'undefined') {
    window.GaugeChart = GaugeChart;
    window.GaugeCluster = GaugeCluster;
    window.BulletChart = BulletChart;
    window.KPICard = KPICard;
}
