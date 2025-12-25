/**
 * NexusAI Dashboard Visualizations
 * ECharts and Plotly chart initialization and updates.
 *
 * @module nexus/components/dashboard
 */

import { VIZ_COLORS } from '../core/config.js';

// ============================================================================
// State
// ============================================================================

let echartsInstances = {};
let plotlyCharts = {};
let enginePerformanceData = [];
let initialized = false;

// ============================================================================
// Public API
// ============================================================================

/**
 * Check if dashboard is initialized.
 * @returns {boolean}
 */
export function isDashboardInitialized() {
    return initialized;
}

/**
 * Initialize all dashboard charts.
 */
export function initDashboard() {
    if (initialized) return;

    // Check for required libraries
    if (typeof echarts === 'undefined') {
        console.warn('ECharts not loaded, skipping gauge initialization');
    } else {
        initTimingGauges();
        initCategoryRadar();
        initSuccessDonut();
    }

    if (typeof Plotly === 'undefined') {
        console.warn('Plotly not loaded, skipping chart initialization');
    } else {
        initTimelineChart();
        initSpeedRankingChart();
    }

    initialized = true;
}

/**
 * Reset dashboard for new analysis.
 */
export function resetDashboard() {
    enginePerformanceData = [];
    initialized = false;

    // Dispose existing instances
    Object.values(echartsInstances).forEach(chart => {
        if (chart && chart.dispose) chart.dispose();
    });
    echartsInstances = {};
    plotlyCharts = {};
}

/**
 * Track engine performance after completion.
 * @param {Object} engine - Engine definition
 * @param {number} duration - Duration in ms
 * @param {'success'|'error'} status
 * @param {number} dataSize - Result data size in bytes
 */
export function trackEnginePerformance(engine, duration, status, dataSize = 0) {
    const existingIdx = enginePerformanceData.findIndex(e => e.name === engine.name);

    const perfData = {
        name: engine.name,
        display: engine.display,
        icon: engine.icon,
        category: engine.category,
        duration: duration,
        status: status,
        dataSize: dataSize,
        timestamp: Date.now()
    };

    if (existingIdx >= 0) {
        enginePerformanceData[existingIdx] = perfData;
    } else {
        enginePerformanceData.push(perfData);
    }

    updateDashboard();
}

/**
 * Get current performance data.
 * @returns {Object[]}
 */
export function getPerformanceData() {
    return enginePerformanceData;
}

// ============================================================================
// Chart Initialization
// ============================================================================

function initTimingGauges() {
    const gaugeConfigs = [
        { id: 'echarts-total-time', max: 120, color: VIZ_COLORS.primary },
        { id: 'echarts-avg-time', max: 10, color: VIZ_COLORS.accent },
        { id: 'echarts-fastest', max: 5, color: VIZ_COLORS.success },
        { id: 'echarts-slowest', max: 30, color: VIZ_COLORS.warning }
    ];

    gaugeConfigs.forEach(config => {
        const container = document.getElementById(config.id);
        if (!container) return;

        const chart = echarts.init(container, null, { renderer: 'canvas' });
        echartsInstances[config.id] = chart;

        chart.setOption({
            series: [{
                type: 'gauge',
                startAngle: 200,
                endAngle: -20,
                min: 0,
                max: config.max,
                splitNumber: 4,
                itemStyle: {
                    color: config.color,
                    shadowColor: config.color,
                    shadowBlur: 10
                },
                progress: { show: true, roundCap: true, width: 12 },
                pointer: { show: false },
                axisLine: {
                    roundCap: true,
                    lineStyle: { width: 12, color: [[1, 'rgba(255,255,255,0.1)']] }
                },
                axisTick: { show: false },
                splitLine: { show: false },
                axisLabel: { show: false },
                title: { show: false },
                detail: {
                    valueAnimation: true,
                    fontSize: 24,
                    fontWeight: 'bold',
                    color: config.color,
                    offsetCenter: [0, 0],
                    formatter: val => val.toFixed(1) + 's'
                },
                data: [{ value: 0 }]
            }]
        });
    });
}

function initCategoryRadar() {
    const container = document.getElementById('echarts-category-radar');
    if (!container) return;

    const chart = echarts.init(container, null, { renderer: 'canvas' });
    echartsInstances['category-radar'] = chart;

    chart.setOption({
        backgroundColor: 'transparent',
        radar: {
            indicator: [
                { name: 'ML & Analytics', max: 100 },
                { name: 'Financial', max: 100 },
                { name: 'Operations', max: 100 },
                { name: 'Advanced', max: 100 },
                { name: 'Speed', max: 100 }
            ],
            shape: 'polygon',
            splitNumber: 4,
            axisName: { color: VIZ_COLORS.textMuted, fontSize: 11 },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
            splitArea: { areaStyle: { color: ['rgba(6,182,212,0.02)', 'rgba(6,182,212,0.05)'] } },
            axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }
        },
        series: [{
            type: 'radar',
            symbol: 'circle',
            symbolSize: 8,
            lineStyle: { width: 2, color: VIZ_COLORS.primary },
            areaStyle: { color: 'rgba(6, 182, 212, 0.2)' },
            itemStyle: { color: VIZ_COLORS.primary, borderColor: VIZ_COLORS.primaryLight, borderWidth: 2 },
            data: [{ value: [0, 0, 0, 0, 0], name: 'Performance' }]
        }]
    });
}

function initSuccessDonut() {
    const container = document.getElementById('echarts-success-rate');
    if (!container) return;

    const chart = echarts.init(container, null, { renderer: 'canvas' });
    echartsInstances['success-donut'] = chart;

    chart.setOption({
        backgroundColor: 'transparent',
        series: [{
            type: 'pie',
            radius: ['50%', '75%'],
            center: ['50%', '50%'],
            avoidLabelOverlap: false,
            itemStyle: { borderRadius: 6, borderColor: VIZ_COLORS.background, borderWidth: 2 },
            label: {
                show: true,
                position: 'center',
                formatter: '{d}%',
                fontSize: 28,
                fontWeight: 'bold',
                color: VIZ_COLORS.success
            },
            emphasis: { label: { show: true, fontSize: 32, fontWeight: 'bold' } },
            data: [
                { value: 0, name: 'Success', itemStyle: { color: VIZ_COLORS.success } },
                { value: 0, name: 'Failed', itemStyle: { color: VIZ_COLORS.error } },
                { value: 22, name: 'Pending', itemStyle: { color: 'rgba(255,255,255,0.1)' } }
            ]
        }]
    });
}

function initTimelineChart() {
    const container = document.getElementById('plotly-timeline');
    if (!container) return;

    Plotly.newPlot(container, [], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 120, r: 20, t: 20, b: 40 },
        xaxis: { title: 'Time (seconds)', gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
        yaxis: { autorange: 'reversed', gridcolor: 'rgba(255,255,255,0.05)' },
        showlegend: false
    }, { responsive: true, displayModeBar: false });

    plotlyCharts['timeline'] = container;
}

function initSpeedRankingChart() {
    const container = document.getElementById('plotly-speed-ranking');
    if (!container) return;

    Plotly.newPlot(container, [], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted },
        margin: { l: 150, r: 30, t: 20, b: 40 }
    }, { responsive: true, displayModeBar: false });

    plotlyCharts['speed-ranking'] = container;
}

// ============================================================================
// Chart Updates
// ============================================================================

function updateDashboard() {
    if (!enginePerformanceData.length) return;

    const completed = enginePerformanceData.filter(e => e.status !== 'pending');
    const successful = completed.filter(e => e.status === 'success');
    const failed = completed.filter(e => e.status === 'error');
    const pending = 22 - completed.length;

    const durations = completed.map(e => e.duration / 1000);
    const totalTime = durations.reduce((a, b) => a + b, 0);
    const avgTime = durations.length ? totalTime / durations.length : 0;
    const fastest = durations.length ? Math.min(...durations) : 0;
    const slowest = durations.length ? Math.max(...durations) : 0;

    // Update gauges
    updateGauge('echarts-total-time', totalTime);
    updateGauge('echarts-avg-time', avgTime);
    updateGauge('echarts-fastest', fastest);
    updateGauge('echarts-slowest', slowest);

    // Update donut
    updateSuccessDonut(successful.length, failed.length, pending);

    // Update radar
    const categoryScores = {};
    ['ml', 'financial', 'operations', 'advanced'].forEach(cat => {
        const catEngines = successful.filter(e => e.category === cat);
        const catTotal = enginePerformanceData.filter(e => e.category === cat).length || 1;
        categoryScores[cat] = (catEngines.length / catTotal) * 100;
    });
    const speedScore = avgTime > 0 ? Math.max(0, 100 - (avgTime * 10)) : 100;
    updateCategoryRadar(categoryScores.ml, categoryScores.financial, categoryScores.operations || 0, categoryScores.advanced, speedScore);

    // Update timeline
    updateTimelineChart(completed);
    updateSpeedRanking(completed);
}

function updateGauge(id, value) {
    const chart = echartsInstances[id];
    if (chart) {
        chart.setOption({ series: [{ data: [{ value }] }] });
    }
}

function updateSuccessDonut(success, failed, pending) {
    const chart = echartsInstances['success-donut'];
    if (!chart) return;

    const total = success + failed + pending;
    const successPercent = total > 0 ? Math.round((success / total) * 100) : 0;

    chart.setOption({
        series: [{
            label: { formatter: successPercent + '%', color: success > failed ? VIZ_COLORS.success : VIZ_COLORS.error },
            data: [
                { value: success, name: 'Success' },
                { value: failed, name: 'Failed' },
                { value: pending, name: 'Pending' }
            ]
        }]
    });
}

function updateCategoryRadar(ml, financial, ops, advanced, speed) {
    const chart = echartsInstances['category-radar'];
    if (chart) {
        chart.setOption({ series: [{ data: [{ value: [ml, financial, ops, advanced, speed] }] }] });
    }
}

function updateTimelineChart(engineData) {
    const container = plotlyCharts['timeline'];
    if (!container || !engineData.length) return;

    const data = [{
        type: 'bar',
        orientation: 'h',
        x: engineData.map(e => e.duration / 1000),
        y: engineData.map(e => e.display),
        marker: {
            color: engineData.map(e => e.status === 'error' ? VIZ_COLORS.error : (VIZ_COLORS[e.category] || VIZ_COLORS.primary)),
            line: { width: 0 }
        },
        text: engineData.map(e => (e.duration / 1000).toFixed(2) + 's'),
        textposition: 'outside',
        textfont: { color: VIZ_COLORS.text, size: 10 },
        hovertemplate: '<b>%{y}</b><br>Duration: %{x:.2f}s<extra></extra>'
    }];

    Plotly.react(container, data, {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 10 },
        margin: { l: 140, r: 50, t: 10, b: 30 },
        xaxis: { title: 'Duration (seconds)', gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
        yaxis: { autorange: 'reversed', gridcolor: 'rgba(255,255,255,0.05)' },
        showlegend: false
    }, { responsive: true, displayModeBar: false });
}

function updateSpeedRanking(engineData) {
    const container = plotlyCharts['speed-ranking'];
    if (!container) return;

    const sorted = [...engineData].filter(e => e.status !== 'pending').sort((a, b) => a.duration - b.duration);

    const data = [{
        type: 'bar',
        orientation: 'h',
        x: sorted.map(e => e.duration / 1000),
        y: sorted.map(e => e.display),
        marker: {
            color: sorted.map((e, i) => {
                if (e.status === 'error') return VIZ_COLORS.error;
                return i < 3 ? VIZ_COLORS.success : i < sorted.length * 0.5 ? VIZ_COLORS.primary : VIZ_COLORS.warning;
            }),
            line: { width: 0 }
        },
        text: sorted.map((e, i) => i < 3 ? '🏆' : ''),
        textposition: 'outside',
        hovertemplate: '<b>%{y}</b><br>%{x:.2f}s<extra></extra>'
    }];

    Plotly.react(container, data, {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 10 },
        margin: { l: 150, r: 40, t: 10, b: 30 },
        xaxis: { title: 'Duration (seconds)', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { autorange: 'reversed', gridcolor: 'rgba(255,255,255,0.05)' }
    }, { responsive: true, displayModeBar: false });
}
