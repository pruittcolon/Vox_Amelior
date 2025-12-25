/**
 * Emotions Module - Chart Rendering
 * Provides all chart rendering functions for emotion analytics.
 * 
 * @module emotions/charts
 */

import { EMOTION_COLORS, METRIC_COLORS, getState, hexToRgba, getSpeakerColor } from './core.js';

// ============================================================================
// Chart Instances
// ============================================================================

let pieChart = null;
let trendChart = null;
let radarChart = null;

// ============================================================================
// Theme Configuration
// ============================================================================

function getChartThemeColors() {
    return {
        textPrimary: '#1e293b',
        textSecondary: '#64748b',
        gridColor: 'rgba(0, 82, 204, 0.1)',
        tooltipBg: '#ffffff',
        tooltipBorder: 'rgba(0, 82, 204, 0.15)'
    };
}

// ============================================================================
// Pie/Doughnut Chart
// ============================================================================

export function renderDistributionChart(canvasId, totals = {}, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const { onSliceClick } = options;
    const state = getState();
    const labels = Object.keys(totals).filter(k => state.selectedEmotions.has(k));
    const data = labels.map(k => totals[k] || 0);
    const colors = labels.map(k => EMOTION_COLORS[k] || '#6b7280');

    if (pieChart) pieChart.destroy();

    pieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{
                data,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.chart._metasets[context.datasetIndex].total;
                            const percentage = Math.round((value / total) * 100) + '%';
                            return `${label}: ${value} (${percentage})`;
                        }
                    }
                }
            },
            onClick: (evt, activeElements) => {
                if (activeElements.length > 0 && onSliceClick) {
                    const index = activeElements[0].index;
                    const label = labels[index];
                    onSliceClick(label);
                }
            },
            onHover: (event, chartElement) => {
                event.native.target.style.cursor = chartElement[0] ? 'pointer' : 'default';
            }
        }
    });

    return pieChart;
}

// ============================================================================
// Timeline/Trend Chart
// ============================================================================

export function renderTrendChart(canvasId, timeline = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const state = getState();
    const theme = getChartThemeColors();

    // Build datasets from timeline emotions
    const emotionData = timeline.emotions || {};
    const labels = timeline.labels || [];

    const datasets = Object.entries(emotionData)
        .filter(([emotion]) => state.selectedEmotions.has(emotion))
        .map(([emotion, values]) => ({
            label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
            data: values,
            borderColor: EMOTION_COLORS[emotion] || '#6b7280',
            backgroundColor: hexToRgba(EMOTION_COLORS[emotion] || '#6b7280', 0.1),
            fill: true,
            tension: 0.4
        }));

    if (trendChart) trendChart.destroy();

    trendChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: theme.gridColor }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { usePointStyle: true }
                }
            }
        }
    });

    return trendChart;
}

// ============================================================================
// Radar Chart (Speech Patterns)
// ============================================================================

export function renderRadarChart(canvasId, speakers = []) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const metrics = ['pace_wpm', 'pitch_mean', 'energy_mean'];
    const labels = metrics.map(m => m.split('_')[0].charAt(0).toUpperCase() + m.split('_')[0].slice(1));

    // Normalize values to 0-100 scale
    const maxValues = {
        pace_wpm: 200,
        pitch_mean: 300,
        energy_mean: 1
    };

    const datasets = speakers.slice(0, 5).map(speaker => {
        const color = getSpeakerColor(speaker.speaker);
        return {
            label: speaker.speaker || 'Unknown',
            data: metrics.map(m => {
                const val = speaker.speech_metrics?.[m] || 0;
                return Math.min(100, (val / maxValues[m]) * 100);
            }),
            borderColor: color,
            backgroundColor: hexToRgba(color, 0.2),
            pointBackgroundColor: color
        };
    });

    if (radarChart) radarChart.destroy();

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { display: false },
                    grid: { color: 'rgba(0, 82, 204, 0.1)' }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { usePointStyle: true }
                }
            }
        }
    });

    return radarChart;
}

// ============================================================================
// Cleanup
// ============================================================================

export function destroyAllCharts() {
    if (pieChart) { pieChart.destroy(); pieChart = null; }
    if (trendChart) { trendChart.destroy(); trendChart = null; }
    if (radarChart) { radarChart.destroy(); radarChart = null; }
}
