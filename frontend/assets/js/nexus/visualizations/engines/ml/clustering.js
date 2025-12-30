/**
 * Clustering Visualization
 * Renders 3D scatter plot and cluster profiles for clustering engine results.
 *
 * @module nexus/visualizations/engines/ml/clustering
 */

import { VIZ_COLORS, isLowPowerMode, PERFORMANCE_LIMITS, getClusterColor, escapeHtml } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

/**
 * Build HTML section for Clustering visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string
 */
export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section clustering-full">
            <div class="clustering-layout">
                <div class="clustering-main">
                    <h5>3D Cluster Visualization</h5>
                    <div class="scatter-3d-container" id="cluster-${vizId}"></div>
                    <div class="pca-explanation" id="pca-${vizId}" style="display:none;"></div>
                </div>
                <div class="clustering-sidebar">
                    <h6>Cluster Profiles</h6>
                    <div class="cluster-cards-container" id="cluster-profiles-${vizId}"></div>
                </div>
            </div>
        </div>
    `;
}

// ============================================================================
// Chart Renderer
// ============================================================================

/**
 * Render Clustering visualization charts.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
export function render(data, vizId) {
    if (!data) return;
    renderCluster3D(data, `cluster-${vizId}`);
    renderClusterMeta(data, vizId);
}

// ============================================================================
// Sampling Helpers
// ============================================================================

function hashString(value) {
    if (!value) return 0;
    let hash = 0;
    for (let i = 0; i < value.length; i += 1) {
        hash = ((hash << 5) - hash) + value.charCodeAt(i);
        hash |= 0;
    }
    return Math.abs(hash);
}

function mulberry32(seed) {
    let t = seed;
    return function () {
        t += 0x6D2B79F5;
        let result = Math.imul(t ^ (t >>> 15), 1 | t);
        result ^= result + Math.imul(result ^ (result >>> 7), 61 | result);
        return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
    };
}

function sampleArray(values, maxItems, seedLabel) {
    if (!Array.isArray(values)) return [];
    if (values.length <= maxItems) return values;
    const seed = hashString(seedLabel || 'sample');
    const rand = mulberry32(seed || 1);
    const copy = values.slice();

    for (let i = copy.length - 1; i > 0; i -= 1) {
        const j = Math.floor(rand() * (i + 1));
        [copy[i], copy[j]] = [copy[j], copy[i]];
    }

    return copy.slice(0, maxItems);
}

function sampleClusterPoints(points, maxPoints, seedLabel) {
    if (!Array.isArray(points)) return [];
    if (points.length <= maxPoints) return points;

    const clusters = new Map();
    points.forEach((point) => {
        const key = point.cluster ?? 0;
        if (!clusters.has(key)) clusters.set(key, []);
        clusters.get(key).push(point);
    });

    const perCluster = Math.max(1, Math.floor(maxPoints / clusters.size));
    let sampled = [];

    let idx = 0;
    clusters.forEach((group) => {
        sampled = sampled.concat(sampleArray(group, perCluster, `${seedLabel}-${idx}`));
        idx += 1;
    });

    if (sampled.length > maxPoints) {
        sampled = sampleArray(sampled, maxPoints, `${seedLabel}-trim`);
    }

    return sampled;
}

function generateSampleClusterData(nClusters, labels) {
    const points = [];
    const rand = mulberry32(42);

    labels.forEach((cluster, idx) => {
        const angle = (cluster / nClusters) * Math.PI * 2;
        const radius = 3 + rand() * 2;
        points.push({
            x: Math.cos(angle) * radius + (rand() - 0.5) * 2,
            y: Math.sin(angle) * radius + (rand() - 0.5) * 2,
            z: (rand() - 0.5) * 4,
            cluster: cluster
        });
    });

    return points;
}

// ============================================================================
// 3D Scatter Plot
// ============================================================================

function renderCluster3D(clusterData, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    let points = [];
    let varianceInfo = null;

    // Extract points from PCA data
    if (clusterData.pca_3d && clusterData.pca_3d.points && clusterData.pca_3d.points.length > 0) {
        points = clusterData.pca_3d.points;
        varianceInfo = {
            explained: clusterData.pca_3d.explained_variance || [],
            total: clusterData.pca_3d.total_variance_explained || 0
        };
    }

    // Fallback to labels if no points
    if (points.length === 0 && clusterData.labels && clusterData.labels.length > 0) {
        points = generateSampleClusterData(clusterData.n_clusters || 3, clusterData.labels);
    }

    if (points.length === 0) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No cluster data available for visualization</p>';
        return;
    }

    // Sample points for performance
    const lowPower = isLowPowerMode();
    const maxPoints = lowPower ? PERFORMANCE_LIMITS.clusterPointsLow : PERFORMANCE_LIMITS.clusterPoints;
    if (points.length > maxPoints) {
        points = sampleClusterPoints(points, maxPoints, `cluster-${containerId}`);
    }

    // Use 2D for low power mode
    if (lowPower) {
        renderCluster2D(points, container, varianceInfo);
        return;
    }

    // Build 3D traces
    const traces = [];
    const clusters = [...new Set(points.map(p => p.cluster))].filter(c => c !== -1).sort((a, b) => a - b);

    clusters.forEach((cluster) => {
        const clusterPoints = points.filter(p => p.cluster === cluster);
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            name: `Cluster ${cluster + 1} (${clusterPoints.length} pts)`,
            x: clusterPoints.map(p => p.x),
            y: clusterPoints.map(p => p.y),
            z: clusterPoints.map(p => p.z),
            marker: {
                size: 4,
                color: getClusterColor(cluster),
                opacity: 0.85,
                line: { width: 0.5, color: 'rgba(255,255,255,0.3)' }
            },
            hovertemplate: '<b>Cluster %{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>',
            text: clusterPoints.map(() => cluster + 1)
        });
    });

    // Add noise points if any
    const noisePoints = points.filter(p => p.cluster === -1);
    if (noisePoints.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            name: `Noise (${noisePoints.length} pts)`,
            x: noisePoints.map(p => p.x),
            y: noisePoints.map(p => p.y),
            z: noisePoints.map(p => p.z),
            marker: { size: 3, color: '#94a3b8', opacity: 0.4, symbol: 'x' }
        });
    }

    let title = '';
    if (varianceInfo && varianceInfo.total > 0) {
        title = `PCA 3D Projection (${(varianceInfo.total * 100).toFixed(1)}% variance explained)`;
    }

    Plotly.newPlot(container, traces, {
        title: title ? { text: title, font: { color: VIZ_COLORS.textMuted, size: 12 } } : undefined,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted },
        margin: { l: 0, r: 0, t: title ? 30 : 0, b: 0 },
        scene: {
            xaxis: { title: 'PC1', gridcolor: VIZ_COLORS.border, showbackground: false },
            yaxis: { title: 'PC2', gridcolor: VIZ_COLORS.border, showbackground: false },
            zaxis: { title: 'PC3', gridcolor: VIZ_COLORS.border, showbackground: false },
            bgcolor: 'transparent'
        },
        showlegend: true,
        legend: {
            x: 0.02, y: 0.98,
            bgcolor: 'rgba(255,255,255,0.9)',
            font: { color: VIZ_COLORS.text, size: 10 }
        }
    }, { responsive: true, displayModeBar: true });
}

function renderCluster2D(points, container, varianceInfo) {
    const traces = [];
    const clusters = [...new Set(points.map(p => p.cluster))].filter(c => c !== -1).sort((a, b) => a - b);

    clusters.forEach((cluster) => {
        const clusterPoints = points.filter(p => p.cluster === cluster);
        traces.push({
            type: 'scatter',
            mode: 'markers',
            name: `Cluster ${cluster + 1} (${clusterPoints.length} pts)`,
            x: clusterPoints.map(p => p.x),
            y: clusterPoints.map(p => p.y),
            marker: {
                size: 6,
                color: getClusterColor(cluster),
                opacity: 0.85,
                line: { width: 0.5, color: 'rgba(255,255,255,0.3)' }
            }
        });
    });

    let title = 'Cluster Projection (2D)';
    if (varianceInfo && varianceInfo.total > 0) {
        title = `PCA Projection (2D, ${Math.round(varianceInfo.total * 100)}% variance)`;
    }

    Plotly.newPlot(container, traces, {
        title: { text: `${title} - Low-Graphics Mode`, font: { color: VIZ_COLORS.textMuted, size: 12 } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted },
        margin: { l: 50, r: 20, t: 40, b: 40 },
        xaxis: { title: 'PC1', gridcolor: VIZ_COLORS.border },
        yaxis: { title: 'PC2', gridcolor: VIZ_COLORS.border }
    }, { responsive: true, displayModeBar: false });
}

// ============================================================================
// Cluster Profile Cards
// ============================================================================

function renderClusterMeta(clusterData, vizId) {
    const profilesContainer = document.getElementById(`cluster-profiles-${vizId}`);
    const pcaContainer = document.getElementById(`pca-${vizId}`);

    if (!profilesContainer) return;

    const profiles = clusterData.cluster_profiles || [];
    const pca3d = clusterData.pca_3d || {};
    const componentLoadings = pca3d.component_loadings || [];

    // Render cluster profile cards
    if (profiles.length > 0) {
        profilesContainer.innerHTML = profiles.map(p => `
            <div class="cluster-card" data-cluster="${p.cluster_id}" onclick="window.NexusViz?.highlightCluster?.(${p.cluster_id}, '${vizId}')">
                <div class="cluster-header">
                    <span class="cluster-badge" style="background: ${getClusterColor(p.cluster_id)}">Cluster ${p.cluster_id + 1}</span>
                    <span class="cluster-size">${p.size} pts (${p.percentage.toFixed(1)}%)</span>
                </div>
                <div class="cluster-stats">
                    ${Object.entries(p.feature_stats || {}).slice(0, 3).map(([feat, stats]) => `
                        <div class="stat-row">
                            <span class="stat-name">${escapeHtml(feat.replace(/_/g, ' '))}</span>
                            <span class="stat-value">u=${stats.mean.toFixed(1)} s=${stats.std.toFixed(1)}</span>
                        </div>
                    `).join('')}
                    ${Object.keys(p.feature_stats || {}).length > 3 ? '<div class="stat-more">+ more features...</div>' : ''}
                </div>
            </div>
        `).join('');
    } else {
        profilesContainer.innerHTML = '<p style="color: #94a3b8;">No cluster profiles available</p>';
    }

    // Render PCA explanation if available
    if (pcaContainer && componentLoadings.length > 0) {
        pcaContainer.style.display = 'block';
        pcaContainer.innerHTML = `
            <h6>What the Axes Mean</h6>
            <div class="pca-components">
                ${componentLoadings.map(pc => `
                    <div class="pca-component">
                        <div class="pc-header">
                            <strong>${pc.component}</strong>
                            <span class="pc-variance">${(pc.variance_explained * 100).toFixed(1)}% variance</span>
                        </div>
                        <div class="pc-features">
                            ${pc.top_features.slice(0, 3).map(f => `
                                <span class="pc-feature ${f.loading > 0 ? 'positive' : 'negative'}">
                                    ${f.loading > 0 ? '+' : '-'} ${escapeHtml(f.feature.replace(/_/g, ' '))}
                                </span>
                            `).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
}

/**
 * Highlight a specific cluster in the 3D plot.
 * @param {number} clusterId - Cluster ID to highlight
 * @param {string} vizId - Visualization ID
 */
export function highlightCluster(clusterId, vizId) {
    const container = document.getElementById(`cluster-${vizId}`);
    if (!container) return;

    // Update cluster card selection
    const cards = document.querySelectorAll('.cluster-card');
    cards.forEach(card => {
        card.classList.remove('selected');
        if (parseInt(card.dataset.cluster) === clusterId) {
            card.classList.add('selected');
        }
    });

    // Animate the Plotly trace for this cluster
    try {
        const plotData = container.data;
        if (!plotData) return;

        const newOpacities = plotData.map((trace) => {
            if (trace.name && trace.name.includes(`Cluster ${clusterId + 1}`)) {
                return { 'marker.opacity': 1.0, 'marker.size': 7 };
            }
            return { 'marker.opacity': 0.3, 'marker.size': 4 };
        });

        Plotly.animate(container, {
            data: newOpacities
        }, { transition: { duration: 300 } });

        // Reset after 3 seconds
        setTimeout(() => {
            Plotly.animate(container, {
                data: plotData.map(() => ({ 'marker.opacity': 0.85, 'marker.size': 4 }))
            }, { transition: { duration: 300 } });
        }, 3000);
    } catch (e) {
        console.log('Cluster highlight animation not supported');
    }
}
