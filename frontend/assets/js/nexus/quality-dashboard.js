/**
 * NexusAI Quality Dashboard Controller
 * 
 * Manages the Quality Intelligence Dashboard UI interactions,
 * 3D viewer initialization, and data loading.
 * 
 * @module nexus/quality-dashboard
 * @security CSP-compliant, JWT-authenticated API calls
 */

// ============================================================================
// State
// ============================================================================

/** @type {Quality3DViewer|null} */
let quality3DViewer = null;

// ============================================================================
// Dashboard Visibility
// ============================================================================

/**
 * Show the Quality Intelligence Dashboard.
 * Initializes the 3D viewer if not already done and loads available insights files.
 * @returns {Promise<void>}
 */
async function showQualityDashboard() {
    const section = document.getElementById('quality-dashboard-section');
    if (!section) return;

    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth' });

    // Load available insights files
    await loadInsightsFileList();

    // Initialize 3D viewer if not already done
    if (!quality3DViewer && typeof window.Quality3DViewer !== 'undefined') {
        quality3DViewer = new window.Quality3DViewer('quality-3d-viewer');
        await quality3DViewer.init();
        quality3DViewer.start();
    }
}

/**
 * Hide the Quality Intelligence Dashboard.
 * Stops the 3D animation to conserve resources.
 */
function hideQualityDashboard() {
    const section = document.getElementById('quality-dashboard-section');
    if (section) {
        section.style.display = 'none';
    }
    if (quality3DViewer) {
        quality3DViewer.stop();
    }
}

// ============================================================================
// Data Loading
// ============================================================================

/**
 * Load available insights files into the dropdown selector.
 * @returns {Promise<void>}
 */
async function loadInsightsFileList() {
    const select = document.getElementById('quality-file-select');
    if (!select) return;

    try {
        const response = await fetch('/quality-insights/files', {
            credentials: 'include'
        });

        if (response.ok) {
            const data = await response.json();
            select.innerHTML = '<option value="">Select insights file...</option>';

            (data.files || []).forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file;
                select.appendChild(option);
            });
        }
    } catch (e) {
        console.warn('Could not load insights files:', e);
    }
}

/**
 * Load quality data and render the 3D visualization.
 * @returns {Promise<void>}
 */
async function loadQualityData() {
    const select = document.getElementById('quality-file-select');
    const file = select?.value;

    if (!file) {
        alert('Please select an insights file first');
        return;
    }

    const container = document.getElementById('quality-3d-viewer');
    container?.classList.add('loading');

    try {
        // Initialize 3D viewer if needed
        if (!quality3DViewer && typeof window.Quality3DViewer !== 'undefined') {
            quality3DViewer = new window.Quality3DViewer('quality-3d-viewer');
            await quality3DViewer.init();
        }

        // Load data
        const analysisData = await quality3DViewer.loadData(file);

        // Update stats
        updateQualityStats(analysisData);

        // Load business savings
        await loadBusinessSavings(file);

        container?.classList.remove('loading');
        container?.classList.add('ready');

    } catch (e) {
        console.error('Failed to load quality data:', e);
        container?.classList.remove('loading');
        alert('Failed to load quality data: ' + e.message);
    }
}

// ============================================================================
// Stats Display
// ============================================================================

/**
 * Update the quality stats display with analysis data.
 * @param {Object} data - Analysis data from the quality insights API
 */
function updateQualityStats(data) {
    if (!data) return;

    const mlReadiness = document.getElementById('stat-ml-readiness');
    const avgQuality = document.getElementById('stat-avg-quality');
    const problemRows = document.getElementById('stat-problem-rows');
    const dataPoints = document.getElementById('stat-data-points');

    if (mlReadiness) mlReadiness.textContent = (data.ml_readiness || 0).toFixed(1) + '%';
    if (avgQuality) avgQuality.textContent = (data.avg_overall || 0).toFixed(1) + '/10';
    if (problemRows) problemRows.textContent = (data.problem_row_count || 0).toLocaleString();
    if (dataPoints) dataPoints.textContent = (data.row_count || 0).toLocaleString();

    // Update recommendations
    const recsContainer = document.getElementById('quality-recommendations');
    if (recsContainer && data.recommendations) {
        recsContainer.innerHTML = data.recommendations.map(rec => `
            <div style="padding: 0.75rem 1rem; background: #ffffff; border-radius: 0.5rem; border-left: 4px solid var(--vox-primary);">
                ${rec}
            </div>
        `).join('');
    }
}

/**
 * Load business savings data and update the savings panel.
 * @param {string} file - The insights file name
 * @returns {Promise<void>}
 */
async function loadBusinessSavings(file) {
    try {
        const response = await fetch('/quality-insights/business-savings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ insights_file: file, use_llm: false })
        });

        if (response.ok) {
            const data = await response.json();

            // Update savings panel
            const panel = document.getElementById('savings-panel');
            const total = document.getElementById('savings-total');
            const issues = document.getElementById('savings-issues');

            if (panel) panel.style.display = 'block';
            if (total) total.textContent = data.estimated_annual_savings || '$0';

            if (issues && data.high_impact_issues) {
                issues.innerHTML = data.high_impact_issues.slice(0, 3).map(issue => `
                    <div class="savings-issue">
                        <span class="savings-issue-name">${issue.issue}</span>
                        <span class="savings-issue-impact">${issue.impact}</span>
                    </div>
                `).join('');
            }
        }
    } catch (e) {
        console.warn('Could not load business savings:', e);
    }
}

// ============================================================================
// Cleanup
// ============================================================================

/**
 * Dispose of 3D viewer resources.
 * Call this when navigating away from the page.
 */
function disposeQualityDashboard() {
    if (quality3DViewer) {
        quality3DViewer.dispose();
        quality3DViewer = null;
    }
}

// ============================================================================
// Exports (attach to window.NexusUI)
// ============================================================================

// Ensure NexusUI exists
window.NexusUI = window.NexusUI || {};

// Attach quality dashboard functions
window.NexusUI.showQualityDashboard = showQualityDashboard;
window.NexusUI.hideQualityDashboard = hideQualityDashboard;
window.NexusUI.loadInsightsFileList = loadInsightsFileList;
window.NexusUI.loadQualityData = loadQualityData;
window.NexusUI.updateQualityStats = updateQualityStats;
window.NexusUI.loadBusinessSavings = loadBusinessSavings;
window.NexusUI.disposeQualityDashboard = disposeQualityDashboard;

// Cleanup on page unload to prevent memory leaks
window.addEventListener('beforeunload', disposeQualityDashboard);
