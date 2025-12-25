/**
 * Emotions Module - Main Entry Point
 * Initializes and orchestrates all emotion analytics components.
 * 
 * @module emotions/index
 */

import * as Core from './core.js';
import * as Charts from './charts.js';
import * as Moments from './moments.js';

// ============================================================================
// Public API
// ============================================================================

export const EMOTION_COLORS = Core.EMOTION_COLORS;
export const SPEAKER_COLORS = Core.SPEAKER_COLORS;

/**
 * Initialize emotions dashboard in a container
 * @param {string} containerId - Root container ID
 * @param {Object} options - Configuration options
 */
export async function initEmotionsDashboard(containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error('Emotions container not found:', containerId);
        return;
    }

    const { period = 'today', onLoad = null } = options;

    Core.setState({ loading: true, period });

    try {
        // Fetch analytics data
        const analytics = await Core.fetchEmotionAnalytics(period);
        Core.setState({ analytics, loading: false });

        // Render all components
        renderDashboard(containerId, analytics);

        if (onLoad) onLoad(analytics);

        return analytics;

    } catch (error) {
        console.error('Failed to initialize emotions dashboard:', error);
        Core.setState({ loading: false, error: error.message });
        container.innerHTML = `<p class="text-muted" style="text-align: center; padding: 2rem;">Failed to load emotion analytics.</p>`;
        throw error;
    }
}

/**
 * Render all dashboard components
 */
function renderDashboard(containerId, analytics) {
    const summary = analytics.summary || analytics;
    const timeline = analytics.timeline || {};
    const speakers = analytics.speakers || [];
    const totals = summary.emotion_totals || summary.emotions || {};

    // Update stat cards
    updateStatCards(summary);

    // Render charts
    // Render charts
    Charts.renderDistributionChart('emotion-distribution-chart', totals, {
        onSliceClick: (emotion) => {
            console.log('Filtering by emotion:', emotion);
            filterMomentsByEmotion(emotion);
            // Also update the filter UI chunks if they exist?
            // For now, focusing on the Moment feed filtering as per user request "get all of the data" (likely context).
        }
    });
    Charts.renderTrendChart('emotion-trend-chart', timeline);
    Charts.renderRadarChart('speech-radar-chart', speakers);

    // Render speaker table
    renderSpeakerTable('speaker-table-body', speakers);

    // Add interaction to speaker table
    const speakerTable = document.getElementById('speaker-table-body');
    if (speakerTable && !speakerTable.dataset.interactive) {
        speakerTable.dataset.interactive = 'true';
        speakerTable.addEventListener('click', (e) => {
            const row = e.target.closest('tr');
            if (row && row.dataset.speaker) {
                const speaker = row.dataset.speaker;
                console.log('Filtering by speaker:', speaker);
                // Toggle speaker selection in state
                const state = Core.getState();
                if (state.selectedSpeakers.has(speaker)) {
                    state.selectedSpeakers.delete(speaker);
                } else {
                    state.selectedSpeakers.clear(); // Single select or multi? User "click on speakers" -> likely single select focus?
                    // Let's support toggle behavior, but cleared others?
                    // Legacy typically enabled single select behavior for drill down.
                    state.selectedSpeakers.add(speaker);
                }

                // Re-render dashboard to reflect speaker filter?
                // If we filter mostly everything by speaker, we should reload/re-render.
                // But simpler: just filter moments for now, or update state and re-render visual elements.
                // For this pass, I will just filter MOMENTS by speaker if the Moments module supports it.
                // Checking Moments module... 'Moments.loadAndRenderMoments' accepts filters.
                Moments.loadAndRenderMoments('emotion-moment-feed', { speaker: speaker });

                // Visual feedback on row
                Array.from(speakerTable.children).forEach(r => r.classList.toggle('selected-row', r.dataset.speaker === speaker));
            }
        });
    }

    // Render moments
    Moments.loadAndRenderMoments('emotion-moment-feed');
}

/**
 * Update stat cards with summary data
 */
function updateStatCards(summary) {
    const elements = {
        'stat-joy': summary.joy_count || summary.joy || 0,
        'stat-negative': (summary.anger || 0) + (summary.sadness || 0) + (summary.fear || 0),
        'stat-total': summary.total_analyzed || summary.total || 0,
        'stat-pace': summary.avg_pace_wpm ? Math.round(summary.avg_pace_wpm) : '-',
        'stat-pitch': summary.avg_pitch_mean ? Math.round(summary.avg_pitch_mean) : '-'
    };

    Object.entries(elements).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    });
}

/**
 * Render speaker comparison table
 */
function renderSpeakerTable(tableBodyId, speakers) {
    const tbody = document.getElementById(tableBodyId);
    if (!tbody) return;

    if (!speakers.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-muted">No speaker data available.</td></tr>';
        return;
    }

    tbody.innerHTML = speakers.slice(0, 10).map(speaker => {
        const color = Core.getSpeakerColor(speaker.speaker);
        const dominantEmotion = speaker.dominant_emotion || 'neutral';
        const emotionColor = Core.EMOTION_COLORS[dominantEmotion] || '#6b7280';
        const speakerName = speaker.speaker || 'Unknown';

        return `
      <tr data-speaker="${Core.escapeHtml(speakerName)}" style="cursor: pointer; transition: background-color 0.2s;">
        <td>
          <span style="display: inline-flex; align-items: center; gap: 0.5rem;">
            <span style="width: 10px; height: 10px; border-radius: 50%; background: ${color};"></span>
            ${Core.escapeHtml(speakerName)}
          </span>
        </td>
        <td>${speaker.segments || 0}</td>
        <td>
          <span style="background: ${emotionColor}20; color: ${emotionColor}; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem;">
            ${dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1)}
          </span>
        </td>
        <td>${speaker.speech_metrics?.pace_wpm ? Math.round(speaker.speech_metrics.pace_wpm) + ' wpm' : '-'}</td>
        <td>${speaker.speech_metrics?.pitch_mean ? Math.round(speaker.speech_metrics.pitch_mean) + ' Hz' : '-'}</td>
      </tr>
    `;
    }).join('');
}

/**
 * Refresh dashboard with new period
 */
export async function refreshDashboard(containerId, period) {
    Core.setState({ period });
    return initEmotionsDashboard(containerId, { period });
}

/**
 * Filter moments by emotion
 */
export function filterMomentsByEmotion(emotion) {
    const state = Core.getState();
    if (emotion === 'all') {
        state.selectedEmotions = new Set(['joy', 'anger', 'sadness', 'fear', 'neutral', 'surprise']);
    } else {
        state.selectedEmotions.clear();
        state.selectedEmotions.add(emotion);
    }
    Moments.loadAndRenderMoments('emotion-moment-feed', { emotion: emotion === 'all' ? null : emotion });
}

/**
 * Cleanup all resources
 */
export function destroy() {
    Charts.destroyAllCharts();
    Core.resetState();
}

// Re-export sub-modules for direct access
export { Core, Charts, Moments };
