/**
 * Emotions Module - Moment Cards
 * Renders emotional moment cards from transcript segments.
 * 
 * @module emotions/moments
 */

import { EMOTION_COLORS, escapeHtml, formatTime, getSpeakerColor, fetchMomentFeed, getState } from './core.js';

// ============================================================================
// Moment Card Rendering
// ============================================================================

export function renderMomentCard(moment) {
    const emotion = moment.emotion || 'neutral';
    const color = EMOTION_COLORS[emotion] || '#6b7280';
    const speakerColor = getSpeakerColor(moment.speaker);

    return `
    <div class="moment-card" data-emotion="${emotion}" data-speaker="${escapeHtml(moment.speaker || '')}">
      <div class="moment-card-header">
        <span class="moment-speaker" style="background: ${speakerColor}20; color: ${speakerColor}; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 500;">
          ${escapeHtml(moment.speaker || 'Speaker')}
        </span>
        <span class="emotion-badge" style="background: ${color}20; color: ${color}; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem;">
          ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}
        </span>
        <span class="moment-time" style="font-size: 0.75rem; color: var(--text-secondary); margin-left: auto;">
          ${formatTime(moment.start_time || moment.start || 0)}
        </span>
      </div>
      <div class="moment-card-body">
        ${escapeHtml(moment.text || moment.content || '')}
      </div>
      <div class="moment-card-footer">
        <span class="confidence" style="font-size: 0.75rem; color: var(--text-muted);">
          ${moment.confidence ? Math.round(moment.confidence * 100) + '% confidence' : ''}
        </span>
        ${moment.job_id ? `
          <button class="btn btn-ghost btn-sm" onclick="window.location.href='transcripts.html?job=${moment.job_id}&t=${moment.start_time || 0}'">
            View Context →
          </button>
        ` : ''}
      </div>
    </div>
  `;
}

export function renderMomentFeed(containerId, moments = [], options = {}) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const { showEmpty = true, maxItems = 20 } = options;

    if (!moments.length) {
        container.innerHTML = showEmpty
            ? '<p class="text-muted" style="text-align: center; padding: 2rem;">No emotional moments captured yet.</p>'
            : '';
        return;
    }

    container.innerHTML = moments
        .slice(0, maxItems)
        .map(m => renderMomentCard(m))
        .join('');
}

// ============================================================================
// Moment Filtering
// ============================================================================

export function filterMoments(moments, filters = {}) {
    const { emotion, speaker, minConfidence = 0 } = filters;

    return moments.filter(m => {
        if (emotion && emotion !== 'all' && m.emotion !== emotion) return false;
        if (speaker && m.speaker !== speaker) return false;
        if (m.confidence && m.confidence < minConfidence) return false;
        return true;
    });
}

// ============================================================================
// Async Moment Loading
// ============================================================================

export async function loadAndRenderMoments(containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
    <div class="skeleton" style="height: 80px; margin-bottom: 1rem;"></div>
    <div class="skeleton" style="height: 80px; margin-bottom: 1rem;"></div>
    <div class="skeleton" style="height: 80px;"></div>
  `;

    try {
        const data = await fetchMomentFeed(options);
        const moments = data.moments || data.items || data || [];
        renderMomentFeed(containerId, moments, options);
        return moments;
    } catch (error) {
        console.error('Error loading moments:', error);
        container.innerHTML = '<p class="text-muted" style="text-align: center; padding: 2rem;">Failed to load moments.</p>';
        return [];
    }
}
