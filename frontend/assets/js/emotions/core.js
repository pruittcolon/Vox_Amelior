/**
 * Emotions Module - Core State & Configuration
 * Provides shared state and configuration for emotion analytics.
 * 
 * @module emotions/core
 */

// ============================================================================
// Constants
// ============================================================================

export const EMOTION_COLORS = {
    joy: '#10b981',
    anger: '#ef4444',
    sadness: '#3b82f6',
    fear: '#8b5cf6',
    surprise: '#f59e0b',
    disgust: '#84cc16',
    neutral: '#6b7280'
};

export const SPEAKER_COLORS = [
    '#a78bfa', '#f472b6', '#fb923c', '#4ade80', '#38bdf8',
    '#facc15', '#f87171', '#22d3ee', '#c084fc', '#a3e635'
];

export const METRIC_LABELS = {
    pace_wpm: 'Pace (WPM)',
    pitch_mean: 'Pitch (Hz)',
    pitch_std: 'Pitch Var',
    energy_mean: 'Energy',
    energy_std: 'Energy Var',
    spectral_centroid_mean: 'Spectral'
};

export const METRIC_COLORS = {
    pace_wpm: '#3b82f6',
    pitch_mean: '#f59e0b',
    pitch_std: '#ef4444',
    energy_mean: '#10b981',
    energy_std: '#8b5cf6',
    spectral_centroid_mean: '#06b6d4'
};

// ============================================================================
// State Management
// ============================================================================

export function createEmotionsState() {
    return {
        period: 'today',
        selectedEmotions: new Set(['joy', 'anger', 'sadness', 'fear', 'neutral', 'surprise']),
        selectedSpeakers: new Set(),
        speakerColorMap: {},
        selectedSpeakersOrder: [],
        analytics: null,
        loading: false,
        error: null
    };
}

let _state = createEmotionsState();

export function getState() {
    return _state;
}

export function setState(updates) {
    _state = { ..._state, ...updates };
    return _state;
}

export function resetState() {
    _state = createEmotionsState();
    return _state;
}

// ============================================================================
// Utility Functions
// ============================================================================

export function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

export function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

export function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    return m > 0 ? `${m} min` : `${Math.round(seconds)}s`;
}

export function formatPercentage(value) {
    return Number.isFinite(value) ? `${Math.round(value)}%` : '-';
}

export function hexToRgba(hex, alpha = 1) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return `rgba(148,163,184,${alpha})`;
    return `rgba(${parseInt(result[1], 16)},${parseInt(result[2], 16)},${parseInt(result[3], 16)},${alpha})`;
}

export function getSpeakerColor(speakerId) {
    const state = getState();
    if (!speakerId) return '#a78bfa';
    if (!state.speakerColorMap[speakerId]) {
        const idx = state.selectedSpeakersOrder.indexOf(speakerId);
        const colorIdx = idx >= 0 ? idx : state.selectedSpeakersOrder.push(speakerId) - 1;
        state.speakerColorMap[speakerId] = SPEAKER_COLORS[colorIdx % SPEAKER_COLORS.length];
    }
    return state.speakerColorMap[speakerId];
}

// ============================================================================
// API Functions
// ============================================================================

export async function fetchEmotionStats(period = 'today') {
    try {
        const response = await fetch(`/emotions/stats?period=${period}`, {
            credentials: 'include'
        });
        if (!response.ok) throw new Error('Failed to fetch emotion stats');
        return await response.json();
    } catch (error) {
        console.error('Error fetching emotion stats:', error);
        throw error;
    }
}

export async function fetchEmotionAnalytics(period = 'today') {
    try {
        const response = await fetch(`/emotions/analytics?period=${period}`, {
            credentials: 'include'
        });
        if (!response.ok) throw new Error('Failed to fetch emotion analytics');
        return await response.json();
    } catch (error) {
        console.error('Error fetching emotion analytics:', error);
        throw error;
    }
}

export async function fetchMomentFeed(options = {}) {
    const { limit = 40, offset = 0, emotion = null, speaker = null } = options;
    try {
        let url = `/emotions/moments?limit=${limit}&offset=${offset}`;
        if (emotion) url += `&emotion=${emotion}`;
        if (speaker) url += `&speaker=${encodeURIComponent(speaker)}`;

        const response = await fetch(url, { credentials: 'include' });
        if (!response.ok) throw new Error('Failed to fetch moments');
        return await response.json();
    } catch (error) {
        console.error('Error fetching moment feed:', error);
        throw error;
    }
}
