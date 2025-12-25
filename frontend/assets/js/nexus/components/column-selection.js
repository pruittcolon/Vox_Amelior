/**
 * NexusAI Column Selection Component
 * Gemma AI or manual column selection modes.
 *
 * @module nexus/components/column-selection
 */

import { askGemmaForColumns } from '../core/api.js';
import {
    getColumnSelection,
    setSelectionMode,
    setTargetColumn,
    setFeatureColumns,
    excludeTarget,
    resetColumnSelection,
    getUploadState
} from '../core/state.js';

// ============================================================================
// State
// ============================================================================

let selectionCallbacks = {
    onModeChange: null,    // (mode) => void
    onTargetChange: null,  // (target) => void
    onFeaturesChange: null // (features) => void
};

// ============================================================================
// Public API
// ============================================================================

/**
 * Register selection event callbacks.
 * @param {Object} callbacks
 */
export function registerSelectionCallbacks(callbacks) {
    selectionCallbacks = { ...selectionCallbacks, ...callbacks };
}

/**
 * Initialize column selection UI.
 * @param {string} panelSelector - Selector for the selection panel
 */
export function initColumnSelection(panelSelector) {
    const panel = document.querySelector(panelSelector);
    if (!panel) {
        console.warn('Column selection panel not found:', panelSelector);
        return;
    }

    // Setup mode toggle buttons
    const gemmaBtn = panel.querySelector('#mode-gemma');
    const manualBtn = panel.querySelector('#mode-manual');

    if (gemmaBtn) {
        gemmaBtn.addEventListener('click', () => switchMode('gemma'));
    }
    if (manualBtn) {
        manualBtn.addEventListener('click', () => switchMode('manual'));
    }

    // Setup target dropdown
    const targetSelect = panel.querySelector('#target-select');
    if (targetSelect) {
        targetSelect.addEventListener('change', (e) => {
            handleTargetChange(e.target.value);
        });
    }
}

/**
 * Populate column selectors with available columns.
 */
export function populateColumnSelectors() {
    const { columns } = getUploadState();
    if (!columns || columns.length === 0) return;

    // Populate target dropdown
    const targetSelect = document.getElementById('target-select');
    if (targetSelect) {
        targetSelect.innerHTML = '<option value="">-- Select target column --</option>' +
            columns.map(col => `<option value="${escapeHtml(col)}">${escapeHtml(col)}</option>`).join('');
    }

    // Populate feature checkboxes
    const featureGrid = document.getElementById('feature-grid');
    if (featureGrid) {
        featureGrid.innerHTML = columns.map(col => `
      <label class="feature-checkbox" data-column="${escapeHtml(col)}">
        <input type="checkbox" value="${escapeHtml(col)}">
        <span class="checkbox-custom"></span>
        <span class="feature-label" title="${escapeHtml(col)}">${escapeHtml(col)}</span>
      </label>
    `).join('');

        // Add click handlers
        featureGrid.querySelectorAll('.feature-checkbox').forEach(checkbox => {
            checkbox.addEventListener('click', (e) => {
                if (e.target.tagName === 'INPUT') return; // Let native checkbox handle it
                handleFeatureToggle(checkbox);
            });
        });
    }

    updateFeatureCount();
}

/**
 * Switch between Gemma and manual selection modes.
 * @param {'gemma'|'manual'} mode
 */
export function switchMode(mode) {
    setSelectionMode(mode);

    // Update UI
    const gemmaBtn = document.getElementById('mode-gemma');
    const manualBtn = document.getElementById('mode-manual');
    const gemmaPanel = document.getElementById('gemma-mode-panel');
    const manualPanel = document.getElementById('manual-mode-panel');

    if (gemmaBtn) gemmaBtn.classList.toggle('active', mode === 'gemma');
    if (manualBtn) manualBtn.classList.toggle('active', mode === 'manual');
    if (gemmaPanel) gemmaPanel.style.display = mode === 'gemma' ? 'block' : 'none';
    if (manualPanel) manualPanel.style.display = mode === 'manual' ? 'block' : 'none';

    if (selectionCallbacks.onModeChange) {
        selectionCallbacks.onModeChange(mode);
    }
}

/**
 * Get current selection (Gemma recommendation or manual).
 * @returns {Promise<{target: string, features: string[]}|null>}
 */
export async function getCurrentSelection() {
    const { columns } = getUploadState();
    const selection = getColumnSelection();

    if (selection.mode === 'gemma') {
        // Ask Gemma
        const recommendation = await askGemmaForColumns(columns, selection.excludedTargets);
        if (recommendation) {
            setTargetColumn(recommendation.target);
            setFeatureColumns(recommendation.features);
            return recommendation;
        }
        return null;
    } else {
        // Manual selection
        return {
            target: selection.target,
            features: selection.features
        };
    }
}

/**
 * Select all feature columns.
 */
export function selectAllFeatures() {
    const selection = getColumnSelection();
    const featureGrid = document.getElementById('feature-grid');
    if (!featureGrid) return;

    const newFeatures = [];
    featureGrid.querySelectorAll('.feature-checkbox').forEach(cb => {
        const column = cb.dataset.column;
        if (column && column !== selection.target) {
            cb.classList.add('selected');
            const input = cb.querySelector('input');
            if (input) input.checked = true;
            newFeatures.push(column);
        }
    });

    setFeatureColumns(newFeatures);
    updateFeatureCount();

    if (selectionCallbacks.onFeaturesChange) {
        selectionCallbacks.onFeaturesChange(newFeatures);
    }
}

/**
 * Deselect all feature columns.
 */
export function deselectAllFeatures() {
    const featureGrid = document.getElementById('feature-grid');
    if (!featureGrid) return;

    featureGrid.querySelectorAll('.feature-checkbox').forEach(cb => {
        cb.classList.remove('selected');
        const input = cb.querySelector('input');
        if (input) input.checked = false;
    });

    setFeatureColumns([]);
    updateFeatureCount();

    if (selectionCallbacks.onFeaturesChange) {
        selectionCallbacks.onFeaturesChange([]);
    }
}

/**
 * Add failed target to exclusion list and reset for retry.
 * @param {string} target
 */
export function excludeFailedTarget(target) {
    excludeTarget(target);
    resetColumnSelection();
}

// ============================================================================
// Internal Handlers
// ============================================================================

function handleTargetChange(target) {
    const selection = getColumnSelection();
    setTargetColumn(target);

    // Remove target from features if selected
    if (target && selection.features.includes(target)) {
        const newFeatures = selection.features.filter(f => f !== target);
        setFeatureColumns(newFeatures);

        // Update UI
        const checkbox = document.querySelector(`.feature-checkbox[data-column="${target}"]`);
        if (checkbox) {
            checkbox.classList.remove('selected');
            const input = checkbox.querySelector('input');
            if (input) input.checked = false;
        }
    }

    updateFeatureCount();

    if (selectionCallbacks.onTargetChange) {
        selectionCallbacks.onTargetChange(target);
    }
}

function handleFeatureToggle(checkbox) {
    const column = checkbox.dataset.column;
    const selection = getColumnSelection();

    // Prevent selecting target as feature
    if (column === selection.target) {
        return;
    }

    checkbox.classList.toggle('selected');
    const input = checkbox.querySelector('input');
    if (input) input.checked = checkbox.classList.contains('selected');

    // Update features list
    let newFeatures;
    if (checkbox.classList.contains('selected')) {
        newFeatures = [...selection.features, column];
    } else {
        newFeatures = selection.features.filter(f => f !== column);
    }
    setFeatureColumns(newFeatures);

    updateFeatureCount();

    if (selectionCallbacks.onFeaturesChange) {
        selectionCallbacks.onFeaturesChange(newFeatures);
    }
}

function updateFeatureCount() {
    const selection = getColumnSelection();
    const countEl = document.getElementById('feature-count');
    if (countEl) {
        const count = selection.features.length;
        countEl.textContent = `${count} feature${count !== 1 ? 's' : ''} selected`;
    }
}

// ============================================================================
// Utility
// ============================================================================

function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
