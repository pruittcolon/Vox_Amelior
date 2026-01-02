/**
 * High-Tech Drag & Drop Column Mapper
 * 
 * Replaces legacy dropdowns with a visual zone-based interface.
 * Supports multi-select with Ctrl/Cmd+Click and Shift+Click.
 */

import {
    setTargetColumn,
    setFeatureColumns,
    getColumnSelection
} from '../core/state.js';

let draggedItem = null;
let currentZones = {
    target: [],
    features: [],
    ignored: []
};

// Multi-select state
let selectedItems = new Set();
let lastSelectedItem = null;

/**
 * Initialize the Drag & Drop Mapper.
 * @param {string|HTMLElement} containerOrSelector - CSS selector or container element.
 */
export function initColumnMapper(containerOrSelector) {
    const container = typeof containerOrSelector === 'string'
        ? document.querySelector(containerOrSelector)
        : containerOrSelector;

    if (!container) {
        console.error('[ColumnMapper] Container not found:', containerOrSelector);
        return;
    }

    container.innerHTML = `
        <div class="column-mapper-container">
            <!-- Selection hint -->
            <div class="mapper-selection-hint">
                Tip: Ctrl+Click to select multiple columns, Shift+Click for range selection
            </div>
            
            <!-- Target Zone -->
            <div class="mapper-zone zone-target" data-zone="target">
                <div class="mapper-zone-header">
                    <span>TARGET</span>
                    <span class="zone-count" id="count-target">0</span>
                </div>
                <div class="mapper-drop-area" id="zone-target-items"></div>
                <div class="zone-empty-msg">Drop target here</div>
            </div>

            <!-- Features Zone -->
            <div class="mapper-zone zone-features" data-zone="features">
                <div class="mapper-zone-header">
                    <span>FEATURES</span>
                    <span class="zone-count" id="count-features">0</span>
                </div>
                <div class="mapper-drop-area" id="zone-features-items"></div>
                <div class="zone-empty-msg" style="display:none">Drop features here</div>
            </div>

            <!-- Ignored Zone -->
            <div class="mapper-zone zone-ignore" data-zone="ignored">
                <div class="mapper-zone-header">
                    <span>IGNORED</span>
                    <span class="zone-count" id="count-ignored">0</span>
                </div>
                <div class="mapper-drop-area" id="zone-ignored-items"></div>
                <div class="zone-empty-msg" style="display:none">Drop unused columns here</div>
            </div>
        </div>
    `;

    setupDragAndDrop();
    setupClickSelection();
}

/**
 * Auto-populate zones based on columns and classification.
 * @param {string[]} allColumns - All available columns.
 * @param {Object} classification - Result from classifyColumns API {target, features}.
 */
export function populateColumnMapper(allColumns, classification) {
    // Reset
    currentZones = { target: [], features: [], ignored: [] };
    selectedItems.clear();
    lastSelectedItem = null;

    const targetCol = classification?.target || null;
    const featureCols = classification?.features || [];

    allColumns.forEach(col => {
        if (col === targetCol) {
            currentZones.target.push(col);
        } else if (featureCols.includes(col)) {
            currentZones.features.push(col);
        } else {
            currentZones.ignored.push(col);
        }
    });

    renderZones();
    syncState();
}

function renderZones() {
    renderZoneItems('target', currentZones.target);
    renderZoneItems('features', currentZones.features);
    renderZoneItems('ignored', currentZones.ignored);
    updateCounts();
}

function renderZoneItems(zoneName, items) {
    const container = document.getElementById('zone-' + zoneName + '-items');
    if (!container) return;

    const emptyMsg = container.parentElement.querySelector('.zone-empty-msg');

    container.innerHTML = '';

    if (items.length === 0) {
        if (emptyMsg) emptyMsg.style.display = 'block';
    } else {
        if (emptyMsg) emptyMsg.style.display = 'none';
        items.forEach(item => {
            const el = document.createElement('div');
            el.className = 'mapper-item';
            if (selectedItems.has(item)) {
                el.classList.add('selected');
            }
            el.draggable = true;
            el.dataset.col = item;
            el.innerHTML = '<span>' + item + '</span><span class="item-type">Col</span>';
            container.appendChild(el);
        });
    }
}

function updateCounts() {
    const targetCount = document.getElementById('count-target');
    const featuresCount = document.getElementById('count-features');
    const ignoredCount = document.getElementById('count-ignored');

    if (targetCount) targetCount.textContent = currentZones.target.length;
    if (featuresCount) featuresCount.textContent = currentZones.features.length;
    if (ignoredCount) ignoredCount.textContent = currentZones.ignored.length;
}

/**
 * Setup click handlers for multi-select
 */
function setupClickSelection() {
    document.addEventListener('click', function (e) {
        const item = e.target.closest('.mapper-item');
        if (!item) {
            // Click outside items - clear selection if not dragging
            if (!e.target.closest('.mapper-zone')) {
                selectedItems.clear();
                updateSelectionUI();
            }
            return;
        }

        const colName = item.dataset.col;
        if (!colName) return;

        // Handle multi-select
        if (e.shiftKey && lastSelectedItem) {
            // Range select
            e.preventDefault();
            selectRange(lastSelectedItem, colName);
        } else if (e.ctrlKey || e.metaKey) {
            // Toggle individual
            e.preventDefault();
            if (selectedItems.has(colName)) {
                selectedItems.delete(colName);
            } else {
                selectedItems.add(colName);
            }
            lastSelectedItem = colName;
        } else {
            // Single select (clear others)
            selectedItems.clear();
            selectedItems.add(colName);
            lastSelectedItem = colName;
        }

        updateSelectionUI();
    });
}

/**
 * Select a range of columns between start and end
 */
function selectRange(start, end) {
    // Get all items in order across all zones
    const allItems = [
        ...currentZones.target,
        ...currentZones.features,
        ...currentZones.ignored
    ];

    const startIdx = allItems.indexOf(start);
    const endIdx = allItems.indexOf(end);

    if (startIdx === -1 || endIdx === -1) return;

    const minIdx = Math.min(startIdx, endIdx);
    const maxIdx = Math.max(startIdx, endIdx);

    for (let i = minIdx; i <= maxIdx; i++) {
        selectedItems.add(allItems[i]);
    }
}

/**
 * Update the visual selection state of all items
 */
function updateSelectionUI() {
    document.querySelectorAll('.mapper-item').forEach(item => {
        const isSelected = selectedItems.has(item.dataset.col);
        item.classList.toggle('selected', isSelected);
    });

    // Update selection count hint
    const hint = document.querySelector('.mapper-selection-hint');
    if (hint) {
        if (selectedItems.size > 1) {
            hint.textContent = selectedItems.size + ' columns selected - drag to move all';
            hint.classList.add('active');
        } else {
            hint.textContent = 'Tip: Ctrl+Click to select multiple columns, Shift+Click for range selection';
            hint.classList.remove('active');
        }
    }
}

function setupDragAndDrop() {
    const zones = document.querySelectorAll('.mapper-zone');

    // Zone Handlers
    zones.forEach(zone => {
        zone.addEventListener('dragover', function (e) {
            e.preventDefault();
            if (!zone.classList.contains('drag-over')) {
                zone.classList.add('drag-over');
            }
        });

        zone.addEventListener('dragleave', function (e) {
            zone.classList.remove('drag-over');
        });

        zone.addEventListener('drop', function (e) {
            e.preventDefault();
            zone.classList.remove('drag-over');

            const targetZoneType = zone.dataset.zone;
            if (!targetZoneType) return;

            // Check if multi-select drag
            const multiData = e.dataTransfer.getData('application/json');
            if (multiData) {
                try {
                    const columns = JSON.parse(multiData);
                    console.log('[ColumnMapper] Multi-drop:', columns.length, 'columns to', targetZoneType);
                    moveMultipleColumns(columns, targetZoneType);
                    return;
                } catch (err) {
                    // Fall through to single item
                }
            }

            // Single item drop
            const colName = e.dataTransfer.getData('text/plain');
            if (!colName) {
                console.warn('[ColumnMapper] Invalid drop - no column data');
                return;
            }

            moveColumn(colName, targetZoneType);
        });
    });

    // Item Handler Delegation (for dynamic items)
    document.addEventListener('dragstart', function (e) {
        if (e.target.classList.contains('mapper-item')) {
            const colName = e.target.dataset.col;
            draggedItem = e.target;
            e.target.classList.add('is-dragging');

            // If dragging an unselected item, select only it
            if (!selectedItems.has(colName)) {
                selectedItems.clear();
                selectedItems.add(colName);
                updateSelectionUI();
            }

            // Set drag data - either multi or single
            if (selectedItems.size > 1) {
                const columns = Array.from(selectedItems);
                e.dataTransfer.setData('application/json', JSON.stringify(columns));
                e.dataTransfer.setData('text/plain', columns.join(','));

                // Mark all selected as dragging
                document.querySelectorAll('.mapper-item.selected').forEach(item => {
                    item.classList.add('is-dragging');
                });
            } else {
                e.dataTransfer.setData('text/plain', colName);
            }

            e.dataTransfer.effectAllowed = 'move';
        }
    });

    document.addEventListener('dragend', function (e) {
        if (e.target.classList.contains('mapper-item')) {
            // Remove dragging class from all items
            document.querySelectorAll('.mapper-item.is-dragging').forEach(item => {
                item.classList.remove('is-dragging');
            });
            draggedItem = null;

            // Clear selection after drop
            selectedItems.clear();
            updateSelectionUI();
        }
    });
}

/**
 * Move multiple columns to a new zone
 */
function moveMultipleColumns(columns, targetZone) {
    if (!columns || !columns.length) return;

    console.log('[ColumnMapper] Moving', columns.length, 'columns to', targetZone);

    // For target zone, only allow one item
    if (targetZone === 'target') {
        // Only move the first selected column to target
        moveColumn(columns[0], 'target');
        // Move rest to features
        for (let i = 1; i < columns.length; i++) {
            moveColumn(columns[i], 'features');
        }
    } else {
        // Move all to the zone
        columns.forEach(colName => {
            moveColumnInternal(colName, targetZone);
        });
        renderZones();
        syncState();
    }
}

/**
 * Move a column internally without re-rendering (for batch moves)
 */
function moveColumnInternal(colName, targetZone) {
    // Remove from old zone
    ['target', 'features', 'ignored'].forEach(z => {
        const idx = currentZones[z].indexOf(colName);
        if (idx > -1) {
            currentZones[z].splice(idx, 1);
        }
    });

    // Add to new zone
    if (!currentZones[targetZone].includes(colName)) {
        currentZones[targetZone].push(colName);
    }
}

/**
 * Move a column to a new zone and update state.
 */
function moveColumn(colName, targetZone) {
    if (!colName || typeof colName !== 'string') {
        console.error('[ColumnMapper] moveColumn called with invalid colName:', colName);
        return;
    }
    if (!targetZone || !['target', 'features', 'ignored'].includes(targetZone)) {
        console.error('[ColumnMapper] moveColumn called with invalid targetZone:', targetZone);
        return;
    }

    console.log('[ColumnMapper] Moving "' + colName + '" to "' + targetZone + '"');

    // Remove from old zone
    ['target', 'features', 'ignored'].forEach(z => {
        const idx = currentZones[z].indexOf(colName);
        if (idx > -1) {
            currentZones[z].splice(idx, 1);
        }
    });

    // Add to new zone
    if (targetZone === 'target') {
        // Swap if exists
        if (currentZones.target.length > 0) {
            const oldTarget = currentZones.target[0];
            if (oldTarget !== colName) {
                currentZones.ignored.push(oldTarget);
            }
        }
        currentZones.target = [colName];
    } else {
        currentZones[targetZone].push(colName);
    }

    renderZones();
    syncState();
}

function syncState() {
    const target = currentZones.target[0] || null;
    const features = currentZones.features;

    setTargetColumn(target);
    setFeatureColumns(features);

    console.log('[Mapper] Synced State: Target=' + target + ', Features=' + features.length);
}

/**
 * Reset the column mapper to show it again
 */
export function resetColumnMapper() {
    selectedItems.clear();
    lastSelectedItem = null;
}
