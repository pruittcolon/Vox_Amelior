/**
 * Database Viewer Module
 * 
 * Provides Excel-like grid view for uploaded database files.
 * Features: pagination, sorting, searching, PII masking, CSV export.
 * 
 * @module database_viewer
 */

(function (global) {
    'use strict';

    // ========================================================================
    // Configuration
    // ========================================================================

    const DEFAULT_PAGE_SIZE = 50;
    const PAGE_SIZE_OPTIONS = [25, 50, 100];

    // ========================================================================
    // State
    // ========================================================================

    const state = {
        filename: null,
        displayName: null,
        columns: [],
        rows: [],
        piiColumns: [],
        page: 1,
        pageSize: DEFAULT_PAGE_SIZE,
        totalRows: 0,
        totalPages: 0,
        sortBy: null,
        sortOrder: 'asc',
        searchQuery: '',
        loading: false
    };

    // ========================================================================
    // API Functions
    // ========================================================================

    /**
     * Load rows from the database with retry logic.
     * 
     * @param {string} filename - Database filename
     * @param {Object} options - Query options
     * @param {number} retryCount - Current retry attempt (internal)
     * @returns {Promise<Object>} Response data
     */
    async function loadRows(filename, options = {}, retryCount = 0) {
        const MAX_RETRIES = 3;
        const RETRY_DELAY_MS = 1000;

        state.loading = true;
        renderLoadingState();

        const params = new URLSearchParams();
        params.set('page', options.page || state.page);
        params.set('page_size', options.pageSize || state.pageSize);

        if (options.sortBy || state.sortBy) {
            params.set('sort_by', options.sortBy || state.sortBy);
            params.set('sort_order', options.sortOrder || state.sortOrder);
        }

        if (options.search || state.searchQuery) {
            params.set('search', options.search || state.searchQuery);
        }

        try {
            const response = await fetch(`/databases/${filename}/rows?${params.toString()}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error(`Failed to load data: ${response.status}`);
            }

            const data = await response.json();

            // Update state
            state.filename = data.filename;
            state.columns = data.columns;
            state.rows = data.rows;
            state.piiColumns = data.pii_columns || [];
            state.page = data.page;
            state.pageSize = data.page_size;
            state.totalRows = data.total_rows;
            state.totalPages = data.total_pages;

            state.loading = false;
            renderGrid();

            return data;
        } catch (error) {
            // Retry on network errors
            if (retryCount < MAX_RETRIES && (error.name === 'TypeError' || error.message.includes('NetworkError'))) {
                console.warn(`Network error loading rows, retry ${retryCount + 1}/${MAX_RETRIES}...`);
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY_MS * (retryCount + 1)));
                return loadRows(filename, options, retryCount + 1);
            }

            console.error('Error loading database rows:', error);
            state.loading = false;
            renderError(error.message);
            throw error;
        }
    }

    /**
     * Load schema information for a database.
     * 
     * @param {string} filename - Database filename
     * @returns {Promise<Object>} Schema data
     */
    async function loadSchema(filename) {
        try {
            const response = await fetch(`/databases/${filename}/schema`, {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error(`Failed to load schema: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error loading schema:', error);
            throw error;
        }
    }

    // ========================================================================
    // Rendering Functions
    // ========================================================================

    /**
     * Get the container element.
     * @returns {HTMLElement|null}
     */
    function getContainer() {
        return document.getElementById('database-excel-viewer');
    }

    /**
     * Show the database viewer for a specific file.
     * 
     * @param {string} filename - Database filename
     * @param {string} displayName - User-friendly display name
     */
    async function showViewer(filename, displayName = null) {
        state.filename = filename;
        state.displayName = displayName || filename;
        state.page = 1;
        state.sortBy = null;
        state.sortOrder = 'asc';
        state.searchQuery = '';

        const container = getContainer();
        if (container) {
            container.style.display = 'block';
        }

        await loadRows(filename);
    }

    /**
     * Hide the database viewer.
     */
    function hideViewer() {
        const container = getContainer();
        if (container) {
            container.style.display = 'none';
        }
        state.filename = null;
        state.rows = [];
    }

    /**
     * Render loading state.
     */
    function renderLoadingState() {
        const container = getContainer();
        if (!container) return;

        const tableBody = container.querySelector('#db-excel-body');
        if (tableBody) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="100" style="text-align: center; padding: 3rem;">
                        <div class="skeleton" style="height: 20px; width: 60%; margin: 0 auto 1rem;"></div>
                        <div class="skeleton" style="height: 20px; width: 80%; margin: 0 auto 1rem;"></div>
                        <div class="skeleton" style="height: 20px; width: 70%; margin: 0 auto;"></div>
                    </td>
                </tr>
            `;
        }
    }

    /**
     * Render error state.
     * @param {string} message - Error message
     */
    function renderError(message) {
        const container = getContainer();
        if (!container) return;

        const tableBody = container.querySelector('#db-excel-body');
        if (tableBody) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="100" class="excel-empty-state">
                        <i data-lucide="alert-circle" style="width: 48px; height: 48px; color: var(--danger);"></i>
                        <p style="margin-top: 1rem;">${escapeHtml(message)}</p>
                        <button class="btn btn-primary" onclick="DatabaseViewer.reload()">Retry</button>
                    </td>
                </tr>
            `;
        }
        if (window.lucide) window.lucide.createIcons();
    }

    /**
     * Render the Excel-like grid with current data.
     */
    function renderGrid() {
        const container = getContainer();
        if (!container) return;

        // Update header
        const headerTitle = container.querySelector('#db-viewer-title');
        if (headerTitle) {
            headerTitle.textContent = state.displayName || state.filename;
        }

        // Update row count
        const rowCount = container.querySelector('#db-excel-row-count');
        if (rowCount) {
            rowCount.textContent = `${state.totalRows.toLocaleString()} rows`;
        }

        // Render table headers
        const headerRow = container.querySelector('#db-excel-header-row');
        if (headerRow) {
            headerRow.innerHTML = state.columns.map(col => {
                const isPII = state.piiColumns.includes(col);
                const isSorted = state.sortBy === col;
                const sortClass = isSorted ? `sorted-${state.sortOrder}` : '';
                const sortIcon = isSorted
                    ? (state.sortOrder === 'asc' ? '↑' : '↓')
                    : '↕';

                return `
                    <th class="excel-th sortable ${sortClass}" 
                        data-column="${escapeHtml(col)}"
                        onclick="DatabaseViewer.sortBy('${escapeHtml(col)}')"
                        title="${isPII ? 'PII Column (values masked)' : col}">
                        <span>${escapeHtml(col)}</span>
                        ${isPII ? '<i data-lucide="shield" style="width: 12px; height: 12px; margin-left: 4px; opacity: 0.6;"></i>' : ''}
                        <span class="sort-icon">${sortIcon}</span>
                    </th>
                `;
            }).join('');
        }

        // Render table body
        const tableBody = container.querySelector('#db-excel-body');
        if (tableBody) {
            if (state.rows.length === 0) {
                tableBody.innerHTML = `
                    <tr>
                        <td colspan="${state.columns.length}" class="excel-empty-state">
                            <i data-lucide="inbox" style="width: 48px; height: 48px;"></i>
                            <p style="margin-top: 1rem;">No data found</p>
                        </td>
                    </tr>
                `;
            } else {
                tableBody.innerHTML = state.rows.map((row, idx) => {
                    const rowClass = idx % 2 === 0 ? '' : '';
                    return `
                        <tr class="excel-tr ${rowClass}" tabindex="0">
                            ${state.columns.map(col => {
                        const value = row[col];
                        const displayValue = formatCellValue(value, col);
                        const isPII = state.piiColumns.includes(col);
                        const cellClass = isPII ? 'pii-cell' : '';

                        return `<td class="excel-td ${cellClass}" title="${escapeHtml(String(value ?? ''))}">${displayValue}</td>`;
                    }).join('')}
                        </tr>
                    `;
                }).join('');
            }
        }

        // Render pagination
        renderPagination(container);

        // Refresh icons
        if (window.lucide) window.lucide.createIcons();
    }

    /**
     * Render pagination controls.
     * @param {HTMLElement} container - Container element
     */
    function renderPagination(container) {
        const paginationEl = container.querySelector('#db-pagination');
        if (!paginationEl) return;

        const { page, totalPages, pageSize, totalRows } = state;
        const startRow = (page - 1) * pageSize + 1;
        const endRow = Math.min(page * pageSize, totalRows);

        paginationEl.innerHTML = `
            <div class="db-pagination-info">
                Showing ${startRow.toLocaleString()}-${endRow.toLocaleString()} of ${totalRows.toLocaleString()}
            </div>
            <div class="db-pagination-controls">
                <button class="excel-action-btn" 
                        onclick="DatabaseViewer.goToPage(1)" 
                        ${page === 1 ? 'disabled' : ''}>
                    ««
                </button>
                <button class="excel-action-btn" 
                        onclick="DatabaseViewer.goToPage(${page - 1})" 
                        ${page === 1 ? 'disabled' : ''}>
                    ‹ Prev
                </button>
                <span class="db-pagination-page">Page ${page} of ${totalPages}</span>
                <button class="excel-action-btn" 
                        onclick="DatabaseViewer.goToPage(${page + 1})" 
                        ${page >= totalPages ? 'disabled' : ''}>
                    Next ›
                </button>
                <button class="excel-action-btn" 
                        onclick="DatabaseViewer.goToPage(${totalPages})" 
                        ${page >= totalPages ? 'disabled' : ''}>
                    »»
                </button>
            </div>
            <div class="db-pagination-size">
                <label>Rows:</label>
                <select onchange="DatabaseViewer.setPageSize(this.value)">
                    ${PAGE_SIZE_OPTIONS.map(size =>
            `<option value="${size}" ${size === pageSize ? 'selected' : ''}>${size}</option>`
        ).join('')}
                </select>
            </div>
        `;
    }

    // ========================================================================
    // User Actions
    // ========================================================================

    /**
     * Sort by a column.
     * @param {string} column - Column name
     */
    function sortByColumn(column) {
        if (state.sortBy === column) {
            // Toggle order
            state.sortOrder = state.sortOrder === 'asc' ? 'desc' : 'asc';
        } else {
            state.sortBy = column;
            state.sortOrder = 'asc';
        }
        state.page = 1;
        loadRows(state.filename);
    }

    /**
     * Go to a specific page.
     * @param {number} page - Page number
     */
    function goToPage(page) {
        if (page < 1 || page > state.totalPages) return;
        state.page = page;
        loadRows(state.filename);
    }

    /**
     * Set page size.
     * @param {number|string} size - Page size
     */
    function setPageSize(size) {
        state.pageSize = parseInt(size, 10);
        state.page = 1;
        loadRows(state.filename);
    }

    /**
     * Search/filter rows.
     * @param {string} query - Search query
     */
    function search(query) {
        state.searchQuery = query;
        state.page = 1;
        loadRows(state.filename);
    }

    /**
     * Reload current data.
     */
    function reload() {
        if (state.filename) {
            loadRows(state.filename);
        }
    }

    /**
     * Export current view as CSV.
     */
    function exportCSV() {
        if (!state.rows.length) return;

        // Build CSV content
        const headers = state.columns.join(',');
        const rows = state.rows.map(row =>
            state.columns.map(col => {
                const val = row[col];
                if (val === null || val === undefined) return '';
                const str = String(val);
                // Escape quotes and wrap in quotes if needed
                if (str.includes(',') || str.includes('"') || str.includes('\n')) {
                    return `"${str.replace(/"/g, '""')}"`;
                }
                return str;
            }).join(',')
        ).join('\n');

        const csv = `${headers}\n${rows}`;

        // Download
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${state.displayName || state.filename}_export.csv`;
        link.click();
        URL.revokeObjectURL(url);
    }

    /**
     * Export entire database as CSV (not just current page).
     * Downloads directly from server to avoid memory issues with large files.
     */
    async function exportAllCSV() {
        if (!state.filename) {
            console.warn('No database loaded');
            return;
        }

        try {
            // Show loading indicator
            const exportBtn = document.querySelector('[onclick*="exportAllCSV"]');
            const originalText = exportBtn?.innerHTML;
            if (exportBtn) {
                exportBtn.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Downloading...';
                exportBtn.disabled = true;
            }

            const response = await fetch(`/databases/${state.filename}/download`, {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error(`Download failed: ${response.status}`);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = state.displayName || state.filename;
            link.click();
            URL.revokeObjectURL(url);

            console.log(`✅ Downloaded complete file: ${state.filename}`);

            // Restore button
            if (exportBtn) {
                exportBtn.innerHTML = originalText;
                exportBtn.disabled = false;
                if (window.lucide) window.lucide.createIcons();
            }
        } catch (error) {
            console.error('Export all failed:', error);
            alert('Failed to export complete file. Please try again.');
        }
    }

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /**
     * Escape HTML special characters.
     * @param {string} str - Input string
     * @returns {string} Escaped string
     */
    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    /**
     * Format a cell value for display.
     * @param {any} value - Cell value
     * @param {string} column - Column name
     * @returns {string} Formatted HTML
     */
    function formatCellValue(value, column) {
        if (value === null || value === undefined) {
            return '<span class="cell-null">—</span>';
        }

        // PII columns show masked indicator
        if (value === '[PII REDACTED]') {
            return '<span class="cell-pii"><i data-lucide="eye-off" style="width: 12px; height: 12px;"></i> [PII]</span>';
        }

        // Numbers
        if (typeof value === 'number') {
            // Check for currency-like columns
            const colLower = column.toLowerCase();
            if (colLower.includes('price') || colLower.includes('amount') ||
                colLower.includes('revenue') || colLower.includes('cost') ||
                colLower.includes('salary') || colLower.includes('total')) {
                return `<span class="cell-currency">$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>`;
            }
            return `<span class="cell-number">${value.toLocaleString()}</span>`;
        }

        // Booleans
        if (typeof value === 'boolean') {
            return value
                ? '<span class="cell-bool cell-bool-true">✓</span>'
                : '<span class="cell-bool cell-bool-false">✗</span>';
        }

        // Strings - truncate if too long
        const str = String(value);
        if (str.length > 100) {
            return `<span class="cell-preview" title="${escapeHtml(str)}">${escapeHtml(str.substring(0, 100))}…</span>`;
        }

        return escapeHtml(str);
    }

    // ========================================================================
    // Public API
    // ========================================================================

    const DatabaseViewer = {
        show: showViewer,
        hide: hideViewer,
        loadRows,
        loadSchema,
        sortBy: sortByColumn,
        goToPage,
        setPageSize,
        search,
        reload,
        exportCSV,
        exportAllCSV,  // Export complete file
        getState: () => ({ ...state })
    };

    // Export to global scope
    global.DatabaseViewer = DatabaseViewer;

})(typeof window !== 'undefined' ? window : this);
