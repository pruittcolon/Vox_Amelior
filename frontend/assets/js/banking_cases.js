/**
 * Banking Cases Module - Case Management System (API-Connected)
 * =============================================================
 * Full case lifecycle: create, assign, escalate, resolve
 * 
 * ZERO-MOCK POLICY: ALL data fetched from /fiserv/api/v1/cases
 * NO hardcoded demo data.
 * 
 * Dependencies: banking_core.js (getCsrfToken)
 * Used by: banking.html
 * 
 * @module banking_cases
 * @version 2.0.0
 */

// =================================================================
// CONFIGURATION
// =================================================================

const CASE_API_BASE = '/fiserv/api/v1/cases';

// =================================================================
// STATE MANAGEMENT
// =================================================================

const CaseState = {
    cases: [],
    filteredCases: [],
    selectedCase: null,
    stats: {
        critical: 0,
        high: 0,
        open: 0,
        resolved: 0,
        avgResolutionHours: 0
    }
};

// =================================================================
// CASE LOADING & FILTERING (API-CONNECTED)
// =================================================================

/**
 * Load cases from real API endpoint
 * ZERO-MOCK: All data from PostgreSQL via fiserv-service
 */
async function loadCases() {
    try {
        console.log('[Cases] Loading cases from API...');

        const response = await fetch(CASE_API_BASE, {
            headers: {
                'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : '',
                'Content-Type': 'application/json'
            },
            credentials: 'include'
        });

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();

        if (data.success && data.cases) {
            CaseState.cases = data.cases.map(formatCaseFromApi);
            CaseState.filteredCases = [...CaseState.cases];
            console.log(`[Cases] Loaded ${CaseState.cases.length} cases from API`);
        } else {
            CaseState.cases = [];
            CaseState.filteredCases = [];
            console.log('[Cases] No cases found');
        }

        await loadCaseStats();
        renderCases();

    } catch (error) {
        console.error('[Cases] Error loading cases:', error);
        CaseState.cases = [];
        CaseState.filteredCases = [];
        renderCases();

        // Show error in UI
        const container = document.getElementById('casesList');
        if (container) {
            container.innerHTML = `<div class="results-empty" style="color: var(--neon-rose);">
                Failed to load cases: ${error.message}
            </div>`;
        }
    }
}

/**
 * Load case statistics from API
 */
async function loadCaseStats() {
    try {
        const response = await fetch(`${CASE_API_BASE}/stats`, {
            headers: {
                'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : ''
            },
            credentials: 'include'
        });

        if (response.ok) {
            const data = await response.json();
            if (data.success && data.stats) {
                CaseState.stats = data.stats;
                updateCaseStats();
            }
        }
    } catch (error) {
        console.error('[Cases] Error loading stats:', error);
    }
}

/**
 * Format API case response to frontend format
 * @param {Object} apiCase - Case from API
 * @returns {Object} Formatted case
 */
function formatCaseFromApi(apiCase) {
    return {
        id: apiCase.id,
        type: apiCase.type,
        subject: apiCase.subject,
        description: apiCase.description || '',
        memberId: apiCase.member_id || '',
        accountId: apiCase.account_id || '',
        priority: apiCase.priority || 'medium',
        status: apiCase.status || 'open',
        assigneeId: apiCase.assignee_id || '',
        assigneeName: apiCase.assignee_name || '',
        createdAt: apiCase.created_at,
        dueDate: apiCase.due_date,
        resolvedAt: apiCase.resolved_at,
        resolutionSummary: apiCase.resolution_summary || '',
        timeline: (apiCase.timeline || []).map(t => ({
            type: t.event_type,
            timestamp: t.timestamp,
            user: t.user_name,
            note: t.note
        }))
    };
}

/**
 * Filter cases based on UI controls
 */
function filterCases() {
    const typeFilter = document.getElementById('caseTypeFilter')?.value || 'all';
    const statusFilter = document.getElementById('caseStatusFilter')?.value || 'all';
    const priorityFilter = document.getElementById('casePriorityFilter')?.value || 'all';
    const searchQuery = (document.getElementById('caseSearchQuery')?.value || '').toLowerCase();

    CaseState.filteredCases = CaseState.cases.filter(c => {
        if (typeFilter !== 'all' && c.type !== typeFilter) return false;
        if (statusFilter !== 'all' && c.status !== statusFilter) return false;
        if (priorityFilter !== 'all' && c.priority !== priorityFilter) return false;
        if (searchQuery && !c.subject.toLowerCase().includes(searchQuery) &&
            !c.id.toLowerCase().includes(searchQuery) &&
            !(c.memberId || '').toLowerCase().includes(searchQuery)) return false;
        return true;
    });

    renderCases();
}

/**
 * Update case statistics from loaded data
 */
function updateCaseStats() {
    const stats = CaseState.stats;

    const criticalEl = document.getElementById('casesCritical');
    const highEl = document.getElementById('casesHigh');
    const openEl = document.getElementById('casesOpen');
    const resolvedEl = document.getElementById('casesResolved');
    const avgTimeEl = document.getElementById('casesAvgTime');

    if (criticalEl) criticalEl.textContent = stats.critical || 0;
    if (highEl) highEl.textContent = stats.high || 0;
    if (openEl) openEl.textContent = stats.open || 0;
    if (resolvedEl) resolvedEl.textContent = stats.resolved || 0;
    if (avgTimeEl) avgTimeEl.textContent = (stats.avg_resolution_hours || 0).toFixed(1);
}

/**
 * Render cases list
 */
function renderCases() {
    const container = document.getElementById('casesList');
    if (!container) return;

    const titleEl = document.getElementById('caseQueueTitle');
    if (titleEl) {
        titleEl.textContent = `My Case Queue (${CaseState.filteredCases.length})`;
    }

    if (CaseState.filteredCases.length === 0) {
        container.innerHTML = '<div class="results-empty">No cases match your filters</div>';
        return;
    }

    container.innerHTML = CaseState.filteredCases.map(c => {
        const priorityColors = {
            critical: '#ef4444',
            high: '#f59e0b',
            medium: '#3b82f6',
            low: '#6b7280'
        };
        const statusColors = {
            open: '#3b82f6',
            in_progress: '#10b981',
            escalated: '#f59e0b',
            closed: '#6b7280'
        };
        const typeIcons = {
            fraud: '🔍',
            dispute: '⚖️',
            complaint: '📞',
            compliance: '📋'
        };

        const isOverdue = c.dueDate && new Date(c.dueDate) < new Date() && c.status !== 'closed';

        return `
            <div onclick="openCaseDetail('${c.id}')" 
                 style="display: flex; align-items: center; gap: 1rem; padding: 1rem; background: ${CaseState.selectedCase === c.id ? '#eff6ff' : 'white'}; border-radius: 8px; margin-bottom: 0.5rem; border: 2px solid ${isOverdue ? '#ef4444' : CaseState.selectedCase === c.id ? '#3b82f6' : '#e5e7eb'}; cursor: pointer; transition: all 0.2s;"
                 onmouseover="this.style.borderColor='#3b82f6'" 
                 onmouseout="this.style.borderColor='${isOverdue ? '#ef4444' : CaseState.selectedCase === c.id ? '#3b82f6' : '#e5e7eb'}'"
                 data-testid="case-${c.id}">
                <div style="font-size: 1.5rem;">${typeIcons[c.type] || '📋'}</div>
                <div style="flex: 1;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-weight: 600; color: #1f2937;">${c.id}</span>
                        ${isOverdue ? '<span style="font-size: 0.65rem; background: #fee2e2; color: #dc2626; padding: 0.15rem 0.5rem; border-radius: 4px;">OVERDUE</span>' : ''}
                    </div>
                    <div style="color: #374151; margin: 0.25rem 0;">${c.subject}</div>
                    <div style="font-size: 0.8rem; color: #6b7280;">
                        Member: ${c.memberId || 'N/A'} • ${c.assigneeName || 'Unassigned'}
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; gap: 0.25rem; align-items: flex-end;">
                    <span style="padding: 0.2rem 0.6rem; background: ${priorityColors[c.priority]}22; color: ${priorityColors[c.priority]}; border-radius: 4px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase;">
                        ${c.priority}
                    </span>
                    <span style="padding: 0.2rem 0.6rem; background: ${statusColors[c.status]}22; color: ${statusColors[c.status]}; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">
                        ${(c.status || '').replace('_', ' ')}
                    </span>
                </div>
            </div>
        `;
    }).join('');
}

// =================================================================
// CASE DETAIL FUNCTIONS
// =================================================================

/**
 * Open case detail panel
 * @param {string} caseId - Case ID
 */
async function openCaseDetail(caseId) {
    try {
        const response = await fetch(`${CASE_API_BASE}/${caseId}`, {
            headers: {
                'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : ''
            },
            credentials: 'include'
        });

        if (!response.ok) {
            throw new Error(`Failed to load case: ${response.status}`);
        }

        const data = await response.json();
        if (!data.success || !data.case) {
            throw new Error('Case not found');
        }

        const caseData = formatCaseFromApi(data.case);
        CaseState.selectedCase = caseId;

        renderCaseDetail(caseData);
        renderCases(); // Re-render to highlight selected

    } catch (error) {
        console.error('[Cases] Error loading case detail:', error);
        if (typeof showNotification === 'function') {
            showNotification(`Error: ${error.message}`, 'error');
        }
    }
}

/**
 * Render case detail panel
 * @param {Object} caseData - Case data
 */
function renderCaseDetail(caseData) {
    document.getElementById('caseDetailTitle').textContent = `Case ${caseData.id}`;
    document.getElementById('caseDetailSubject').textContent = caseData.subject;
    document.getElementById('caseDetailType').textContent = caseData.type.charAt(0).toUpperCase() + caseData.type.slice(1);
    document.getElementById('caseDetailMember').textContent = caseData.memberId || 'N/A';
    document.getElementById('caseDetailAssignee').textContent = caseData.assigneeName || 'Unassigned';
    document.getElementById('caseDetailDue').textContent = caseData.dueDate ? new Date(caseData.dueDate).toLocaleDateString() : '—';

    // Priority badge
    const priorityEl = document.getElementById('caseDetailPriority');
    const priorityColors = { critical: '#ef4444', high: '#f59e0b', medium: '#3b82f6', low: '#6b7280' };
    priorityEl.textContent = caseData.priority.charAt(0).toUpperCase() + caseData.priority.slice(1);
    priorityEl.style.background = priorityColors[caseData.priority];

    // Status badge
    const statusEl = document.getElementById('caseDetailStatus');
    const statusColors = { open: '#3b82f6', in_progress: '#10b981', escalated: '#f59e0b', closed: '#6b7280' };
    statusEl.textContent = (caseData.status || '').replace('_', ' ');
    statusEl.style.background = statusColors[caseData.status];

    renderCaseTimeline(caseData.timeline || []);
    document.getElementById('caseDetailPanel').style.display = 'block';
    document.getElementById('caseDetailPanel').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Render case timeline
 * @param {Array} timeline - Timeline events
 */
function renderCaseTimeline(timeline) {
    const container = document.getElementById('caseTimeline');
    if (!container) return;

    const typeColors = {
        created: '#3b82f6',
        assigned: '#6366f1',
        escalated: '#f59e0b',
        note: '#10b981',
        resolved: '#22c55e',
        status_change: '#8b5cf6',
        closed: '#6b7280'
    };

    container.innerHTML = timeline.map(e => `
        <div style="position: relative; padding-bottom: 1.5rem;">
            <div style="position: absolute; left: -2rem; top: 0; width: 12px; height: 12px; background: ${typeColors[e.type] || '#6b7280'}; border-radius: 50%; border: 2px solid white;"></div>
            <div style="font-size: 0.75rem; color: #6b7280;">${new Date(e.timestamp).toLocaleString()} • ${e.user}</div>
            <div style="color: #1f2937;">${e.note}</div>
        </div>
    `).join('');
}

/**
 * Close case detail panel
 */
function closeCaseDetail() {
    CaseState.selectedCase = null;
    document.getElementById('caseDetailPanel').style.display = 'none';
    renderCases();
}

// =================================================================
// CASE ACTIONS (API-CONNECTED)
// =================================================================

/**
 * Show create case modal
 */
function showCreateCaseModal() {
    document.getElementById('createCaseModal').style.display = 'flex';
}

/**
 * Hide create case modal
 */
function hideCreateCaseModal() {
    document.getElementById('createCaseModal').style.display = 'none';
    // Clear form
    ['newCaseType', 'newCasePriority', 'newCaseSubject', 'newCaseMemberId',
        'newCaseAccountId', 'newCaseDescription', 'newCaseAssignee', 'newCaseDueDate'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = '';
        });
}

/**
 * Create a new case via API
 */
async function createCase() {
    const type = document.getElementById('newCaseType')?.value;
    const priority = document.getElementById('newCasePriority')?.value || 'medium';
    const subject = (document.getElementById('newCaseSubject')?.value || '').trim();
    const memberId = (document.getElementById('newCaseMemberId')?.value || '').trim();
    const accountId = (document.getElementById('newCaseAccountId')?.value || '').trim();
    const description = (document.getElementById('newCaseDescription')?.value || '').trim();
    const assignee = document.getElementById('newCaseAssignee')?.value || '';
    const dueDate = document.getElementById('newCaseDueDate')?.value || '';

    if (!type || !subject) {
        alert('Please fill in required fields (Type and Subject)');
        return;
    }

    try {
        const response = await fetch(CASE_API_BASE, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : ''
            },
            credentials: 'include',
            body: JSON.stringify({
                case_type: type,
                subject,
                description,
                member_id: memberId || null,
                account_id: accountId || null,
                priority,
                assignee_name: assignee || null,
                due_date: dueDate ? new Date(dueDate).toISOString() : null
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to create case: ${response.status}`);
        }

        const data = await response.json();

        hideCreateCaseModal();
        await loadCases(); // Reload from API

        if (typeof showNotification === 'function') {
            showNotification(`Case ${data.case?.id || ''} created`, 'success');
        }

        console.log('[Cases] Created case:', data.case?.id);

    } catch (error) {
        console.error('[Cases] Error creating case:', error);
        alert(`Error creating case: ${error.message}`);
    }
}

/**
 * Update case via API
 * @param {string} caseId - Case ID
 * @param {Object} updates - Fields to update
 */
async function updateCaseApi(caseId, updates) {
    try {
        const response = await fetch(`${CASE_API_BASE}/${caseId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : ''
            },
            credentials: 'include',
            body: JSON.stringify(updates)
        });

        if (!response.ok) {
            throw new Error(`Failed to update case: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('[Cases] Update error:', error);
        throw error;
    }
}

/**
 * Add note to case via API
 * @param {string} caseId - Case ID
 * @param {string} note - Note text
 */
async function addNoteApi(caseId, note) {
    const response = await fetch(`${CASE_API_BASE}/${caseId}/notes`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : ''
        },
        credentials: 'include',
        body: JSON.stringify({ note, user_name: 'Current User' })
    });

    if (!response.ok) {
        throw new Error(`Failed to add note: ${response.status}`);
    }

    return await response.json();
}

/**
 * Assign case to analyst
 */
async function assignCase() {
    if (!CaseState.selectedCase) return;

    const analyst = prompt('Assign to (enter name):');
    if (!analyst) return;

    try {
        await updateCaseApi(CaseState.selectedCase, {
            assignee_name: analyst
        });

        await openCaseDetail(CaseState.selectedCase);
        await loadCases();

        if (typeof showNotification === 'function') {
            showNotification(`Case assigned to ${analyst}`, 'success');
        }
    } catch (error) {
        alert(`Error assigning case: ${error.message}`);
    }
}

/**
 * Escalate case
 */
async function escalateCase() {
    if (!CaseState.selectedCase) return;

    const reason = prompt('Escalation reason:');
    if (!reason) return;

    try {
        await updateCaseApi(CaseState.selectedCase, {
            status: 'escalated',
            assignee_name: 'Supervisor'
        });

        await addNoteApi(CaseState.selectedCase, `Escalated: ${reason}`);
        await openCaseDetail(CaseState.selectedCase);
        await loadCases();

        if (typeof showNotification === 'function') {
            showNotification('Case escalated to supervisor', 'success');
        }
    } catch (error) {
        alert(`Error escalating case: ${error.message}`);
    }
}

/**
 * Add note to case
 */
async function addCaseNote() {
    if (!CaseState.selectedCase) return;

    const note = prompt('Add note:');
    if (!note) return;

    try {
        await addNoteApi(CaseState.selectedCase, note);
        await openCaseDetail(CaseState.selectedCase);

        if (typeof showNotification === 'function') {
            showNotification('Note added', 'success');
        }
    } catch (error) {
        alert(`Error adding note: ${error.message}`);
    }
}

/**
 * Link transaction to case
 */
async function linkTransaction() {
    if (!CaseState.selectedCase) return;

    const txId = prompt('Transaction ID to link:');
    if (!txId) return;

    try {
        await addNoteApi(CaseState.selectedCase, `Linked transaction: ${txId}`);
        await openCaseDetail(CaseState.selectedCase);

        if (typeof showNotification === 'function') {
            showNotification(`Transaction ${txId} linked`, 'success');
        }
    } catch (error) {
        alert(`Error linking transaction: ${error.message}`);
    }
}

/**
 * Resolve case
 */
async function resolveCase() {
    if (!CaseState.selectedCase) return;

    const disposition = prompt('Resolution summary:');
    if (!disposition) return;

    try {
        await updateCaseApi(CaseState.selectedCase, {
            status: 'closed',
            resolution_summary: disposition
        });

        closeCaseDetail();
        await loadCases();

        if (typeof showNotification === 'function') {
            showNotification('Case resolved', 'success');
        }
    } catch (error) {
        alert(`Error resolving case: ${error.message}`);
    }
}

/**
 * Export cases to CSV
 */
function exportCases() {
    const headers = ['Case ID', 'Type', 'Subject', 'Member', 'Priority', 'Status', 'Assignee', 'Created', 'Due'];
    const rows = CaseState.filteredCases.map(c => [
        c.id,
        c.type,
        `"${(c.subject || '').replace(/"/g, '""')}"`,
        c.memberId,
        c.priority,
        c.status,
        c.assigneeName || 'Unassigned',
        c.createdAt ? new Date(c.createdAt).toLocaleDateString() : '',
        c.dueDate ? new Date(c.dueDate).toLocaleDateString() : ''
    ]);

    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `cases_export_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();

    if (typeof showNotification === 'function') {
        showNotification('Cases exported', 'success');
    }
}

// =================================================================
// INITIALIZATION
// =================================================================

/**
 * Initialize cases module
 */
function initCasesModule() {
    console.log('[Cases] Initializing (API-Connected, Zero-Mock)...');
    loadCases();
    console.log('[Cases] Module initialized');
}

// =================================================================
// GLOBAL EXPORTS
// =================================================================

window.CaseState = CaseState;
window.loadCases = loadCases;
window.filterCases = filterCases;
window.renderCases = renderCases;
window.openCaseDetail = openCaseDetail;
window.closeCaseDetail = closeCaseDetail;
window.showCreateCaseModal = showCreateCaseModal;
window.hideCreateCaseModal = hideCreateCaseModal;
window.createCase = createCase;
window.assignCase = assignCase;
window.escalateCase = escalateCase;
window.addCaseNote = addCaseNote;
window.linkTransaction = linkTransaction;
window.resolveCase = resolveCase;
window.exportCases = exportCases;
window.initCasesModule = initCasesModule;

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCasesModule);
} else {
    initCasesModule();
}

console.log('[Cases] Module loaded - API-Connected Case Management ready');
