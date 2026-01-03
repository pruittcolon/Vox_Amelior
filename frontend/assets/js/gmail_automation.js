/**
 * Gmail Automation JavaScript Module
 * 
 * Handles OAuth flow, email fetching, and Gemma-powered analysis
 * with SSE streaming for real-time progress updates.
 */

// State
let fetchedEmails = [];
let selectedEmailIds = new Set();
let currentPreset = 'summarize';

/**
 * Safely parse JSON, returning null on failure
 */
function safeParse(str) {
    if (!str || str === 'undefined') return null;
    try {
        return JSON.parse(str);
    } catch {
        return null;
    }
}

/**
 * Check Gmail OAuth connection status
 */
async function checkGmailStatus() {
    try {
        const response = await fetch('/api/gmail/oauth/status');
        const data = await response.json();

        updateConnectionUI(data.connected, data.email);
    } catch (error) {
        console.error('Failed to check Gmail status:', error);
        updateConnectionUI(false);
    }
}

/**
 * Update connection UI based on status
 */
function updateConnectionUI(connected, email = null) {
    const statusEl = document.getElementById('connection-status');
    const emailEl = document.getElementById('connection-email');
    const connectBtn = document.getElementById('connect-btn');
    const disconnectBtn = document.getElementById('disconnect-btn');
    const fetchSection = document.getElementById('fetch-section');

    if (connected) {
        statusEl.textContent = 'Connected';
        statusEl.className = 'status-connected';
        emailEl.textContent = email || '';
        connectBtn.style.display = 'none';
        disconnectBtn.style.display = 'inline-block';
        fetchSection.style.opacity = '1';
        fetchSection.style.pointerEvents = 'auto';
    } else {
        statusEl.textContent = 'Not Connected';
        statusEl.className = 'status-disconnected';
        emailEl.textContent = 'Connect your Gmail to get started';
        connectBtn.style.display = 'inline-block';
        disconnectBtn.style.display = 'none';
        fetchSection.style.opacity = '0.5';
        fetchSection.style.pointerEvents = 'none';
    }
}

/**
 * Initiate Gmail OAuth connection
 */
async function connectGmail() {
    try {
        const response = await fetch('/api/gmail/oauth/url');
        const data = await response.json();

        if (data.authorization_url) {
            // Open OAuth popup
            const popup = window.open(
                data.authorization_url,
                'gmail_oauth',
                'width=600,height=700,left=200,top=100'
            );

            // Listen for callback
            window.addEventListener('message', async (event) => {
                if (event.data.type === 'gmail_oauth_callback') {
                    try {
                        const callbackResponse = await fetch('/api/gmail/oauth/callback', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                code: event.data.code,
                                state: event.data.state
                            })
                        });

                        if (callbackResponse.ok) {
                            popup?.close();
                            checkGmailStatus();
                        }
                    } catch (error) {
                        console.error('OAuth callback failed:', error);
                    }
                }
            });
        }
    } catch (error) {
        console.error('Failed to get OAuth URL:', error);
        alert('Failed to connect Gmail. Please try again.');
    }
}

/**
 * Disconnect Gmail
 */
async function disconnectGmail() {
    if (!confirm('Disconnect your Gmail account?')) return;

    try {
        await fetch('/api/gmail/oauth/disconnect', { method: 'POST' });
        checkGmailStatus();

        // Clear state
        fetchedEmails = [];
        selectedEmailIds.clear();
        document.getElementById('email-section').style.display = 'none';
        document.getElementById('results-section').style.display = 'none';
    } catch (error) {
        console.error('Failed to disconnect Gmail:', error);
    }
}

/**
 * Fetch emails from Gmail
 */
async function fetchEmails() {
    const fetchBtn = document.getElementById('fetch-btn');
    const emailSection = document.getElementById('email-section');

    fetchBtn.disabled = true;
    fetchBtn.innerHTML = '<i data-lucide="loader-2" style="width: 16px; height: 16px; animation: spin 1s linear infinite;"></i> Fetching...';

    try {
        const response = await fetch('/api/gmail/emails/fetch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                timeframe: document.getElementById('timeframe-select').value,
                max_results: parseInt(document.getElementById('max-emails-select').value),
                query: document.getElementById('search-query').value || null
            })
        });

        const data = await response.json();

        if (data.success && data.emails) {
            fetchedEmails = data.emails;
            // Auto-select all emails
            fetchedEmails.forEach(email => selectedEmailIds.add(email.id));
            renderEmailList();
            emailSection.style.display = 'block';

            // Auto-analyze if emails found
            if (fetchedEmails.length > 0) {
                setTimeout(() => analyzeEmails(), 500);
            }
        } else {
            alert('Failed to fetch emails: ' + (data.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Failed to fetch emails:', error);
        alert('Failed to fetch emails. Please try again.');
    } finally {
        fetchBtn.disabled = false;
        fetchBtn.innerHTML = '<i data-lucide="download" style="width: 16px; height: 16px;"></i> Fetch Emails';
        lucide.createIcons();
    }
}

/**
 * Render email list
 */
function renderEmailList() {
    const listEl = document.getElementById('email-list');
    const countEl = document.getElementById('email-count');

    countEl.textContent = fetchedEmails.length;

    if (fetchedEmails.length === 0) {
        listEl.innerHTML = `
            <div class="empty-state">
                <i data-lucide="inbox" style="width: 48px; height: 48px;"></i>
                <p>No emails found for the selected criteria</p>
            </div>
        `;
        lucide.createIcons();
        return;
    }

    listEl.innerHTML = fetchedEmails.map(email => `
        <div class="email-item ${selectedEmailIds.has(email.id) ? 'selected' : ''}" 
             onclick="toggleEmailSelection('${email.id}')">
            <input type="checkbox" class="email-checkbox" 
                   ${selectedEmailIds.has(email.id) ? 'checked' : ''} 
                   onclick="event.stopPropagation(); toggleEmailSelection('${email.id}')">
            <div class="email-content">
                <div class="email-subject">${escapeHtml(email.subject)}</div>
                <div class="email-meta">
                    <span>${escapeHtml(email.sender_name || email.sender)}</span>
                    <span>${formatDate(email.date)}</span>
                </div>
                <div class="email-snippet">${escapeHtml(email.snippet)}</div>
            </div>
        </div>
    `).join('');

    updateSelectedCount();
}

/**
 * Toggle email selection
 */
function toggleEmailSelection(emailId) {
    if (selectedEmailIds.has(emailId)) {
        selectedEmailIds.delete(emailId);
    } else {
        selectedEmailIds.add(emailId);
    }
    renderEmailList();
}

/**
 * Select all emails
 */
function selectAllEmails() {
    fetchedEmails.forEach(email => selectedEmailIds.add(email.id));
    renderEmailList();
}

/**
 * Deselect all emails
 */
function deselectAllEmails() {
    selectedEmailIds.clear();
    renderEmailList();
}

/**
 * Update selected count
 */
function updateSelectedCount() {
    const countEl = document.getElementById('selected-count');
    const analyzeBtn = document.getElementById('analyze-btn');

    countEl.textContent = selectedEmailIds.size;
    analyzeBtn.disabled = selectedEmailIds.size === 0;
}

/**
 * Analyze selected emails with Gemma
 */
async function analyzeEmails() {
    if (selectedEmailIds.size === 0) return;

    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsSection = document.getElementById('results-section');
    const resultsContainer = document.getElementById('analysis-results');
    const progressBar = document.getElementById('progress-bar');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    // Get selected preset
    const activePreset = document.querySelector('.preset-btn.active');
    const preset = activePreset?.dataset.preset || 'summarize';
    const customPrompt = preset === 'custom'
        ? document.getElementById('custom-prompt').value
        : null;

    if (preset === 'custom' && !customPrompt?.trim()) {
        alert('Please enter a custom prompt');
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i data-lucide="loader-2" style="width: 16px; height: 16px; animation: spin 1s linear infinite;"></i> Analyzing...';

    resultsSection.style.display = 'block';
    progressBar.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Starting analysis...';
    resultsContainer.innerHTML = '';

    // Build query params
    const params = new URLSearchParams({
        email_ids: Array.from(selectedEmailIds).join(','),
        preset: preset,
        max_tokens: 512,
        temperature: 0.4
    });

    if (customPrompt) {
        params.append('custom_prompt', customPrompt);
    }

    try {
        // Use SSE for streaming
        const eventSource = new EventSource(`/api/gmail/emails/analyze/stream?${params.toString()}`);

        eventSource.addEventListener('progress', (e) => {
            const data = safeParse(e.data);
            if (data) {
                progressFill.style.width = `${(data.progress || 0) * 100}%`;
                progressText.textContent = data.message || 'Processing...';
            }
        });

        eventSource.addEventListener('result', (e) => {
            const data = safeParse(e.data);
            if (data && data.result) {
                appendResult(data.result);
            }
        });

        eventSource.addEventListener('error', (e) => {
            const data = safeParse(e.data);
            progressText.textContent = `Error: ${data?.error || 'Unknown error'}`;
            eventSource.close();
        });

        eventSource.addEventListener('done', (e) => {
            const data = safeParse(e.data);
            progressBar.style.display = 'none';
            progressText.textContent = data?.message || 'Analysis complete';
            eventSource.close();

            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `<i data-lucide="sparkles" style="width: 16px; height: 16px;"></i> Analyze Selected (<span id="selected-count">${selectedEmailIds.size}</span>)`;
            lucide.createIcons();
        });

        eventSource.onerror = () => {
            progressText.textContent = 'Connection error. Please try again.';
            eventSource.close();

            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `<i data-lucide="sparkles" style="width: 16px; height: 16px;"></i> Analyze Selected (<span id="selected-count">${selectedEmailIds.size}</span>)`;
            lucide.createIcons();
        };

    } catch (error) {
        console.error('Analysis failed:', error);
        progressText.textContent = 'Analysis failed. Please try again.';

        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `<i data-lucide="sparkles" style="width: 16px; height: 16px;"></i> Analyze Selected (<span id="selected-count">${selectedEmailIds.size}</span>)`;
        lucide.createIcons();
    }
}

/**
 * Append analysis result to results container
 */
function appendResult(result) {
    const container = document.getElementById('analysis-results');

    const resultHtml = `
        <div class="result-item">
            <div class="result-header">
                <div class="result-subject">${escapeHtml(result.subject)}</div>
                <span class="text-muted" style="font-size: 0.8rem;">
                    ${result.tokens_used} tokens | ${result.processing_time_ms}ms
                </span>
            </div>
            <div class="result-analysis">${escapeHtml(result.analysis)}</div>
        </div>
    `;

    container.insertAdjacentHTML('beforeend', resultHtml);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Format date for display
 */
function formatDate(dateStr) {
    try {
        const date = new Date(dateStr);
        const now = new Date();
        const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));

        if (diffDays === 0) {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } else if (diffDays === 1) {
            return 'Yesterday';
        } else if (diffDays < 7) {
            return date.toLocaleDateString([], { weekday: 'short' });
        } else {
            return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
        }
    } catch {
        return '';
    }
}

// CSS for spinner animation
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);
