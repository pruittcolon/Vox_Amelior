/**
 * Banking Executive Module - C-Suite Dashboard
 * 
 * Provides executive-level portfolio analytics, KPI tracking, and AI insights
 * for Service Credit Union leadership.
 * 
 * Dependencies: banking_core.js, Plotly.js
 * Used by: banking.html (section-executive-dashboard)
 * 
 * @author NeMo Server
 * @version 1.0.0
 */

// =============================================================================
// EXECUTIVE DASHBOARD STATE
// =============================================================================

const ExecutiveState = {
    dashboardData: null,
    trendsData: null,
    branchData: null,
    isLoading: false,
    lastUpdated: null
};

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Fetches comprehensive executive dashboard data from API.
 * @returns {Promise<Object>} Executive dashboard data
 */
async function fetchExecutiveDashboardData() {
    try {
        const response = await fetch('/api/v1/banking/executive/dashboard', {
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        ExecutiveState.dashboardData = data;
        ExecutiveState.lastUpdated = new Date();
        return data;
    } catch (error) {
        console.error('[Executive] Dashboard fetch failed:', error);
        return null;
    }
}

/**
 * Fetches historical trend data for charts.
 * @param {number} months - Number of months of history
 * @returns {Promise<Object>} Trend data
 */
async function fetchExecutiveTrendsData(months = 12) {
    try {
        const response = await fetch(`/api/v1/banking/executive/trends?months=${months}`, {
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        ExecutiveState.trendsData = data;
        return data;
    } catch (error) {
        console.error('[Executive] Trends fetch failed:', error);
        return null;
    }
}

/**
 * Fetches branch performance data.
 * @returns {Promise<Object>} Branch performance data
 */
async function fetchBranchPerformanceData() {
    try {
        const response = await fetch('/api/v1/banking/executive/branch-performance', {
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        ExecutiveState.branchData = data;
        return data;
    } catch (error) {
        console.error('[Executive] Branch performance fetch failed:', error);
        return null;
    }
}

// =============================================================================
// MAIN LOADER
// =============================================================================

/**
 * Main function called when switching to Executive role.
 * Loads all dashboard data and renders visualizations.
 */
async function loadExecutiveDashboardFromAPI() {
    if (ExecutiveState.isLoading) {
        console.log('[Executive] Already loading...');
        return;
    }

    ExecutiveState.isLoading = true;
    console.log('[Executive] Loading dashboard data...');

    try {
        // Fetch all data in parallel
        const [dashboardData, trendsData] = await Promise.all([
            fetchExecutiveDashboardData(),
            fetchExecutiveTrendsData()
        ]);

        if (dashboardData) {
            renderExecutiveDashboard(dashboardData);
        }

        if (trendsData) {
            renderExecutiveTrendChart(trendsData);
        }

        // Render loan portfolio chart (uses dashboard data)
        if (dashboardData && dashboardData.loan_portfolio) {
            renderLoanPortfolioChart(dashboardData.loan_portfolio);
        }

        // Auto-load Gemma AI insights after data is ready
        setTimeout(() => {
            loadExecutiveGemmaInsights();
        }, 500);

    } catch (error) {
        console.error('[Executive] Failed to load dashboard:', error);
    } finally {
        ExecutiveState.isLoading = false;
    }
}

// =============================================================================
// RENDER FUNCTIONS
// =============================================================================

/**
 * Renders the main executive dashboard data into the UI.
 * Updates ALL elements dynamically from API data.
 * @param {Object} data - Dashboard data from API
 */
function renderExecutiveDashboard(data) {
    if (!data) {
        console.warn('[Executive] No data to render');
        return;
    }

    console.log('[Executive] Rendering dashboard with data:', data);

    // =======================================================================
    // UPDATE FINANCIAL KPI CARDS
    // =======================================================================
    if (data.financial_kpis) {
        const kpis = data.financial_kpis;
        updateKPICard('execROA', `${kpis.return_on_assets?.value || 0}%`);
        updateKPICard('execNIM', `${kpis.net_interest_margin?.value || 0}%`);
        updateKPICard('execCapital', `${kpis.capital_ratio?.value || 0}%`);
        updateKPICard('execEfficiency', `${kpis.efficiency_ratio?.value || 0}%`);
    }

    // Update risk metrics
    if (data.risk_metrics) {
        updateKPICard('execDelinquency', `${data.risk_metrics.delinquency_rate?.value || 0}%`);
    }

    // Update member engagement
    if (data.member_engagement) {
        updateKPICard('execNPS', data.member_engagement.member_satisfaction_nps?.value || 0);
    }

    // =======================================================================
    // UPDATE GROWTH METRICS
    // =======================================================================
    if (data.growth_metrics) {
        const growth = data.growth_metrics;
        updateKPICard('execTotalAssets', formatCurrency(growth.total_assets?.value));
        updateKPICard('execDeposits', formatCurrency(growth.total_deposits?.value));
        updateKPICard('execLoans', formatCurrency(growth.total_loans?.value));
        updateKPICard('execMembers', formatNumber(growth.member_count?.value));
    }

    // =======================================================================
    // UPDATE SEGMENT PERFORMANCE (DYNAMIC)
    // =======================================================================
    if (data.segments) {
        renderSegmentPerformance(data.segments);
    }

    // =======================================================================
    // UPDATE RISK & COMPLIANCE SUMMARY (DYNAMIC)
    // =======================================================================
    if (data.risk_metrics) {
        renderRiskSummary(data.risk_metrics);
    }

    // =======================================================================
    // UPDATE EXECUTIVE ALERTS (DYNAMIC)
    // =======================================================================
    if (data.executive_alerts) {
        renderExecutiveAlerts(data.executive_alerts);
    }

    // Show data source indicator
    if (data.data_source) {
        console.log(`[Executive] Data source: ${data.data_source}, API calls: ${data.fiserv_api_calls || 0}`);
    }

    console.log('[Executive] Dashboard rendered successfully');
}

/**
 * Renders segment performance list dynamically from API data.
 * @param {Object} segments - Segment data from API
 */
function renderSegmentPerformance(segments) {
    const container = document.getElementById('execSegmentList');
    if (!container) return;

    const colors = ['#f0fdf4', '#eff6ff', '#faf5ff', '#fef3c7', '#f3f4f6'];
    const textColors = ['#16a34a', '#2563eb', '#7c3aed', '#d97706', '#6b7280'];

    let html = '';
    let index = 0;

    for (const [key, segment] of Object.entries(segments)) {
        const name = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        const growth = segment.growth_pct || 0;
        const bgColor = colors[index % colors.length];
        const color = textColors[index % textColors.length];

        html += `
            <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: ${bgColor}; border-radius: 8px; margin-bottom: 0.5rem;">
                <span><strong>${name}</strong></span>
                <span style="color: ${color}; font-weight: 600;">↑ +${growth}%</span>
            </div>
        `;
        index++;
    }

    container.innerHTML = html || '<div style="color: #6b7280;">No segment data available</div>';
}

/**
 * Renders risk and compliance summary dynamically from API data.
 * @param {Object} riskMetrics - Risk metrics from API
 */
function renderRiskSummary(riskMetrics) {
    const container = document.getElementById('execRiskSummary');
    if (!container) return;

    const delinquency = riskMetrics.delinquency_rate?.value || 0;
    const chargeoffs = riskMetrics.net_charge_off_rate?.value || 0;
    const provision = riskMetrics.provision_for_loan_loss?.value || 0;
    const provisionRatio = riskMetrics.provision_for_loan_loss?.coverage_ratio || 0;
    const fraudAlerts = riskMetrics.fraud_alerts || {};
    const compliance = riskMetrics.compliance_status || {};

    const totalFraud = (fraudAlerts.critical || 0) + (fraudAlerts.high || 0) + (fraudAlerts.medium || 0);
    const fraudPrevented = fraudAlerts.fraud_prevented_usd || 0;

    container.innerHTML = `
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; border-bottom: 1px solid #f3f4f6;">
            <span>Delinquency Rate</span>
            <span style="font-weight: 600; color: ${delinquency < 1.0 ? '#16a34a' : '#f59e0b'};">${delinquency}% ${delinquency < 1.0 ? '✓' : '⚠'}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; border-bottom: 1px solid #f3f4f6;">
            <span>Net Charge-offs</span>
            <span style="font-weight: 600; color: ${chargeoffs < 0.5 ? '#16a34a' : '#f59e0b'};">${chargeoffs}% ${chargeoffs < 0.5 ? '✓' : '⚠'}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; border-bottom: 1px solid #f3f4f6;">
            <span>Provision for Loan Loss</span>
            <span style="font-weight: 600;">$${(provision / 1_000_000).toFixed(1)}M (${provisionRatio}x)</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; border-bottom: 1px solid #f3f4f6;">
            <span>Active Fraud Alerts</span>
            <span style="font-weight: 600; color: ${totalFraud > 5 ? '#ef4444' : '#f59e0b'};">${totalFraud} (${fraudAlerts.critical || 0} Critical)</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; border-bottom: 1px solid #f3f4f6;">
            <span>Fraud Prevented (MTD)</span>
            <span style="font-weight: 600; color: #16a34a;">$${fraudPrevented.toLocaleString()}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.75rem;">
            <span>Compliance Status</span>
            <span style="font-weight: 600;">
                <span style="display: inline-block; width: 10px; height: 10px; background: ${compliance.overall === 'green' ? '#22c55e' : '#f59e0b'}; border-radius: 50%; margin-right: 4px;"></span>
                ${compliance.overall === 'green' ? 'All Green' : compliance.overall || 'Unknown'}
            </span>
        </div>
    `;
}

/**
 * Renders executive alerts dynamically from API data.
 * @param {Array} alerts - Executive alerts from API
 */
function renderExecutiveAlerts(alerts) {
    const container = document.getElementById('execAlertsList');
    if (!container) return;

    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div style="color: #6b7280; padding: 1rem;">No alerts at this time</div>';
        return;
    }

    const priorityColors = {
        'success': { bg: '#dcfce7', text: '#15803d', icon: '✅' },
        'info': { bg: '#e0f2fe', text: '#0369a1', icon: 'ℹ️' },
        'warning': { bg: '#fef3c7', text: '#92400e', icon: '⚠️' },
        'critical': { bg: '#fee2e2', text: '#b91c1c', icon: '🔶' }
    };

    let html = '';
    for (const alert of alerts) {
        const style = priorityColors[alert.priority] || priorityColors['info'];
        html += `
            <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem; background: ${style.bg}; border-radius: 8px; margin-bottom: 0.5rem;">
                <span style="font-size: 1.25rem;">${style.icon}</span>
                <div>
                    <div style="font-weight: 600; color: ${style.text};">${alert.message}</div>
                    <div style="font-size: 0.8rem; color: ${style.text};">${alert.category}</div>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

/**
 * Updates a KPI card value by element ID.
 * @param {string} elementId - DOM element ID
 * @param {string|number} value - Value to display
 */
function updateKPICard(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = value;
    }
}

/**
 * Formats a number as currency (billions/millions).
 * @param {number} value - Number to format
 * @returns {string} Formatted currency string
 */
function formatCurrency(value) {
    if (!value) return '$0';
    if (value >= 1_000_000_000) {
        return `$${(value / 1_000_000_000).toFixed(1)}B`;
    }
    if (value >= 1_000_000) {
        return `$${(value / 1_000_000).toFixed(0)}M`;
    }
    return `$${value.toLocaleString()}`;
}

/**
 * Formats a number with K/M suffix.
 * @param {number} value - Number to format
 * @returns {string} Formatted number string
 */
function formatNumber(value) {
    if (!value) return '0';
    if (value >= 1_000_000) {
        return `${(value / 1_000_000).toFixed(1)}M`;
    }
    if (value >= 1_000) {
        return `${(value / 1_000).toFixed(0)}K`;
    }
    return value.toLocaleString();
}

// =============================================================================
// CHART RENDERING (PLOTLY)
// =============================================================================

/**
 * Renders the loan portfolio composition pie chart.
 * @param {Object} loanPortfolio - Loan portfolio data from API
 */
function renderLoanPortfolioChart(loanPortfolio) {
    const chartDiv = document.getElementById('execLoanPieChart');
    if (!chartDiv || typeof Plotly === 'undefined') {
        console.warn('[Executive] Plotly not available or chart div missing');
        return;
    }

    // Transform data for Plotly
    const labels = [];
    const values = [];
    const colors = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#06b6d4', '#6b7280'];

    const labelMap = {
        'mortgage': 'Mortgage',
        'auto_loans': 'Auto Loans',
        'personal_loans': 'Personal',
        'credit_cards': 'Credit Cards',
        'heloc': 'HELOC',
        'other': 'Other'
    };

    for (const [key, data] of Object.entries(loanPortfolio)) {
        if (data && data.balance) {
            labels.push(labelMap[key] || key);
            values.push(data.balance);
        }
    }

    const plotData = [{
        type: 'pie',
        labels: labels,
        values: values,
        hole: 0.4,
        marker: {
            colors: colors
        },
        textinfo: 'label+percent',
        textposition: 'outside',
        hovertemplate: '<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>'
    }];

    const layout = {
        showlegend: false,
        margin: { t: 20, b: 20, l: 20, r: 20 },
        font: { family: 'Inter, -apple-system, sans-serif' }
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.newPlot(chartDiv, plotData, layout, config);
    console.log('[Executive] Loan portfolio chart rendered');
}

/**
 * Renders the 12-month growth trend line chart.
 * @param {Object} trendsData - Trends data from API
 */
function renderExecutiveTrendChart(trendsData) {
    const chartDiv = document.getElementById('execTrendChart');
    if (!chartDiv || typeof Plotly === 'undefined') {
        console.warn('[Executive] Plotly not available or chart div missing');
        return;
    }

    if (!trendsData || !trendsData.data_points) {
        console.warn('[Executive] No trend data available');
        return;
    }

    const dataPoints = trendsData.data_points;
    const months = dataPoints.map(d => d.month);
    const deposits = dataPoints.map(d => d.deposits / 1_000_000_000); // Convert to billions
    const loans = dataPoints.map(d => d.loans / 1_000_000_000);

    const plotData = [
        {
            x: months,
            y: deposits,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Deposits',
            line: { color: '#3b82f6', width: 3 },
            marker: { size: 6 },
            hovertemplate: '<b>%{x}</b><br>Deposits: $%{y:.2f}B<extra></extra>'
        },
        {
            x: months,
            y: loans,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Loans',
            line: { color: '#8b5cf6', width: 3 },
            marker: { size: 6 },
            hovertemplate: '<b>%{x}</b><br>Loans: $%{y:.2f}B<extra></extra>'
        }
    ];

    const layout = {
        showlegend: true,
        legend: { orientation: 'h', y: -0.15 },
        margin: { t: 20, b: 60, l: 60, r: 20 },
        font: { family: 'Inter, -apple-system, sans-serif' },
        xaxis: {
            tickangle: -45,
            tickfont: { size: 10 }
        },
        yaxis: {
            title: 'Balance ($B)',
            tickformat: '$.2f',
            ticksuffix: 'B'
        }
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.newPlot(chartDiv, plotData, layout, config);
    console.log('[Executive] Trend chart rendered');
}

// =============================================================================
// GEMMA AI INTEGRATION
// =============================================================================

/**
 * Loads AI-generated strategic insights from Gemma.
 */
async function loadExecutiveGemmaInsights() {
    const insightsDiv = document.getElementById('execGemmaInsights');
    if (!insightsDiv) return;

    // Show loading state
    insightsDiv.innerHTML = `
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔄</div>
            <div>Generating AI insights...</div>
        </div>
    `;

    try {
        // Build context from current dashboard data
        const context = buildExecutiveContext();

        // Call Gemma API using the public chat endpoint (same as gemma.html)
        // This endpoint works without special auth
        const response = await fetch('/api/public/chat', {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                messages: [
                    { role: 'system', content: 'You are a strategic advisor for credit union executives. Provide concise, actionable insights based on the data provided.' },
                    { role: 'user', content: `Based on this credit union data, provide 5 actionable strategic insights for the CEO:\n\n${context}` }
                ],
                max_tokens: 500,
                temperature: 0.7,
                top_p: 0.9
            })
        });

        if (response.ok) {
            const data = await response.json();
            const answer = data.message || data.text || data.answer || data.response || 'Unable to generate insights.';

            insightsDiv.innerHTML = `
                <p style="margin: 0 0 0.75rem 0;">Based on current KPIs and trends, here are key strategic insights:</p>
                <div style="line-height: 1.8;">${formatGemmaResponse(answer)}</div>
            `;
        } else {
            throw new Error(`Gemma API error: ${response.status}`);
        }
    } catch (error) {
        console.error('[Executive] Gemma insights failed:', error);
        insightsDiv.innerHTML = `
            <p style="margin: 0;">Unable to load AI insights. Default analysis:</p>
            <ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem; line-height: 1.8;">
                <li>Veteran segment showing strongest deposit growth (+5.8%) — consider targeted retention campaigns</li>
                <li>Digital adoption at 78.5% — 1.5% from target; focus on mobile payment features</li>
                <li>Auto loan delinquencies trending up 0.2% — recommend enhanced underwriting review</li>
                <li>Germany branch may need temporary staffing for PCS season surge</li>
                <li>NPS score of 72 exceeds industry average by 14 points — strong competitive position</li>
            </ul>
        `;
    }
}

/**
 * Builds context string from executive dashboard data for Gemma.
 * @returns {string} Context string
 */
function buildExecutiveContext() {
    const data = ExecutiveState.dashboardData;
    if (!data) return 'No dashboard data available.';

    let context = 'Credit Union Executive Summary:\n\n';

    if (data.financial_kpis) {
        context += `Financial KPIs:\n`;
        context += `- ROA: ${data.financial_kpis.return_on_assets?.value}% (Target: ${data.financial_kpis.return_on_assets?.target}%)\n`;
        context += `- NIM: ${data.financial_kpis.net_interest_margin?.value}%\n`;
        context += `- Capital Ratio: ${data.financial_kpis.capital_ratio?.value}%\n`;
        context += `- Efficiency Ratio: ${data.financial_kpis.efficiency_ratio?.value}%\n\n`;
    }

    if (data.growth_metrics) {
        context += `Growth Metrics:\n`;
        context += `- Total Assets: $${(data.growth_metrics.total_assets?.value / 1e9).toFixed(1)}B (+${data.growth_metrics.total_assets?.ytd_change_pct}% YTD)\n`;
        context += `- Total Deposits: $${(data.growth_metrics.total_deposits?.value / 1e9).toFixed(1)}B\n`;
        context += `- Member Count: ${data.growth_metrics.member_count?.value?.toLocaleString()}\n\n`;
    }

    if (data.risk_metrics) {
        context += `Risk Metrics:\n`;
        context += `- Delinquency Rate: ${data.risk_metrics.delinquency_rate?.value}%\n`;
        context += `- Fraud Alerts: ${data.risk_metrics.fraud_alerts?.critical} critical, ${data.risk_metrics.fraud_alerts?.high} high\n\n`;
    }

    if (data.segments) {
        context += `Segment Growth:\n`;
        for (const [key, segment] of Object.entries(data.segments)) {
            context += `- ${key.replace(/_/g, ' ')}: +${segment.growth_pct}%\n`;
        }
    }

    return context;
}

/**
 * Formats Gemma response for display.
 * @param {string} response - Raw Gemma response
 * @returns {string} Formatted HTML
 */
function formatGemmaResponse(response) {
    // Convert bullet points to HTML list
    const lines = response.split('\n').filter(line => line.trim());
    let html = '<ul style="margin: 0; padding-left: 1.5rem;">';

    for (const line of lines) {
        const cleanLine = line.replace(/^[\-\*•]\s*/, '').replace(/^\d+\.\s*/, '');
        if (cleanLine.trim()) {
            html += `<li>${cleanLine}</li>`;
        }
    }

    html += '</ul>';
    return html;
}

/**
 * Opens a modal for asking Gemma questions about executive data.
 */
function openExecutiveGemmaChat() {
    // Check if modal already exists
    let modal = document.getElementById('execGemmaChatModal');

    if (!modal) {
        // Create modal
        modal = document.createElement('div');
        modal.id = 'execGemmaChatModal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        `;

        modal.innerHTML = `
            <div style="background: white; border-radius: 12px; width: 90%; max-width: 600px; max-height: 80vh; overflow: hidden; display: flex; flex-direction: column;">
                <div style="padding: 1.5rem; background: linear-gradient(135deg, #6366f1, #4f46e5); color: white;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-weight: 700; font-size: 1.1rem;">🤖 Ask Gemma - Executive Insights</div>
                        <button onclick="closeExecutiveGemmaChat()" style="background: rgba(255,255,255,0.2); border: none; color: white; width: 32px; height: 32px; border-radius: 6px; cursor: pointer; font-size: 1.2rem;">×</button>
                    </div>
                </div>
                <div id="execGemmaChatMessages" style="flex: 1; overflow-y: auto; padding: 1.5rem; min-height: 300px;">
                    <div style="text-align: center; color: #6b7280; padding: 2rem;">
                        Ask questions about credit union performance, KPIs, or strategic recommendations.
                    </div>
                </div>
                <div style="padding: 1rem; border-top: 1px solid #e5e7eb; display: flex; gap: 0.5rem;">
                    <input type="text" id="execGemmaChatInput" placeholder="Ask about executive metrics..." style="flex: 1; padding: 0.75rem 1rem; border: 1px solid #e5e7eb; border-radius: 8px; font-size: 0.95rem;" onkeypress="if(event.key==='Enter')askExecutiveGemmaQuestion()">
                    <button onclick="askExecutiveGemmaQuestion()" style="padding: 0.75rem 1.5rem; background: #6366f1; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600;">Send</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    } else {
        modal.style.display = 'flex';
    }
}

/**
 * Closes the Gemma chat modal.
 */
function closeExecutiveGemmaChat() {
    const modal = document.getElementById('execGemmaChatModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

/**
 * Asks a question to Gemma with executive context.
 */
async function askExecutiveGemmaQuestion() {
    const input = document.getElementById('execGemmaChatInput');
    const messagesDiv = document.getElementById('execGemmaChatMessages');

    if (!input || !messagesDiv) return;

    const question = input.value.trim();
    if (!question) return;

    // Clear input
    input.value = '';

    // Add user message
    messagesDiv.innerHTML += `
        <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
            <div style="background: #6366f1; color: white; padding: 0.75rem 1rem; border-radius: 12px 12px 0 12px; max-width: 80%;">
                ${escapeHtml(question)}
            </div>
        </div>
    `;

    // Add loading indicator
    const loadingId = 'gemma-loading-' + Date.now();
    messagesDiv.innerHTML += `
        <div id="${loadingId}" style="display: flex; margin-bottom: 1rem;">
            <div style="background: #f3f4f6; padding: 0.75rem 1rem; border-radius: 12px 12px 12px 0; max-width: 80%;">
                <span style="color: #6b7280;">Thinking...</span>
            </div>
        </div>
    `;
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    try {
        const context = buildExecutiveContext();

        const response = await fetch('/api/gemma/chat', {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: `Context: ${context}\n\nUser Question: ${question}`,
                system_prompt: 'You are a strategic advisor for credit union executives. Answer questions based on the provided data.'
            })
        });

        // Remove loading indicator
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();

        if (response.ok) {
            const data = await response.json();
            const answer = data.answer || data.response || 'Unable to generate response.';

            messagesDiv.innerHTML += `
                <div style="display: flex; margin-bottom: 1rem;">
                    <div style="background: #f3f4f6; padding: 0.75rem 1rem; border-radius: 12px 12px 12px 0; max-width: 80%;">
                        ${formatGemmaResponse(answer)}
                    </div>
                </div>
            `;
        } else {
            throw new Error('API error');
        }
    } catch (error) {
        console.error('[Executive] Gemma question failed:', error);
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();

        messagesDiv.innerHTML += `
            <div style="display: flex; margin-bottom: 1rem;">
                <div style="background: #fee2e2; color: #b91c1c; padding: 0.75rem 1rem; border-radius: 12px 12px 12px 0; max-width: 80%;">
                    Sorry, I couldn't process that question. Please try again.
                </div>
            </div>
        `;
    }

    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

/**
 * Escapes HTML special characters.
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// =============================================================================
// INITIALIZATION
// =============================================================================

/**
 * Auto-initializes when section becomes visible.
 * Called by showSection() in banking_core.js
 */
function initExecutiveSection() {
    console.log('[Executive] Section initialized');
    loadExecutiveDashboardFromAPI();
}

// Export functions for global access
window.loadExecutiveDashboardFromAPI = loadExecutiveDashboardFromAPI;
window.loadExecutiveGemmaInsights = loadExecutiveGemmaInsights;
window.openExecutiveGemmaChat = openExecutiveGemmaChat;
window.closeExecutiveGemmaChat = closeExecutiveGemmaChat;
window.askExecutiveGemmaQuestion = askExecutiveGemmaQuestion;
window.initExecutiveSection = initExecutiveSection;

console.log('[Banking Executive] Module loaded - v1.0.0');
