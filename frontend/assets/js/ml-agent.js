// ml-agent.js

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const tasksContainer = document.getElementById('tasks-container');
    const analysisContainer = document.getElementById('analysis-container');

    let currentAnalysisContext = null;

    function resetUI() {
        dropZone.classList.remove('hidden');
        
        // Reset internal states
        const content = document.getElementById('drop-zone-content');
        const loading = document.getElementById('drop-zone-loading');
        if (content) content.classList.remove('hidden');
        if (loading) loading.classList.add('hidden');

        uploadStatus.innerHTML = '';
        tasksContainer.innerHTML = '';
        analysisContainer.innerHTML = '';
        tasksContainer.classList.add('hidden'); // Hide tasks container initially
    }

    // Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    const API_BASE_URL = `http://${window.location.hostname}:8006`;

    async function handleFileUpload(file) {
        resetUI(); // Clear previous analysis
        
        // Show loading state
        const content = document.getElementById('drop-zone-content');
        const loading = document.getElementById('drop-zone-loading');
        if (content) content.classList.add('hidden');
        if (loading) loading.classList.remove('hidden');
        
        lucide.createIcons();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/ingest`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');

            const profile = await response.json();
            uploadStatus.innerHTML = `✅ Uploaded <strong>${profile.filename}</strong> (${profile.row_count} rows).`;

            initiateConversation(profile);
            dropZone.classList.add('hidden'); // Hide the upload interface
            uploadStatus.innerHTML = ''; // Clear status message
            tasksContainer.classList.remove('hidden'); // Ensure tasks container is visible if it was hidden
        } catch (error) {
            console.error(error);
            uploadStatus.innerHTML = `❌ Error: ${error.message}`;
            
            // Revert loading state on error
            if (content) content.classList.remove('hidden');
            if (loading) loading.classList.add('hidden');
        }
    }

    window.initiateConversation = async function (profile) {
        resetUI(); // Clear previous analysis
        tasksContainer.innerHTML = `
            <div class="glass-card fade-in-up" style="border-left: 4px solid var(--primary-color);">
                <div class="flex" style="align-items: flex-start; gap: 1rem;">
                    <div style="background: var(--primary-color); padding: 0.5rem; border-radius: 50%;">
                        <i data-lucide="bot" style="color: white;"></i>
                    </div>
                    <div>
                        <h4>Gemma Analysis</h4>
                        <p>I've analyzed the dataset. I see columns like <strong>${profile.columns.slice(0, 3).join(', ')}</strong>.</p>
                        <p>Based on this, here are high-impact analyses I can perform:</p>
                    </div>
                </div>
                <div id="goals-list" style="margin-top: 1.5rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                    <div class="text-center"><i data-lucide="loader-2" class="spin"></i> Thinking...</div>
                </div>
            </div>
        `;
        lucide.createIcons();

        const response = await fetch('http://localhost:8006/propose-goals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile)
        });
        const goals = await response.json();
        renderGoals(goals, profile);
    }

    function renderGoals(goals, profile) {
        const list = document.getElementById('goals-list');
        list.innerHTML = '';

        goals.forEach(goal => {
            const card = document.createElement('div');
            card.className = 'glass-card goal-card';
            card.style.cursor = 'pointer';
            card.style.border = '1px solid rgba(255,255,255,0.1)';
            card.style.transition = 'transform 0.2s';

            card.onmouseover = () => card.style.transform = 'translateY(-2px)';
            card.onmouseout = () => card.style.transform = 'translateY(0)';

            card.innerHTML = `
                <div class="flex-between" style="margin-bottom: 0.5rem;">
                    <h5 style="margin: 0;">${goal.title}</h5>
                    <i data-lucide="arrow-right" style="width: 16px;"></i>
                </div>
                <p class="text-muted" style="font-size: 0.85rem; margin-bottom: 0.5rem;">${goal.description}</p>
                <span class="badge badge-secondary" style="font-size: 0.7rem;">${goal.type}</span>
            `;

            card.addEventListener('click', () => executeGoal(goal, profile));
            list.appendChild(card);
        });
        lucide.createIcons();
    }

    async function executeGoal(goal, profile) {
        analysisContainer.innerHTML = `
            <div class="glass-card fade-in-up">
                <div class="flex" style="gap: 1rem; align-items: center; margin-bottom: 1rem;">
                    <i data-lucide="terminal" style="color: var(--primary-color);"></i>
                    <h4 style="margin: 0;">Executing: ${goal.title}</h4>
                </div>
                <div class="terminal-window" style="background: #1e1e1e; padding: 1rem; border-radius: 6px; font-family: monospace; font-size: 0.9rem; color: #a9b7c6; height: 150px; overflow-y: auto;">
                    <div class="line">> Initializing Python environment...</div>
                    <div class="line">> Loading pandas and scikit-learn...</div>
                    <div class="line">> Reading dataset ${profile.filename}...</div>
                    <div class="line">> Running ${goal.type} algorithms...</div>
                    <div class="line typing-cursor">_</div>
                </div>
            </div>
        `;
        lucide.createIcons();
        analysisContainer.scrollIntoView({ behavior: 'smooth' });

        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: profile.filename, goal_id: goal.id })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Analysis failed');
            }

            const result = await response.json();
            currentAnalysisContext = result;
            renderResult(result);
        } catch (e) {
            analysisContainer.innerHTML += `<p class="text-danger">Error: ${e.message}</p>`;
        }
    }

    function renderResult(result) {
        const terminal = document.querySelector('.terminal-window');
        if (terminal) {
            terminal.innerHTML += `<div class="line" style="color: #4caf50;">> Analysis Complete.</div>`;
            terminal.scrollTop = terminal.scrollHeight;
        }

        // Safe check for insights
        const insights = result.insights || ["No specific insights generated."];

        setTimeout(() => {
            const narration = result.gemma_narration || result.summary;

            const resultHtml = `
                <div class="glass-card fade-in-up" style="margin-top: 1rem; border-top: 4px solid #4caf50;">
                    <div class="flex" style="gap: 0.5rem; align-items: center; margin-bottom: 1rem;">
                        <i data-lucide="bot" style="color: var(--primary-color);"></i>
                        <h3 style="margin: 0;">Gemma AI Presentation</h3>
                    </div>
                    <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem; border-left: 3px solid var(--primary-color);">
                        <p style="font-size: 1.05rem; line-height: 1.6; margin: 0;">${narration}</p>
                    </div>

                    <h4><i data-lucide="sparkles" style="color: gold;"></i> Key Insights</h4>
                    <ul style="list-style: none; padding: 0; margin-bottom: 2rem;">
                        ${insights.map(i => `
                            <li style="margin-bottom: 0.8rem; padding-left: 1.5rem; position: relative;">
                                <i data-lucide="check-circle" style="width: 16px; position: absolute; left: 0; top: 4px; color: #4caf50;"></i>
                                ${i}
                            </li>
                        `).join('')}
                    </ul>

                    ${renderCharts(result.charts || [result.chart_data].filter(Boolean))}
                </div>
            `;

            const resultDiv = document.createElement('div');
            resultDiv.innerHTML = resultHtml;
            analysisContainer.appendChild(resultDiv);

            // Add New Analysis button
            const newAnalysisButton = document.createElement('div');
            newAnalysisButton.innerHTML = `
                <div class="flex-center" style="margin-top: 2rem;">
                    <button class="btn btn-secondary" onclick="resetUI()">
                        <i data-lucide="plus-circle" style="width: 16px; height: 16px;"></i>
                        New Analysis
                    </button>
                </div>
            `;
            analysisContainer.appendChild(newAnalysisButton);

            lucide.createIcons();

            renderAllCharts(result.charts || [result.chart_data].filter(Boolean));
        }, 1000);
    }

    function renderCharts(charts) {
        if (!charts || charts.length === 0) return '';

        return `
            <h4><i data-lucide="bar-chart-3"></i> Visualizations</h4>
            <div class="charts-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 2rem; margin-top: 1.5rem;">
                ${charts.map((chart, idx) => `
                    <div class="chart-container glass-card" style="padding: 1.5rem;">
                        <h5 style="margin: 0 0 1rem 0;">${chart.title || chart.label || `Chart ${idx + 1}`}</h5>
                        <div style="height: 300px; position: relative;">
                            <canvas id="chart_${idx}" data-chart-index="${idx}"></canvas>
                        </div>
                        ${chart.insight_ref ? `
                            <p class="text-muted" style="font-size: 0.85rem; margin-top: 0.5rem;">
                                <i data-lucide="info"></i> Related to insights: ${chart.insight_ref.map(i => i + 1).join(', ')}
                            </p>
                        ` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }

    function renderAllCharts(charts) {
        if (!charts || charts.length === 0) return;

        charts.forEach((chartData, idx) => {
            const canvas = document.getElementById(`chart_${idx}`);
            if (!canvas) return;

            const ctx = canvas.getContext('2d');

            const colorSchemes = {
                success: ['rgba(75, 192, 192, 0.8)', 'rgba(54, 162, 235, 0.8)', 'rgba(153, 102, 255, 0.8)'],
                warning: ['rgba(255, 206, 86, 0.8)', 'rgba(255, 159, 64, 0.8)', 'rgba(255, 99, 132, 0.8)'],
                primary: ['rgba(54, 162, 235, 0.8)', 'rgba(153, 102, 255, 0.8)', 'rgba(255, 159, 64, 0.8)'],
                mixed: ['rgba(75, 192, 192, 0.8)', 'rgba(255, 206, 86, 0.8)', 'rgba(255, 99, 132, 0.8)']
            };

            const scheme = chartData.color_scheme || 'primary';
            const colors = colorSchemes[scheme] || colorSchemes.primary;

            new Chart(ctx, {
                type: chartData.type,
                data: {
                    labels: chartData.labels,
                    datasets: [{
                        label: chartData.label || chartData.title || 'Data',
                        data: chartData.data,
                        backgroundColor: colors,
                        borderColor: colors.map(c => c.replace('0.8', '1')),
                        borderWidth: 2,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: '#fff', font: { size: 11 } }
                        },
                        title: {
                            display: false
                        }
                    },
                    scales: chartData.type !== 'doughnut' && chartData.type !== 'pie' ? {
                        y: {
                            ticks: { color: '#ccc' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        x: {
                            ticks: { color: '#ccc', maxRotation: 45, minRotation: 0 },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        }
                    } : {}
                }
            });
        });

        lucide.createIcons();
    }
});

// DataSourceManager Implementation
const DataSourceManager = {
    openModal: function() {
        const modal = document.getElementById('source-modal');
        if (modal) {
            modal.classList.remove('hidden');
            this.renderSourceSelector();
            
            // Add click listener to close on background click
            modal.onclick = (e) => {
                if (e.target === modal) {
                    this.closeModal();
                }
            };
        }
    },

    closeModal: function() {
        const modal = document.getElementById('source-modal');
        if (modal) {
            modal.classList.add('hidden');
            this.resetSelection();
        }
    },

    renderSourceSelector: function() {
        const selector = document.getElementById('source-selector');
        const form = document.getElementById('source-form');
        
        if (selector && form) {
            selector.classList.remove('hidden');
            form.classList.add('hidden');
            
            const sources = [
                { id: 'local', name: 'Local File', icon: 'upload-cloud' },
                { id: 'postgres', name: 'PostgreSQL', icon: 'database' },
                { id: 'mysql', name: 'MySQL', icon: 'database' },
                { id: 'mongodb', name: 'MongoDB', icon: 'server' },
                { id: 'redis', name: 'Redis', icon: 'layers' },
                { id: 'bigquery', name: 'BigQuery', icon: 'cloud' },
                { id: 'snowflake', name: 'Snowflake', icon: 'cloud-snow' },
                { id: 's3', name: 'Amazon S3', icon: 'hard-drive' },
                { id: 'rest', name: 'REST API', icon: 'globe' }
            ];

            selector.innerHTML = sources.map(src => `
                <div onclick="DataSourceManager.selectSource('${src.id}')" 
                     style="cursor: pointer; padding: 1rem; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; text-align: center; transition: all 0.2s;">
                    <i data-lucide="${src.icon}" style="margin-bottom: 0.5rem;"></i>
                    <div style="font-size: 0.85rem;">${src.name}</div>
                </div>
            `).join('');
            
            lucide.createIcons();
        }
    },

    selectSource: function(sourceId) {
        if (sourceId === 'local') {
            document.getElementById('file-input').click();
            this.closeModal();
            return;
        }

        const selector = document.getElementById('source-selector');
        const form = document.getElementById('source-form');
        const fields = document.getElementById('source-form-fields');
        const title = document.getElementById('source-form-title');

        if (selector && form && fields && title) {
            selector.classList.add('hidden');
            form.classList.remove('hidden');
            title.textContent = `Configure ${sourceId.toUpperCase()}`;
            
            // Generate fields based on source type (Simplified for now)
            let inputs = '';
            if (['postgres', 'mysql'].includes(sourceId)) {
                inputs = `
                    <input type="text" class="input" placeholder="Host (e.g., localhost)" name="host" required>
                    <input type="number" class="input" placeholder="Port (e.g., 5432)" name="port" required>
                    <input type="text" class="input" placeholder="Database Name" name="database" required>
                    <input type="text" class="input" placeholder="Username" name="user" required>
                    <input type="password" class="input" placeholder="Password" name="password" required>
                    <textarea class="input" placeholder="SQL Query (Optional)" name="query"></textarea>
                `;
            } else if (sourceId === 'rest') {
                inputs = `
                    <input type="url" class="input" placeholder="API Endpoint URL" name="endpoint" required>
                    <input type="text" class="input" placeholder="Data Key (e.g., data.items)" name="data_key">
                `;
            } else {
                inputs = `<p class="text-muted">Configuration for ${sourceId} is coming soon.</p>`;
            }
            
            fields.innerHTML = inputs;
            
            // Handle form submission
            form.onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData(form);
                const data = Object.fromEntries(formData.entries());
                
                // Construct payload for /ingest/source
                const payload = {
                    source_type: sourceId,
                    connection_params: { ...data }, // naive spread, refinement needed based on specific loaders
                    query_params: { query: data.query }
                };
                
                // Clean up payload
                delete payload.connection_params.query;
                
                try {
                    const btn = form.querySelector('button[type="submit"]');
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Connecting...';
                    
                    // Use the globally defined API_BASE_URL, falling back to dynamic construction if undefined (for safety)
                    const baseUrl = (typeof API_BASE_URL !== 'undefined') ? API_BASE_URL : `http://${window.location.hostname}:8006`;
                    
                    const response = await fetch(`${baseUrl}/ingest/source`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    
                    if (!response.ok) throw new Error('Connection failed');
                    
                    const profile = await response.json();
                    this.closeModal();
                    
                    // Trigger main flow
                    const uploadStatus = document.getElementById('upload-status');
                    if (uploadStatus) {
                        uploadStatus.innerHTML = `✅ Connected to <strong>${sourceId}</strong> (${profile.row_count} rows).`;
                    }
                    window.initiateConversation(profile);
                    
                    const dropZone = document.getElementById('drop-zone');
                    if (dropZone) dropZone.classList.add('hidden');
                    if (uploadStatus) uploadStatus.innerHTML = '';
                    const tasksContainer = document.getElementById('tasks-container');
                    if (tasksContainer) tasksContainer.classList.remove('hidden');
                    
                } catch (err) {
                    alert(`Error: ${err.message}`);
                } finally {
                    const btn = form.querySelector('button[type="submit"]');
                    if (btn) btn.innerHTML = '<i data-lucide="plug"></i> Connect';
                    lucide.createIcons();
                }
            };
        }
    },

    resetSelection: function() {
        this.renderSourceSelector();
    }
};