/**
 * Predictions Page - File Upload Handler
 */

import { PredictionsState, DOM, Logger, API_BASE } from './predictions-state.js';

const Upload = {
  init() {
    if (!DOM.uploadArea || !DOM.fileInput) return;
    
    DOM.uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      DOM.uploadArea.classList.add('dragover');
    });
    
    DOM.uploadArea.addEventListener('dragleave', () => {
      DOM.uploadArea.classList.remove('dragover');
    });
    
    DOM.uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      DOM.uploadArea.classList.remove('dragover');
      if (e.dataTransfer.files[0]) {
        this.handleFile(e.dataTransfer.files[0]);
      }
    });
    
    DOM.fileInput.addEventListener('change', (e) => {
      if (e.target.files[0]) {
        this.handleFile(e.target.files[0]);
      }
    });
  },
  
  async handleFile(file) {
    Logger.setStatus(`Uploading ${file.name}...`, 'loading', 20);
    Logger.log(`Uploading: ${file.name}`);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });
      
      const data = await res.json();
      
      if (data.error || data.detail) {
        throw new Error(Logger.errorToString(data.error || data.detail));
      }
      
      // Update state
      PredictionsState.setUploadData(data.filename, data.columns || [], data.rows || 0);
      
      Logger.setStatus('✅ Upload successful! Ready to analyze.', 'success', 100);
      Logger.log(`File uploaded: ${data.filename} (${data.rows} rows, ${data.columns.length} cols)`, 'success');
      
      // Enable analysis buttons
      DOM.analyzeBtn.disabled = false;
      DOM.testAllBtn.disabled = false;
      
      // Update upload area display
      DOM.uploadArea.innerHTML = `
        <div class="upload-icon">📄</div>
        <div class="upload-text">${data.filename}</div>
        <div class="upload-hint">${data.columns?.length || '?'} columns • ${(data.rows || 0).toLocaleString()} rows</div>
      `;
      
      Logger.log('Upload complete: ' + data.filename, 'success');
      
      // Show column selection card
      const columnCard = document.getElementById('column-selection-section');
      if (columnCard) {
        columnCard.classList.add('visible');
        this.populateColumnSelectors();
      }
      
      // Detect time columns for forecasting
      this.detectTimeColumns();
      Logger.log('🔮 Time-series forecasting available after analysis', 'info');
      Logger.log('Column selection available - Choose Gemma AI or Manual mode', 'info');
      
    } catch (err) {
      Logger.setStatus(`Upload failed: ${err.message}`, 'error', 0);
      Logger.log(`Upload error: ${err.message}`, 'error');
    }
  },
  
  populateColumnSelectors() {
    const targetSelect = document.getElementById('target-select');
    const featureGrid = document.getElementById('feature-grid');
    
    if (!targetSelect || !featureGrid) return;
    
    // Populate target dropdown
    targetSelect.innerHTML = '<option value="">-- Select target column --</option>' +
      PredictionsState.uploadedColumns.map(col => 
        `<option value="${col}">${col}</option>`
      ).join('');
    
    // Populate feature checkboxes
    featureGrid.innerHTML = PredictionsState.uploadedColumns.map(col =>
      `<label class="feature-checkbox" data-column="${col}" onclick="window.ColumnSelection && window.ColumnSelection.toggleFeature(this)">
        <input type="checkbox" value="${col}">
        <span class="checkbox-custom"></span>
        <span class="feature-label" title="${col}">${col}</span>
      </label>`
    ).join('');
    
    this.updateFeatureCount();
  },
  
  updateFeatureCount() {
    const count = PredictionsState.manualFeatures.length;
    const label = document.getElementById('feature-count');
    if (label) {
      label.textContent = `${count} feature${count !== 1 ? 's' : ''} selected`;
    }
  },
  
  detectTimeColumns() {
    const timePatterns = [
      /date|time|timestamp|datetime|year|month|day|week|quarter/i,
      /^date_|_date$|^time_|_time$|^ts_|_ts$/i
    ];
    
    const detected = PredictionsState.uploadedColumns.some(col =>
      timePatterns.some(pattern => pattern.test(col))
    );
    
    PredictionsState.hasTimeColumn = detected;
    PredictionsState.detectedTimeColumn = detected ? 
      PredictionsState.uploadedColumns.find(col => timePatterns[0].test(col)) : null;
    
    return detected;
  }
};

export default Upload;
