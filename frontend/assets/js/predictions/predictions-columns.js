/**
 * Predictions Page - Column Selection Handler
 */

import { PredictionsState, Logger, API_BASE } from './predictions-state.js';

const ColumnSelection = {
  init() {
    window.ColumnSelection = this;
    const modeGemma = document.getElementById('mode-gemma');
    const modeManual = document.getElementById('mode-manual');
    
    if (modeGemma) modeGemma.addEventListener('click', () => this.setSelectionMode('gemma'));
    if (modeManual) modeManual.addEventListener('click', () => this.setSelectionMode('manual'));
  },
  
  setSelectionMode(mode) {
    PredictionsState.selectionMode = mode;
    
    document.getElementById('mode-gemma')?.classList.toggle('active', mode === 'gemma');
    document.getElementById('mode-manual')?.classList.toggle('active', mode === 'manual');
    document.getElementById('gemma-mode-panel').style.display = mode === 'gemma' ? 'block' : 'none';
    document.getElementById('manual-mode-panel').style.display = mode === 'manual' ? 'block' : 'none';
    
    Logger.log(`Selection mode: ${mode === 'gemma' ? '🤖 Gemma AI' : '✋ Manual'}`, 'info');
  },
  
  onTargetChange() {
    PredictionsState.manualTarget = document.getElementById('target-select')?.value;
    
    if (PredictionsState.manualTarget) {
      const featureCheckbox = document.querySelector(
        `.feature-checkbox[data-column="${PredictionsState.manualTarget}"]`
      );
      if (featureCheckbox?.classList.contains('selected')) {
        featureCheckbox.classList.remove('selected');
        const input = featureCheckbox.querySelector('input');
        if (input) input.checked = false;
        PredictionsState.manualFeatures = PredictionsState.manualFeatures.filter(
          f => f !== PredictionsState.manualTarget
        );
      }
    }
    
    this.updateFeatureCount();
    Logger.log(`Target column set: ${PredictionsState.manualTarget || '(none)'}`, 'info');
  },
  
  toggleFeature(checkbox) {
    const column = checkbox.dataset.column;
    
    if (column === PredictionsState.manualTarget) {
      Logger.log('Cannot select target column as a feature', 'warning');
      return;
    }
    
    checkbox.classList.toggle('selected');
    const input = checkbox.querySelector('input');
    input.checked = !input.checked;
    
    if (input.checked) {
      if (!PredictionsState.manualFeatures.includes(column)) {
        PredictionsState.manualFeatures.push(column);
      }
    } else {
      PredictionsState.manualFeatures = PredictionsState.manualFeatures.filter(f => f !== column);
    }
    
    this.updateFeatureCount();
  },
  
  selectAllFeatures() {
    const checkboxes = document.querySelectorAll('#feature-grid .feature-checkbox');
    PredictionsState.manualFeatures = [];
    
    checkboxes.forEach(cb => {
      const column = cb.dataset.column;
      if (column && column !== PredictionsState.manualTarget) {
        cb.classList.add('selected');
        const input = cb.querySelector('input');
        if (input) input.checked = true;
        PredictionsState.manualFeatures.push(column);
      }
    });
    
    this.updateFeatureCount();
  },
  
  deselectAllFeatures() {
    const checkboxes = document.querySelectorAll('#feature-grid .feature-checkbox');
    checkboxes.forEach(cb => {
      cb.classList.remove('selected');
      const input = cb.querySelector('input');
      if (input) input.checked = false;
    });
    
    PredictionsState.manualFeatures = [];
    this.updateFeatureCount();
  },
  
  updateFeatureCount() {
    const count = PredictionsState.manualFeatures.length;
    const label = document.getElementById('feature-count');
    if (label) {
      label.textContent = `${count} feature${count !== 1 ? 's' : ''} selected`;
    }
  },
  
  async askGemmaForColumns() {
    const availableColumns = PredictionsState.uploadedColumns.filter(
      col => !PredictionsState.excludedTargets.includes(col)
    );
    const columnsStr = availableColumns.map((col, i) => `${i + 1}. ${col}`).join('\n');
    
    const exclusionNote = PredictionsState.excludedTargets.length > 0
      ? `\n\nIMPORTANT: Do NOT select these columns as the target (they were already tried and gave poor results): ${PredictionsState.excludedTargets.join(', ')}`
      : '';
    
    const prompt = `You are analyzing a dataset with these columns:

${columnsStr}

Task: Select the best TARGET column for prediction (what we want to predict) and the FEATURE columns to use (input variables).

Rules:
- The target should be something meaningful to predict (not an ID)
- Features should be potential predictors of the target
- Do NOT include the target column in the features list${exclusionNote}

Respond in this EXACT format (no extra text):
target: [column name]
features: [column1, column2, column3, ...]`;
    
    try {
      const res = await fetch(`${API_BASE}/api/public/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 500
        })
      });
      
      const data = await res.json();
      const response = data.message || data.response || '';
      
      // Parse Gemma's selection
      const targetMatch = response.match(/target:\s*(.+)/i);
      const featuresMatch = response.match(/features:\s*(.+)/i);
      
      if (targetMatch && featuresMatch) {
        const target = targetMatch[1].trim().replace(/[\[\]"']/g, '');
        const featuresRaw = featuresMatch[1].replace(/[\[\]]/g, '').split(',')
          .map(f => f.trim().replace(/["']/g, ''));
        const features = featuresRaw.filter(f => 
          PredictionsState.uploadedColumns.includes(f) && f !== target
        );
        
        return { target, features, raw: response };
      }
    } catch (err) {
      console.warn('Gemma column selection failed:', err);
    }
    
    return null;
  }
};

export default ColumnSelection;
