(function (global) {
  const state = {
    allMemories: [],
    currentQuery: '',
    grid: {
      rows: [],
      selected: new Set(),
      lastSelectedIndex: null,
      bound: false,
    },
  };

  const els = {
    gridBody: document.getElementById('memories-grid-body'),
    rowCount: document.getElementById('memories-row-count'),
    status: document.getElementById('memories-status'),
    selection: document.getElementById('memories-selection'),
    filterSummary: document.getElementById('memories-filter-summary'),
    previewBody: document.getElementById('memories-preview-body'),
    previewEmpty: document.getElementById('memories-preview-empty'),
    previewTitle: document.getElementById('memories-preview-title'),
    previewMeta: document.getElementById('memories-preview-meta'),
    previewSnippet: document.getElementById('memories-preview-snippet'),
    previewContext: document.getElementById('memories-preview-context'),
    previewView: document.getElementById('memories-preview-view'),
    previewCopy: document.getElementById('memories-preview-copy'),
    copyAll: document.getElementById('memories-copy'),
    copySelected: document.getElementById('memories-copy-selected'),
    exportBtn: document.getElementById('memories-export'),
    searchInput: document.getElementById('memory-search'),
  };

  const statEls = {
    total: document.getElementById('total-memories'),
  };

  function escapeHTML(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return `${text.substring(0, maxLength)}...`;
  }

  function formatDateTime(value) {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString();
  }

  function highlightMatch(text, query) {
    if (!query) return escapeHTML(text || '');
    const safe = escapeHTML(text || '');
    const pattern = new RegExp(`(${query.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')})`, 'gi');
    return safe.replace(pattern, '<mark>$1</mark>');
  }

  function updateStats() {
    if (statEls.total) {
      statEls.total.textContent = state.allMemories.length;
    }
  }

  async function loadMemories() {
    if (els.status) els.status.textContent = 'Loading memories…';
    try {
      const transResult = await api.getRecentTranscripts(2000);
      const transcripts = (transResult.transcripts || [])
        .filter((t) => (t.full_text || '').trim())
        .map((t) => ({
          memory_id: t.job_id,
          title: t.title || `Session ${t.session_id ? t.session_id.substring(7, 20) : 'Unknown'}`,
          body: t.full_text,
          created_at: t.created_at,
          session_id: t.session_id,
          audio_duration: t.audio_duration,
          segment_count: Array.isArray(t.segments) ? t.segments.length : (t.segment_count || 0),
          speakers: t.speakers || [],
          dominant_emotion: t.dominant_emotion,
          job_id: t.job_id,
          segments: t.segments || [],
        }));
      state.allMemories = transcripts;
      updateStats();
      applyMemoryFilter();
      if (els.status) els.status.textContent = `Loaded ${transcripts.length} memories`;
    } catch (error) {
      console.error('[MEMORIES] Failed to load transcripts:', error);
      if (els.status) els.status.textContent = 'Failed to load memories';
      showToast('Error', 'Failed to load memories', 'error');
    }
  }

  function applyMemoryFilter() {
    const query = state.currentQuery.toLowerCase();
    const filtered = !query
      ? state.allMemories.slice()
      : state.allMemories.filter((memory) => {
          const haystack = `${memory.title || ''} ${memory.body || ''}`.toLowerCase();
          return haystack.includes(query);
        });
    renderMemoriesGrid(filtered);
    if (els.filterSummary) {
      els.filterSummary.innerHTML = query ? `<span class="badge">Search: ${escapeHTML(state.currentQuery)}</span>` : '';
    }
  }

  function renderMemoriesGrid(memories) {
    state.grid.rows = memories;
    state.grid.selected = new Set();
    state.grid.lastSelectedIndex = null;
    if (els.gridBody) {
      els.gridBody.innerHTML = memories.map(buildMemoryRow).join('');
    }
    attachGridEvents();
    updateRowCount(memories.length);
    updateSelectionStatus();
    resetPreview();
    if (global.lucide && global.lucide.createIcons) {
      global.lucide.createIcons();
    }
  }

  function buildMemoryRow(memory, index) {
    const created = formatDateTime(memory.created_at);
    const emotion = memory.dominant_emotion
      ? `<span class="sheet-emotion-pill emotion-${memory.dominant_emotion}">${escapeHTML(memory.dominant_emotion)}</span>`
      : '—';
    const speakers = (memory.speakers || []).slice(0, 3).map((speaker) => `<span class="sheet-chip">${escapeHTML(speaker)}</span>`).join('');
    const transcript = memory.memory_id || memory.job_id || '—';
    const segments = memory.segment_count || 0;
    const snippet = highlightMatch(truncateText(memory.body || '', 200), state.currentQuery);
    return `
      <tr data-index="${index}" tabindex="0">
        <td class="sheet-row-number">${index + 1}</td>
        <td><span class="sheet-chip">${escapeHTML(created)}</span></td>
        <td>${escapeHTML(memory.title || 'Transcript')}</td>
        <td>${emotion}</td>
        <td>${speakers || '—'}</td>
        <td><span class="sheet-chip">${escapeHTML(transcript)}</span></td>
        <td class="align-right">${segments}</td>
        <td class="sheet-text">${snippet}</td>
      </tr>`;
  }

  function attachGridEvents() {
    if (state.grid.bound || !els.gridBody) return;
    els.gridBody.addEventListener('click', handleRowClick);
    els.gridBody.addEventListener('keydown', handleRowKeydown);
    els.gridBody.addEventListener('dblclick', () => {
      if (state.grid.lastSelectedIndex == null) return;
      const memory = state.grid.rows[state.grid.lastSelectedIndex];
      if (memory) {
        viewTranscriptDetails(memory.memory_id || memory.job_id);
      }
    });
    state.grid.bound = true;
  }

  function handleRowClick(event) {
    const row = event.target.closest('tr[data-index]');
    if (!row) return;
    const index = Number(row.dataset.index);
    const additive = event.metaKey || event.ctrlKey;
    const range = event.shiftKey && state.grid.lastSelectedIndex != null;
    selectRow(index, { additive, range });
  }

  function handleRowKeydown(event) {
    if (!['ArrowDown', 'ArrowUp', 'Enter'].includes(event.key)) return;
    const row = event.target.closest('tr[data-index]');
    if (!row) return;
    const index = Number(row.dataset.index);
    if (event.key === 'Enter') {
      selectRow(index);
      return;
    }
    event.preventDefault();
    const delta = event.key === 'ArrowDown' ? 1 : -1;
    const next = Math.max(0, Math.min(state.grid.rows.length - 1, index + delta));
    const nextRow = els.gridBody.querySelector(`tr[data-index="${next}"]`);
    if (nextRow) nextRow.focus();
    selectRow(next);
  }

  function selectRow(index, { additive = false, range = false } = {}) {
    if (index < 0 || index >= state.grid.rows.length) return;
    let selection = new Set(state.grid.selected);
    if (range && state.grid.lastSelectedIndex != null) {
      const start = Math.min(state.grid.lastSelectedIndex, index);
      const end = Math.max(state.grid.lastSelectedIndex, index);
      for (let i = start; i <= end; i += 1) {
        selection.add(i);
      }
    } else if (additive) {
      if (selection.has(index)) selection.delete(index);
      else selection.add(index);
    } else {
      selection = new Set([index]);
    }
    state.grid.selected = selection;
    state.grid.lastSelectedIndex = index;
    updateSelectionStatus();
    updatePreview(state.grid.rows[index], index);
    if (els.gridBody) {
      els.gridBody.querySelectorAll('tr[data-index]').forEach((row) => {
        const idx = Number(row.dataset.index);
        if (selection.has(idx)) row.classList.add('selected');
        else row.classList.remove('selected');
      });
    }
  }

  function updateRowCount(count) {
    if (els.rowCount) {
      els.rowCount.textContent = `${count} ${count === 1 ? 'row' : 'rows'}`;
    }
  }

  function updateSelectionStatus() {
    const count = state.grid.selected.size;
    if (els.selection) {
      els.selection.textContent = count ? `Selected ${count} ${count === 1 ? 'row' : 'rows'}` : 'No rows selected';
    }
  }

  function resetPreview() {
    if (els.previewBody) els.previewBody.hidden = true;
    if (els.previewEmpty) els.previewEmpty.hidden = false;
  }

  function updatePreview(memory, index) {
    if (!els.previewBody || !els.previewEmpty) return;
    els.previewBody.hidden = false;
    els.previewEmpty.hidden = true;
    if (els.previewTitle) {
      els.previewTitle.textContent = `${memory.title || 'Transcript'} • ${formatDateTime(memory.created_at)}`;
    }
    if (els.previewMeta) {
      const badges = [];
      if (memory.job_id) badges.push(`<span class="sheet-chip">Job ${escapeHTML(memory.job_id)}</span>`);
      if (memory.segment_count) badges.push(`<span class="sheet-chip">${memory.segment_count} segments</span>`);
      els.previewMeta.innerHTML = badges.join('');
    }
    if (els.previewSnippet) {
      els.previewSnippet.innerHTML = highlightMatch(memory.body || '', state.currentQuery);
    }
    if (els.previewContext) {
      const speakers = (memory.speakers || []).map((speaker) => `<span class="sheet-chip">${escapeHTML(speaker)}</span>`).join('');
      els.previewContext.innerHTML = speakers || '<p class="text-muted">No speakers recorded</p>';
    }
    if (els.previewView) els.previewView.dataset.index = String(index);
    if (els.previewCopy) els.previewCopy.dataset.index = String(index);
  }

  function getSelectedRows(selectedOnly = false) {
    if (!selectedOnly || state.grid.selected.size === 0) {
      return state.grid.rows.slice();
    }
    const indices = Array.from(state.grid.selected).sort((a, b) => a - b);
    return indices.map((idx) => state.grid.rows[idx]).filter(Boolean);
  }

  function copyRows(selectedOnly = false) {
    const rows = getSelectedRows(selectedOnly);
    if (!rows.length) {
      showToast('Clipboard', 'No rows to copy.', 'warning');
      return;
    }
    const lines = rows.map((memory) => {
      const time = formatDateTime(memory.created_at);
      const text = (memory.body || '').replace(/\s+/g, ' ').trim();
      return `${time}\t${memory.title || 'Transcript'}\t${text}`;
    });
    const payload = lines.join('\n');
    const write = navigator?.clipboard?.writeText
      ? navigator.clipboard.writeText(payload)
      : Promise.reject();
    Promise.resolve(write)
      .then(() => showToast('Copied', 'Rows copied to clipboard.', 'success'))
      .catch(() => {
        try {
          const ta = document.createElement('textarea');
          ta.value = payload;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
          showToast('Copied', 'Rows copied to clipboard.', 'success');
        } catch (error) {
          showToast('Clipboard', 'Copy failed.', 'error');
        }
      });
  }

  function copySnippet(memory) {
    const text = memory.body || '';
    const write = navigator?.clipboard?.writeText
      ? navigator.clipboard.writeText(text)
      : Promise.reject();
    Promise.resolve(write)
      .then(() => showToast('Copied', 'Snippet copied to clipboard.', 'success'))
      .catch(() => showToast('Clipboard', 'Copy failed.', 'error'));
  }

  function exportCsv() {
    const rows = state.grid.rows;
    if (!rows.length) {
      showToast('Export', 'No memories to export.', 'warning');
      return;
    }
    const header = ['Created At', 'Title', 'Emotion', 'Speakers', 'Transcript', 'Segments', 'Snippet'];
    const lines = [header.join(',')];
    rows.forEach((memory) => {
      const speakers = (memory.speakers || []).join(' | ');
      const line = [
        formatDateTime(memory.created_at),
        memory.title || 'Transcript',
        memory.dominant_emotion || '',
        speakers,
        memory.job_id || '',
        memory.segment_count || 0,
        (memory.body || '').replace(/\s+/g, ' ').trim(),
      ].map((value) => '"' + String(value ?? '').replace(/"/g, '""') + '"');
      lines.push(line.join(','));
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const ts = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
    a.download = `memories_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Export', `Exported ${rows.length} rows.`, 'success');
  }

  // --- CRUD helpers reused from previous implementation ---
  function showCreateMemoryModal() {
    const modal = showModal('Create New Memory', `
      <form id="create-memory-form">
        <div style="margin-bottom: 1rem;">
          <label style="display: block; margin-bottom: 0.5rem; font-weight: 600;">Title</label>
          <input type="text" class="input" id="memory-title" placeholder="Memory title..." required>
        </div>
        <div style="margin-bottom: 1.5rem;">
          <label style="display: block; margin-bottom: 0.5rem; font-weight: 600;">Content</label>
          <textarea class="input" id="memory-content" rows="6" placeholder="Memory content..." required></textarea>
        </div>
        <div style="display: flex; gap: 0.5rem; justify-content: flex-end;">
          <button type="button" class="btn btn-ghost" onclick="closeModal()">Cancel</button>
          <button type="submit" class="btn btn-primary">Create</button>
        </div>
      </form>
    `);

    document.getElementById('create-memory-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const title = document.getElementById('memory-title').value;
      const content = document.getElementById('memory-content').value;

      try {
        await api.createMemory(title, content);
        showToast('Success', 'Memory created successfully', 'success');
        closeModal();
        loadMemories();
      } catch (error) {
        showToast('Error', 'Failed to create memory', 'error');
      }
    });

    return modal;
  }

  async function editMemory(id) {
    const memory = state.allMemories.find((m) => (m.memory_id === id || m.id === id));
    if (!memory) return;

    showModal('Edit Memory', `
      <form id="edit-memory-form">
        <input type="hidden" id="edit-memory-id" value="${id}">
        <div style="margin-bottom: 1rem;">
          <label style="display: block; margin-bottom: 0.5rem; font-weight: 600;">Title</label>
          <input type="text" class="input" id="edit-memory-title" value="${escapeHTML(memory.title)}" required>
        </div>
        <div style="margin-bottom: 1.5rem;">
          <label style="display: block; margin-bottom: 0.5rem; font-weight: 600;">Content</label>
          <textarea class="input" id="edit-memory-content" rows="6" required>${escapeHTML(memory.body || memory.content || '')}</textarea>
        </div>
        <div style="display: flex; gap: 0.5rem; justify-content: flex-end;">
          <button type="button" class="btn btn-ghost" onclick="closeModal()">Cancel</button>
          <button type="submit" class="btn btn-primary">Save</button>
        </div>
      </form>
    `);

    document.getElementById('edit-memory-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const title = document.getElementById('edit-memory-title').value;
      const content = document.getElementById('edit-memory-content').value;

      try {
        await api.updateMemory(id, title, content);
        showToast('Success', 'Memory updated successfully', 'success');
        closeModal();
        loadMemories();
      } catch (error) {
        showToast('Error', 'Failed to update memory', 'error');
      }
    });
  }

  async function deleteMemory(id) {
    if (!confirm('Are you sure you want to delete this memory?')) return;

    try {
      await api.deleteMemory(id);
      showToast('Success', 'Memory deleted successfully', 'success');
      loadMemories();
    } catch (error) {
      showToast('Error', 'Failed to delete memory', 'error');
    }
  }

  function showComprehensiveAnalysis() {
    window.location.href = 'analysis.html';
  }

  async function viewTranscriptDetails(jobId) {
    const memory = state.allMemories.find((m) => m.memory_id === jobId || m.job_id === jobId);
    if (!memory) {
      showToast('Error', 'Transcript not found', 'error');
      return;
    }

    const emotionColors = {
      joy: '#10b981',
      sadness: '#3b82f6',
      anger: '#ef4444',
      fear: '#8b5cf6',
      surprise: '#f59e0b',
      disgust: '#84cc16',
      neutral: '#6b7280',
    };

    let segmentsHtml = '';
    if (memory.segments && memory.segments.length > 0) {
      segmentsHtml = memory.segments.map((seg) => {
        const emotionColor = seg.emotion ? (emotionColors[seg.emotion] || '#6b7280') : '#6b7280';
        const normalizedConfidence = typeof global.normalizeEmotionConfidence === 'function'
          ? global.normalizeEmotionConfidence(seg.emotion_confidence)
          : (typeof seg.emotion_confidence === 'number' ? Math.max(0, Math.min(1, seg.emotion_confidence)) : 0.7);
        const confidence = `${Math.round(normalizedConfidence * 100)}%`;
        const timeRange = seg.start_time !== null && seg.end_time !== null
          ? `${seg.start_time.toFixed(1)}s - ${seg.end_time.toFixed(1)}s`
          : '';

        return `
          <div style="padding: 1rem; background: var(--glass-bg); border-radius: 8px; margin-bottom: 0.75rem; border-left: 3px solid ${emotionColor};">
            <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
              <span class="badge badge-info" style="font-size: 0.75rem;">
                <i data-lucide="user" style="width: 12px; height: 12px;"></i>
                ${escapeHTML(seg.speaker || 'Unknown')}
              </span>
              ${seg.emotion ? `
                <span class="badge" style="font-size: 0.75rem; background: ${emotionColor}20; color: ${emotionColor}; border: 1px solid ${emotionColor}40;">
                  <i data-lucide="smile" style="width: 12px; height: 12px;"></i>
                  ${escapeHTML(seg.emotion)} ${confidence}
                </span>
              ` : ''}
              ${timeRange ? `
                <span class="badge badge-secondary" style="font-size: 0.7rem;">
                  <i data-lucide="clock" style="width: 10px; height: 10px;"></i>
                  ${timeRange}
                </span>
              ` : ''}
            </div>
            <p style="margin: 0; line-height: 1.6;">${escapeHTML(seg.text || '')}</p>
          </div>
        `;
      }).join('');
    } else {
      segmentsHtml = '<p class="text-muted">No segment data available</p>';
    }

    const timestamp = memory.created_at ? new Date(memory.created_at).toLocaleString() : 'Unknown';
    const duration = memory.audio_duration ? `${Math.round(memory.audio_duration)}s` : 'Unknown';

    showModal(`Transcript Details - ${escapeHTML(memory.title)}`, `
      <div style="margin-bottom: 1.5rem;">
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem;">
          <span class="badge badge-secondary">
            <i data-lucide="clock" style="width: 12px; height: 12px;"></i>
            ${timestamp}
          </span>
          <span class="badge badge-secondary">
            <i data-lucide="timer" style="width: 12px; height: 12px;"></i>
            ${duration}
          </span>
          <span class="badge badge-secondary">
            <i data-lucide="layers" style="width: 12px; height: 12px;"></i>
            ${memory.segment_count} segments
          </span>
        </div>

        <h4 style="margin-bottom: 0.5rem;">Speakers</h4>
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.5rem;">
          ${(memory.speakers || []).map((speaker) => `
            <span class="badge badge-info">
              <i data-lucide="user" style="width: 12px; height: 12px;"></i>
              ${escapeHTML(speaker)}
            </span>
          `).join('') || '<span class="text-muted">No speaker data</span>'}
        </div>

        <h4 style="margin-bottom: 1rem;">Transcript Segments</h4>
        <div style="max-height: 400px; overflow-y: auto;">
          ${segmentsHtml}
        </div>
      </div>

      <div style="display: flex; justify-content: flex-end; margin-top: 1.5rem;">
        <button class="btn btn-primary" onclick="closeModal()">Close</button>
      </div>
    `);

    if (global.lucide && global.lucide.createIcons) {
      global.lucide.createIcons();
    }
  }

  function bindEvents() {
    els.searchInput?.addEventListener('input', () => {
      state.currentQuery = els.searchInput.value.trim();
      applyMemoryFilter();
    });
    els.copyAll?.addEventListener('click', () => copyRows(false));
    els.copySelected?.addEventListener('click', () => copyRows(true));
    els.exportBtn?.addEventListener('click', () => exportCsv());
    els.previewCopy?.addEventListener('click', (event) => {
      const idx = Number(event.currentTarget.dataset.index);
      const memory = state.grid.rows[idx];
      if (memory) copySnippet(memory);
    });
    els.previewView?.addEventListener('click', (event) => {
      const idx = Number(event.currentTarget.dataset.index);
      const memory = state.grid.rows[idx];
      if (memory) viewTranscriptDetails(memory.memory_id || memory.job_id);
    });
  }

  async function initPage() {
    const authenticated = await Auth.init({ requireAuth: true });
    if (!authenticated) return;
    await loadMemories();
  }

  bindEvents();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
  } else {
    initPage();
  }

  global.showCreateMemoryModal = showCreateMemoryModal;
  global.editMemory = editMemory;
  global.deleteMemory = deleteMemory;
  global.showComprehensiveAnalysis = showComprehensiveAnalysis;
  global.viewTranscriptDetails = viewTranscriptDetails;
})(window);
