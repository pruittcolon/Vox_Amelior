/* global api, lucide, showToast */

(function () {
  let timelineChart = null;
  let topWordsChart = null;
  let selectedSpeakers = ['pruitt', 'ericah'];

  function ensureCanvasSize(canvas) {
    if (!canvas) return { width: 800, height: 320 };
    const parentRect = canvas.parentElement ? canvas.parentElement.getBoundingClientRect() : null;
    const rect = parentRect && parentRect.width > 0 ? parentRect : canvas.getBoundingClientRect();
    const maxDim = 4096;
    const minWidth = 320;
    const minHeight = 240;
    const width = Math.min(Math.max(rect.width || minWidth, minWidth), maxDim);
    const height = Math.min(Math.max(rect.height || 320, minHeight), maxDim);
    canvas.width = width;
    canvas.height = height;
    return { width, height };
  }

  function safeNumber(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '0.00';
    return n.toFixed(digits);
  }

  function parseSpeakerInput(value) {
    if (!value) return [];
    return value
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  }

  async function ensureAuth() {
    try {
      const response = await fetch('/api/auth/check', { credentials: 'include' });
      const data = await response.json();
      if (!data.valid) {
        window.location.href = 'login.html';
        return null;
      }
      const userDisplay = document.getElementById('current-user');
      if (userDisplay && data.user) {
        userDisplay.textContent = `${data.user.username} (${data.user.role})`;
      }
      const logoutBtn = document.getElementById('logout-btn');
      if (logoutBtn) {
        logoutBtn.style.display = 'flex';
        logoutBtn.onclick = async () => {
          await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
          window.location.href = 'login.html';
        };
      }
      return data.user || {};
    } catch (err) {
      console.error('[VOCAB] auth check failed', err);
      window.location.href = 'login.html';
      return null;
    }
  }

  function updateGlobalStats(globalStats) {
    if (!globalStats) return;
    const totalWordsEl = document.getElementById('vocab-total-words');
    const uniqueWordsEl = document.getElementById('vocab-unique-words');
    const ttrEl = document.getElementById('vocab-ttr');
    const daysEl = document.getElementById('vocab-days');

    if (totalWordsEl) totalWordsEl.textContent = (globalStats.total_words || 0).toLocaleString();
    if (uniqueWordsEl) uniqueWordsEl.textContent = (globalStats.unique_words || 0).toLocaleString();
    if (ttrEl) ttrEl.textContent = safeNumber(globalStats.type_token_ratio || 0, 3);
    if (daysEl) daysEl.textContent = String(globalStats.days_covered || 0);

    const highlightsEl = document.getElementById('vocab-speaker-highlights');
    if (highlightsEl) {
      highlightsEl.innerHTML = '';
      const talker = globalStats.most_talkative_speaker;
      const richest = globalStats.richest_speaker;
      if (!talker && !richest) {
        const p = document.createElement('p');
        p.className = 'text-muted';
        p.style.fontSize = '0.85rem';
        p.textContent = 'No speaker highlights yet for this range.';
        highlightsEl.appendChild(p);
      } else {
        if (talker) {
          const row = document.createElement('div');
          row.className = 'stat-multi-row';
          row.innerHTML = `
            <span>Most talkative speaker</span>
            <span><strong>${talker}</strong></span>
          `;
          highlightsEl.appendChild(row);
        }
        if (richest) {
          const row = document.createElement('div');
          row.className = 'stat-multi-row';
          row.innerHTML = `
            <span>Richest vocabulary</span>
            <span><strong>${richest}</strong></span>
          `;
          highlightsEl.appendChild(row);
        }
      }
    }
  }

  function updateSpeakerTable(perSpeaker) {
    const tbody = document.querySelector('#vocab-speaker-table tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    const entries = Object.entries(perSpeaker || {}).sort((a, b) => {
      const aw = a[1]?.total_words || 0;
      const bw = b[1]?.total_words || 0;
      return bw - aw;
    });
    if (!entries.length) {
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 8;
      cell.textContent = 'No speaker data for this range yet.';
      cell.className = 'text-muted';
      row.appendChild(cell);
      tbody.appendChild(row);
      return;
    }
    entries.forEach(([speaker, stats]) => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${speaker}</td>
        <td>${stats.total_segments ?? 0}</td>
        <td>${(stats.total_words ?? 0).toLocaleString()}</td>
        <td>${(stats.unique_words ?? 0).toLocaleString()}</td>
        <td>${safeNumber(stats.type_token_ratio ?? 0, 3)}</td>
        <td>${safeNumber(stats.avg_words_per_segment ?? 0, 2)}</td>
        <td>${safeNumber(stats.avg_words_per_sentence ?? 0, 2)}</td>
        <td>${safeNumber((stats.share_of_global_words || 0) * 100, 1)}%</td>
      `;
      tbody.appendChild(row);
    });
  }

  function renderTimeline(timeline) {
    const ctx = document.getElementById('vocab-timeline-chart');
    if (!ctx) return;
    const { width, height } = ensureCanvasSize(ctx);
    const dates = (timeline && Array.isArray(timeline.dates) && timeline.dates) || [];
    let labels = dates.slice();
    let uniqueSeries = (timeline && timeline.unique_words) || [];
    let countSeries = (timeline && timeline.word_counts) || [];

    // Trim very long timelines so the chart does not grow unbounded horizontally.
    // Keep the most recent 20 points to make the view consistently readable.
    const MAX_POINTS = 20;
    if (labels.length > MAX_POINTS) {
      const start = labels.length - MAX_POINTS;
      labels = labels.slice(start);
      uniqueSeries = uniqueSeries.slice(start);
      countSeries = countSeries.slice(start);
    }

    if (timelineChart) {
      timelineChart.destroy();
      timelineChart = null;
    }

    // Normalize and coerce to numbers (Chart.js is picky about nulls/strings).
    const coerce = (arr) =>
      (arr || []).map((v) => {
        const n = Number(v);
        return Number.isFinite(n) ? n : 0;
      });
    uniqueSeries = coerce(uniqueSeries);
    countSeries = coerce(countSeries);

    // When there is no data, render a placeholder chart instead of leaving it blank.
    if (!labels.length || (!uniqueSeries.some((v) => v !== 0) && !countSeries.some((v) => v !== 0))) {
      labels = ['No data'];
      uniqueSeries = [0];
      countSeries = [0];
    }

    // Compute a reasonable Y max for better visibility, clipping outliers.
    const allValues = [...uniqueSeries, ...countSeries].filter((v) => typeof v === 'number' && isFinite(v));
    const quantile = (arr, p) => {
      if (!arr.length) return 0;
      const sorted = [...arr].sort((a, b) => a - b);
      const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * sorted.length)));
      return sorted[idx];
    };
    const q95 = quantile(allValues, 0.95);
    const maxValue = allValues.length ? Math.max(...allValues) : 0;
    const baseMax = Math.max(q95, maxValue * 0.5);
    const suggestedMax = baseMax > 0 ? Math.max(baseMax * 1.2, 10) : 1;

    try {
      timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Unique words',
              data: uniqueSeries,
              borderColor: 'rgba(56, 189, 248, 0.9)',
              backgroundColor: 'rgba(56, 189, 248, 0.2)',
              tension: 0.25,
              borderWidth: 2,
              pointRadius: 4,
              pointBackgroundColor: 'rgba(56,189,248,1)',
              spanGaps: false,
            },
            {
              label: 'Total words',
              data: countSeries,
              borderColor: 'rgba(129, 140, 248, 0.9)',
              backgroundColor: 'rgba(129, 140, 248, 0.2)',
              tension: 0.25,
              borderWidth: 2,
              pointRadius: 4,
              pointBackgroundColor: 'rgba(129,140,248,1)',
              spanGaps: false,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          resizeDelay: 150,
          layout: {
            padding: { top: 8, right: 8, bottom: 12, left: 8 },
          },
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: {
              labels: { color: '#e5e7eb', font: { size: 11 } },
            },
          },
          scales: {
            x: {
              ticks: { color: '#9ca3af', maxTicksLimit: 7 },
              grid: { color: 'rgba(55,65,81,0.5)' },
            },
            y: {
              ticks: { color: '#9ca3af' },
              suggestedMin: 0,
              suggestedMax,
              grid: { color: 'rgba(31,41,55,0.5)' },
            },
          },
        },
      });
    } catch (err) {
      console.error('[VOCAB] Failed to render timeline chart', err, { width, height, labelsCount: labels.length });
      const fallback = document.createElement('p');
      fallback.className = 'text-muted';
      fallback.style.fontSize = '0.9rem';
      fallback.style.margin = '0.5rem 0';
      fallback.textContent = 'Timeline chart unavailable on this device/browser.';
      if (ctx.parentElement) {
        ctx.parentElement.appendChild(fallback);
      }
    }
  }

  function renderTopWords(topWords) {
    const ctx = document.getElementById('vocab-top-words-chart');
    if (!ctx) return;
    const { width, height } = ensureCanvasSize(ctx);
    const items = (topWords && topWords.filtered) || (topWords && topWords.global) || [];
    // Show a compact set of top words to avoid an overly dense bar strip.
    const trimmed = items.slice(0, 6);
    const labels = trimmed.map((i) => i.word);
    const counts = trimmed.map((i) => i.count);

    if (topWordsChart) {
      topWordsChart.destroy();
      topWordsChart = null;
    }
    if (!labels.length) {
      return;
    }

    const maxVal = counts.length ? Math.max(...counts) : 0;
    const suggestedMax = maxVal > 0 ? Math.max(maxVal * 1.2, maxVal + 5, 10) : 1;

    try {
      topWordsChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'Frequency',
              data: counts,
              backgroundColor: 'rgba(129, 140, 248, 0.9)',
              maxBarThickness: 36,
              borderRadius: 6,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          resizeDelay: 150,
          layout: {
            padding: {
              top: 8,
              bottom: 32,
              left: 8,
              right: 8,
            },
          },
          plugins: {
            legend: { display: false },
          },
          scales: {
            x: {
              ticks: { color: '#9ca3af', maxRotation: 60, minRotation: 30, font: { size: 10 } },
              grid: { display: false },
            },
            y: {
              ticks: { color: '#9ca3af' },
              suggestedMin: 0,
              suggestedMax,
              grid: { color: 'rgba(31,41,55,0.5)' },
            },
          },
        },
      });
    } catch (err) {
      console.error('[VOCAB] Failed to render top-words chart', err, { width, height, labelsCount: labels.length });
      const fallback = document.createElement('p');
      fallback.className = 'text-muted';
      fallback.style.fontSize = '0.9rem';
      fallback.style.margin = '0.5rem 0';
      fallback.textContent = 'Top words chart unavailable on this device/browser.';
      if (ctx.parentElement) {
        ctx.parentElement.appendChild(fallback);
      }
    }
  }

  function normalizeSpeakerId(name) {
    return (name || '').toString().trim().toLowerCase();
  }

  function filterPerSpeaker(perSpeaker) {
    if (!Array.isArray(selectedSpeakers) || !selectedSpeakers.length) return perSpeaker || {};
    const selected = new Set(selectedSpeakers.map(normalizeSpeakerId));
    const out = {};
    Object.entries(perSpeaker || {}).forEach(([k, v]) => {
      if (selected.has(normalizeSpeakerId(k))) {
        out[k] = v;
      }
    });
    return out;
  }

  function aggregateTopWords(topWords) {
    if (!topWords) return [];
    if (!Array.isArray(selectedSpeakers) || !selectedSpeakers.length) {
      return topWords.global || [];
    }
    const per = topWords.per_speaker || {};
    const selected = new Set(selectedSpeakers.map(normalizeSpeakerId));
    const counter = new Map();
    Object.entries(per).forEach(([speaker, words]) => {
      if (!selected.has(normalizeSpeakerId(speaker))) return;
      (words || []).forEach((entry) => {
        const w = (entry.word || '').trim();
        if (w.length < 3) return;
        const c = Number(entry.count) || 0;
        counter.set(w, (counter.get(w) || 0) + c);
      });
    });
    const merged = Array.from(counter.entries())
      .map(([word, count]) => ({ word, count }))
      .sort((a, b) => b.count - a.count);
    return merged.slice(0, 40);
  }

  function renderSpeakerChips(perSpeaker) {
    const container = document.getElementById('vocab-speaker-chips');
    if (!container) return;
    container.innerHTML = '';
    const keys = Object.keys(perSpeaker || {});
    if (!keys.length) {
      container.textContent = 'No speakers available yet.';
      return;
    }
    const sorted = keys.sort((a, b) => a.localeCompare(b));

    function toggleSpeaker(name) {
      const id = normalizeSpeakerId(name);
      if (!id) return;
      if (selectedSpeakers.map(normalizeSpeakerId).includes(id)) {
        selectedSpeakers = selectedSpeakers.filter((s) => normalizeSpeakerId(s) !== id);
      } else {
        selectedSpeakers = [...selectedSpeakers, name];
      }
      if (!selectedSpeakers.length) {
        selectedSpeakers = [];
      }
    }

    // "All" chip
    const allBtn = document.createElement('button');
    allBtn.className = 'filter-chip';
    if (!selectedSpeakers.length) allBtn.classList.add('active');
    allBtn.textContent = 'All speakers';
    allBtn.onclick = () => {
      selectedSpeakers = [];
      renderSpeakerChips(perSpeaker);
      loadVocabSummary();
    };
    container.appendChild(allBtn);

    sorted.forEach((name) => {
      const btn = document.createElement('button');
      btn.className = 'filter-chip';
      if (selectedSpeakers.length && selectedSpeakers.map(normalizeSpeakerId).includes(normalizeSpeakerId(name))) {
        btn.classList.add('active');
      }
      btn.textContent = name;
      btn.onclick = () => {
        toggleSpeaker(name);
        // If we removed all, treat as "all"
        renderSpeakerChips(perSpeaker);
        loadVocabSummary();
      };
      container.appendChild(btn);
    });
  }

  function renderFingerprints(distinctive) {
    const fingerprintsEl = document.getElementById('vocab-fingerprints');
    if (!fingerprintsEl) return;
    fingerprintsEl.innerHTML = '';
    const selected = new Set((selectedSpeakers || []).map(normalizeSpeakerId));
    const entries = Object.entries(distinctive || {}).filter(([name]) => {
      if (!selected.size) return true;
      return selected.has(normalizeSpeakerId(name));
    });
    if (!entries.length) {
      const p = document.createElement('p');
      p.className = 'text-muted';
      p.style.fontSize = '0.85rem';
      p.textContent = 'Distinctive word patterns will appear here once there is enough data for selected speakers.';
      fingerprintsEl.appendChild(p);
      return;
    }
    entries.sort((a, b) => ((b[1] || []).length || 0) - ((a[1] || []).length || 0));
    entries.forEach(([speaker, words]) => {
      if (!words || !words.length) return;
      const chip = document.createElement('div');
      chip.className = 'filter-chip';
      const topWords = words.slice(0, 5).map((w) => w.word).join(', ');
      chip.textContent = `${speaker}: ${topWords}`;
      fingerprintsEl.appendChild(chip);
    });
  }

  async function loadVocabSummary() {
    const startInput = document.getElementById('vocab-start-date');
    const endInput = document.getElementById('vocab-end-date');
    const params = {};
    if (startInput && startInput.value) params.start_date = startInput.value;
    if (endInput && endInput.value) params.end_date = endInput.value;
    if (selectedSpeakers && selectedSpeakers.length) {
      params.speakers = selectedSpeakers;
    }

    try {
      const summary = await api.getVocabSummary(params);
      if (!summary) return;
      const perSpeaker = summary.per_speaker || {};

      // Normalize selected speakers against available list (fallback to all if none match)
      const available = Object.keys(perSpeaker);
      if (selectedSpeakers.length) {
        const normalizedSel = new Set(selectedSpeakers.map(normalizeSpeakerId));
        const matched = available.filter((s) => normalizedSel.has(normalizeSpeakerId(s)));
        selectedSpeakers = matched;
      }

      renderSpeakerChips(perSpeaker);

      updateGlobalStats(summary.global || {});
      const filteredPerSpeaker = filterPerSpeaker(perSpeaker);
      updateSpeakerTable(filteredPerSpeaker);
      renderTimeline(summary.timeline || {});
      const filteredTopWords = aggregateTopWords(summary.top_words || {});
      renderTopWords({ filtered: filteredTopWords, global: summary.top_words?.global || [] });

      // Notable shifts
      const shifts = (summary.timeline && summary.timeline.notable_shifts) || [];
      const shiftsEl = document.getElementById('vocab-notable-shifts');
      if (shiftsEl) {
        shiftsEl.innerHTML = '';
        if (!shifts.length) {
          const li = document.createElement('li');
          li.className = 'text-muted';
          li.style.fontSize = '0.85rem';
          li.textContent = 'No standout vocabulary shifts detected in this range yet.';
          shiftsEl.appendChild(li);
        } else {
          shifts.forEach((shift) => {
            const li = document.createElement('li');
            const sign = shift.delta_unique > 0 ? '+' : '';
            const dirLabel = shift.direction === 'up' ? 'increase' : 'drop';
            li.textContent = `${shift.date}: ${sign}${shift.delta_unique} unique words (${dirLabel})`;
            shiftsEl.appendChild(li);
          });
        }
      }

      // Lexical fingerprints
      renderFingerprints((summary.top_words && summary.top_words.distinctive_per_speaker) || {});
    } catch (err) {
      console.error('[VOCAB] Failed to load summary', err);
      if (typeof showToast === 'function') {
        showToast('Error', 'Failed to load vocabulary insights', 'error');
      }
    }
  }

  async function initVocabPage() {
    // Ensure we are served via gateway, not file://
    if (window.location.protocol === 'file:') {
      // Let app.js redirect to /ui if configured
      if (typeof window.redirectToGatewayUI === 'function') {
        window.redirectToGatewayUI('vocab.html');
        return;
      }
    }

    const user = await ensureAuth();
    if (!user) return;

    // Speaker input mirrors selectedSpeakers for manual override
    const speakersInput = document.getElementById('vocab-speakers');
    if (speakersInput) {
      speakersInput.value = selectedSpeakers.join(', ');
    }

    // Wire up quick date chips
    const startInput = document.getElementById('vocab-start-date');
    const endInput = document.getElementById('vocab-end-date');
    const chipContainer = document.getElementById('vocab-quick-dates');

    function formatDate(d) {
      const year = d.getFullYear();
      const month = String(d.getMonth() + 1).padStart(2, '0');
      const day = String(d.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    }

    function applyQuickRange(days) {
      if (!startInput || !endInput) return;
      if (days === 'all') {
        startInput.value = '';
        endInput.value = '';
        return;
      }
      const n = Number(days);
      if (!Number.isFinite(n) || n <= 0) return;
      const end = new Date();
      const start = new Date();
      start.setDate(end.getDate() - (n - 1));
      startInput.value = formatDate(start);
      endInput.value = formatDate(end);
    }

    if (chipContainer) {
      chipContainer.querySelectorAll('button[data-vocab-quick-days]').forEach((btn) => {
        btn.addEventListener('click', () => {
          const value = btn.getAttribute('data-vocab-quick-days');
          chipContainer.querySelectorAll('button[data-vocab-quick-days]').forEach((other) => {
            other.classList.remove('active');
          });
          btn.classList.add('active');
          applyQuickRange(value);
          loadVocabSummary();
        });
      });
    }

    // Default to all time on first load
    applyQuickRange('all');
    if (chipContainer) {
      const allBtn = chipContainer.querySelector('button[data-vocab-quick-days="all"]');
      if (allBtn) {
        allBtn.classList.add('active');
      }
    }

    // Wire up filters
    const applyBtn = document.getElementById('vocab-apply-filters');
    if (applyBtn) {
      applyBtn.addEventListener('click', () => {
        const speakersInputEl = document.getElementById('vocab-speakers');
        if (speakersInputEl && speakersInputEl.value) {
          const parsed = parseSpeakerInput(speakersInputEl.value);
          selectedSpeakers = parsed;
        } else if (speakersInputEl && !speakersInputEl.value) {
          selectedSpeakers = [];
        }
        loadVocabSummary();
      });
    }

    // Initial load
    await loadVocabSummary();

    // Icons
    if (typeof lucide !== 'undefined' && lucide.createIcons) {
      lucide.createIcons();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initVocabPage);
  } else {
    initVocabPage();
  }
})();
