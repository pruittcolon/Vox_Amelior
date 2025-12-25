(function (global) {
  const apiClient = global.api;
  const DEFAULT_SPEAKER_FOCUS = ['pruitt', 'ericah'];
  const EMO_CONF_THRESH = 0.8;

  function uuid() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
    return 'cli-' + Math.random().toString(36).slice(2, 10);
  }

  function nowTs() {
    return new Date().toLocaleTimeString();
  }

  function tokenize(input) {
    const tokens = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < input.length; i += 1) {
      const ch = input[i];
      if (ch === '"') {
        inQuotes = !inQuotes;
        continue;
      }
      if (!inQuotes && /\s/.test(ch)) {
        if (current) tokens.push(current);
        current = '';
        continue;
      }
      current += ch;
    }
    if (current) tokens.push(current);
    return tokens;
  }

  function formatMs(ms) {
    if (!Number.isFinite(ms)) return 'n/a';
    if (ms < 1000) return `${ms.toFixed(0)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
  }

  function summarizeResults(results = []) {
    if (!Array.isArray(results) || !results.length) return 'no results';
    const topScore = typeof results[0].score === 'number' ? results[0].score.toFixed(3) : 'n/a';
    return `${results.length} results • top score ${topScore}`;
  }

  function buildQueryFromTokens(tokens, startIndex = 0) {
    return tokens.slice(startIndex).join(' ').trim();
  }

  function parseSpeakers(tokens) {
    const speakers = [];
    let allSpeakers = false;
    for (let i = 0; i < tokens.length; i += 1) {
      const t = tokens[i].toLowerCase();
      if (t === '--speaker' || t === '-s') {
        const val = tokens[i + 1];
        if (val) speakers.push(val.toLowerCase());
      }
      if (t === '--all-speakers') {
        allSpeakers = true;
      }
    }
    if (allSpeakers) return [];
    if (speakers.length) return speakers;
    return DEFAULT_SPEAKER_FOCUS.slice();
  }

  function parsePeriod(tokens) {
    const idx = tokens.findIndex((t) => t === '--period');
    if (idx === -1 || !tokens[idx + 1]) return null;
    const period = tokens[idx + 1].toLowerCase();
    const today = new Date();
    const endDate = today.toISOString().slice(0, 10);
    if (period === 'today') {
      return { start_date: endDate, end_date: endDate, label: 'today (last 24h)' };
    }
    if (period === 'week' || period === 'thisweek' || period === 'this-week') {
      const start = new Date();
      start.setDate(start.getDate() - 6);
      return { start_date: start.toISOString().slice(0, 10), end_date: endDate, label: 'this week (last 7d)' };
    }
    if (period === 'month') {
      const start = new Date();
      start.setDate(start.getDate() - 29);
      return { start_date: start.toISOString().slice(0, 10), end_date: endDate, label: 'last 30d' };
    }
    return null;
  }

  function parseTimeWindowFromText(text = '') {
    const lower = text.toLowerCase();
    const today = new Date();
    const fmtDate = (d) => d.toISOString().slice(0, 10);
    const startEnd = (start, end, label) => ({ start_date: fmtDate(start), end_date: fmtDate(end), label });
    if (/\ball time\b/.test(lower)) return { start_date: null, end_date: null, label: 'all time' };
    if (lower.includes('yesterday')) {
      const d = new Date();
      d.setDate(today.getDate() - 1);
      const next = new Date(d);
      next.setDate(d.getDate() + 1);
      return startEnd(d, next, 'yesterday (24h)');
    }
    if (lower.includes('today')) {
      const d = new Date();
      const next = new Date(d);
      next.setDate(d.getDate() + 1);
      return startEnd(d, next, 'today (24h)');
    }
    if (/last\s+24|past\s+24/.test(lower)) {
      const start = new Date();
      start.setDate(today.getDate() - 1);
      return { start_date: start.toISOString(), end_date: new Date().toISOString(), label: 'last 24h' };
    }
    if (/last\s+7\s+days|past\s+7\s+days|last\s+week|previous\s+week/.test(lower)) {
      const start = new Date();
      start.setDate(today.getDate() - 7);
      const end = new Date();
      end.setDate(today.getDate() + 1);
      return startEnd(start, end, 'last 7d');
    }
    if (/\bthis\s+week\b/.test(lower)) {
      const start = new Date();
      start.setDate(today.getDate() - today.getDay());
      const end = new Date();
      end.setDate(today.getDate() + 1);
      return startEnd(start, end, 'this week');
    }
    if (/earlier\s+this\s+week/.test(lower)) {
      const start = new Date();
      start.setDate(today.getDate() - today.getDay());
      return startEnd(start, today, 'earlier this week');
    }
    if (/last\s+few\s+days|past\s+few\s+days/.test(lower) || /last\s+3\s+days|past\s+3\s+days/.test(lower)) {
      const start = new Date();
      start.setDate(today.getDate() - 3);
      const end = new Date();
      end.setDate(today.getDate() + 1);
      return startEnd(start, end, 'last 3d');
    }
    return { start_date: null, end_date: null, label: null };
  }

  function parseEmotionsFromText(text = '') {
    const lower = text.toLowerCase();
    const emos = new Set();
    if (lower.includes('negative')) ['anger', 'fear', 'sadness', 'disgust'].forEach((e) => emos.add(e));
    if (lower.includes('anger') || lower.includes('angry')) emos.add('anger');
    if (lower.includes('sad')) emos.add('sadness');
    if (lower.includes('fear') || lower.includes('afraid')) emos.add('fear');
    if (lower.includes('disgust')) emos.add('disgust');
    if (lower.includes('joy') || lower.includes('happy')) emos.add('joy');
    return Array.from(emos);
  }

  function parseTimeWindowFromText(text = '') {
    const lower = text.toLowerCase();
    const today = new Date();
    const fmtDate = (d) => d.toISOString().slice(0, 10);
    const startEnd = (start, end, label) => ({ start_date: fmtDate(start), end_date: fmtDate(end), label });
    if (/\ball time\b/.test(lower)) return { start_date: null, end_date: null, label: 'all time' };
    if (lower.includes('yesterday')) {
      const d = new Date();
      d.setDate(today.getDate() - 1);
      const next = new Date(d);
      next.setDate(d.getDate() + 1);
      return startEnd(d, next, 'yesterday (24h)');
    }
    if (lower.includes('today')) {
      const d = new Date();
      const next = new Date(d);
      next.setDate(d.getDate() + 1);
      return startEnd(d, next, 'today (24h)');
    }
    if (/last\s+24|past\s+24/.test(lower)) {
      const start = new Date();
      start.setDate(today.getDate() - 1);
      return { start_date: start.toISOString(), end_date: new Date().toISOString(), label: 'last 24h' };
    }
    if (/last\s+7\s+days|past\s+7\s+days|last\s+week|previous\s+week/.test(lower)) {
      const start = new Date();
      start.setDate(today.getDate() - 7);
      const end = new Date();
      end.setDate(today.getDate() + 1);
      return startEnd(start, end, 'last 7d');
    }
    if (/\bthis\s+week\b/.test(lower)) {
      const start = new Date();
      start.setDate(today.getDate() - today.getDay());
      const end = new Date();
      end.setDate(today.getDate() + 1);
      return startEnd(start, end, 'this week');
    }
    if (/earlier\s+this\s+week/.test(lower)) {
      const start = new Date();
      start.setDate(today.getDate() - today.getDay());
      return startEnd(start, today, 'earlier this week');
    }
    if (/last\s+few\s+days|past\s+few\s+days/.test(lower) || /last\s+3\s+days|past\s+3\s+days/.test(lower)) {
      const start = new Date();
      start.setDate(today.getDate() - 3);
      const end = new Date();
      end.setDate(today.getDate() + 1);
      return startEnd(start, end, 'last 3d');
    }
    return { start_date: null, end_date: null, label: null };
  }

  function parseEmotionsFromText(text = '') {
    const lower = text.toLowerCase();
    const emos = new Set();
    if (lower.includes('negative')) ['anger', 'fear', 'sadness', 'disgust'].forEach((e) => emos.add(e));
    if (lower.includes('anger') || lower.includes('angry')) emos.add('anger');
    if (lower.includes('sad')) emos.add('sadness');
    if (lower.includes('fear') || lower.includes('afraid')) emos.add('fear');
    if (lower.includes('disgust')) emos.add('disgust');
    if (lower.includes('joy') || lower.includes('happy')) emos.add('joy');
    return Array.from(emos);
  }

  function initRagCliShell({ mount, toggleButtonId, onToggle } = {}) {
    if (!apiClient) {
      console.error('[RAG CLI] API client unavailable');
      return null;
    }
    const root = document.querySelector(mount);
    if (!root) return null;

    const logEl = root.querySelector('#rag-cli-log');
    const inputEl = root.querySelector('#rag-cli-input');
    const runBtn = root.querySelector('#rag-cli-run');
    const clearBtn = root.querySelector('#rag-cli-clear');
    const helpBtn = root.querySelector('#rag-cli-help');
    const sessionChip = root.querySelector('#rag-cli-session-chip');
    const toggleBtn = toggleButtonId ? document.getElementById(toggleButtonId) : null;

    const state = {
      sessionId: uuid(),
      logs: [],
      history: [],
      historyIndex: -1,
      raw: true,
      lastRagResults: null,
      lastGemmaResponse: null,
      lastEmotionAnalyses: null,
      lastEmotionQuestion: null,
    };

    function setSessionChip() {
      if (sessionChip) {
        sessionChip.textContent = `Session: ${state.sessionId.slice(0, 8)}`;
      }
    }

    function appendLog(level, message, data) {
      if (!logEl) return;
      const entry = document.createElement('div');
      entry.className = `rag-cli-log-entry ${level}`;
      const tsSpan = document.createElement('span');
      tsSpan.className = 'ts';
      tsSpan.textContent = `[${nowTs()}]`;
      entry.appendChild(tsSpan);
      const msgSpan = document.createElement('span');
      msgSpan.textContent = ` ${message}`;
      entry.appendChild(msgSpan);
      if (state.raw && data !== undefined) {
        const pre = document.createElement('pre');
        pre.style.margin = '0.35rem 0 0';
        pre.style.whiteSpace = 'pre-wrap';
        pre.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
        entry.appendChild(pre);
      }
      logEl.appendChild(entry);
      logEl.scrollTop = logEl.scrollHeight;
      if (state.logs.length > 400) {
        state.logs.shift();
        if (logEl.firstChild) logEl.removeChild(logEl.firstChild);
      }
    }

    function clearLog() {
      state.logs = [];
      if (logEl) logEl.innerHTML = '';
    }

    function setRaw(enabled) {
      state.raw = enabled;
      appendLog('info', `Raw JSON ${enabled ? 'enabled' : 'disabled'}`);
    }

    async function runWithTiming(fn, label) {
      const start = performance.now();
      try {
        const result = await fn();
        const elapsed = performance.now() - start;
        return { result, elapsed };
      } catch (err) {
        const elapsed = performance.now() - start;
        throw { err, elapsed };
      }
    }

    function parseNumericFlag(tokens, name, defaultValue) {
      const idx = tokens.findIndex((t) => t === name);
      if (idx >= 0 && tokens[idx + 1] !== undefined) {
        const maybe = Number(tokens[idx + 1]);
        if (Number.isFinite(maybe) && maybe > 0) return maybe;
      }
      return defaultValue;
    }

    async function handleRag(tokens) {
      if (!tokens.length) {
        appendLog('warn', 'Usage: rag query "text" [--top-k N] [--last-n M]');
        return;
      }
      const sub = tokens[0];
      if (sub !== 'query') {
        appendLog('warn', 'Unknown rag command; try: rag query "text"');
        return;
      }
      const topK = parseNumericFlag(tokens, '--top-k', 8);
      const lastN = parseNumericFlag(tokens, '--last-n', 50);
      const speakers = parseSpeakers(tokens);
      const period = parsePeriod(tokens);
      const query = buildQueryFromTokens(tokens, 1).replace(/--top-k\s+\d+|--last-n\s+\d+/g, '').trim();
      if (!query) {
        appendLog('warn', 'Please provide a query text.');
        return;
      }
      appendLog('info', `RAG query → "${query}" (top_k=${topK}, last_n=${lastN}, speakers=${speakers.length ? speakers.join(',') : 'all'})${period ? ` • ${period.label}` : ''}`);
      try {
        const { result, elapsed } = await runWithTiming(
          () => apiClient.post('/search/semantic', {
            query,
            top_k: topK,
            last_n_transcripts: lastN,
            speakers: speakers.length ? speakers : undefined,
            start_date: period?.start_date,
            end_date: period?.end_date,
          }),
          'rag'
        );
        const results = Array.isArray(result?.results) ? result.results : [];
        state.lastRagResults = results;
        appendLog('success', `[RAG] ${summarizeResults(results)} in ${formatMs(elapsed)}`, state.raw ? result : undefined);
      } catch (e) {
        const err = e.err || e;
        const elapsed = e.elapsed || null;
        appendLog('error', `[RAG] failed${elapsed ? ` after ${formatMs(elapsed)}` : ''}: ${err?.message || err}`);
      }
    }

    async function handleGemma(tokens) {
      if (!tokens.length) {
        appendLog('warn', 'Usage: gemma rag "question" | gemma chat "question"');
        return;
      }
      const mode = tokens[0];
      const question = buildQueryFromTokens(tokens, 1);
      if (!question) {
        appendLog('warn', 'Please provide a question.');
        return;
      }
      const topK = parseNumericFlag(tokens, '--top-k', 8);
      const maxTokens = parseNumericFlag(tokens, '--max-tokens', 512);
      const speakers = parseSpeakers(tokens);
      const period = parsePeriod(tokens);
      if (mode === 'rag') {
        const tw = period || parseTimeWindowFromText(question);
        appendLog('info', `Gemma RAG → "${question}" (top_k=${topK}, max_tokens=${maxTokens}, speakers=${speakers.length ? speakers.join(',') : 'all'})${tw?.label ? ` • ${tw.label}` : ''}`);
        // Preview context via semantic search for transparency before sending to Gemma.
        try {
          const previewPayload = {
            query: question,
            top_k: Math.max(3, Math.min(topK, 10)),
            last_n_transcripts: 1000,
            speakers: speakers.length ? speakers : DEFAULT_SPEAKER_FOCUS,
            start_date: tw?.start_date || undefined,
            end_date: tw?.end_date || undefined,
          };
          const { result: preview, elapsed: previewElapsed } = await runWithTiming(
            () => apiClient.post('/search/semantic', previewPayload),
            'gemma-rag-preview'
          );
          const results = Array.isArray(preview?.results) ? preview.results : [];
          const ctxLines = results.slice(0, 3).map((r, idx) => {
            const md = r.metadata || {};
            const speaker = md.speaker || 'unknown';
            const txt = (r.text || '').slice(0, 200);
            const ts = typeof md.start_time === 'number' ? `${Math.floor(md.start_time / 60)}:${String(Math.floor(md.start_time % 60)).padStart(2, '0')}` : '--:--';
            return `[C${idx + 1}] ${speaker} ${ts} :: ${txt}`;
          });
          appendLog('info', `[Gemma RAG] preview context (${formatMs(previewElapsed)}):\n${ctxLines.join('\n') || '[none]'}`, state.raw ? preview : undefined);
        } catch (e) {
          const err = e.err || e;
          appendLog('error', `[Gemma RAG] preview context failed: ${err?.message || err}`);
        }
        try {
          const payload = {
            query: question,
            top_k_results: topK,
            max_tokens: maxTokens,
            session_id: state.sessionId,
            speakers: speakers.length ? speakers : undefined,
            start_date: tw?.start_date,
            end_date: tw?.end_date,
          };
          const { result, elapsed } = await runWithTiming(
            () => apiClient.post('/gemma/chat-rag', payload),
            'gemma-rag'
          );
          state.lastGemmaResponse = result;
          const summary = `[GEMMA RAG] ${result?.tokens_generated ?? 0} tokens • ${result?.mode || 'n/a'} • conf ${result?.confidence ?? 'n/a'}`;
          appendLog('success', `${summary} in ${formatMs(elapsed)}`, state.raw ? { payload, result } : undefined);
          appendLog('info', `[Gemma RAG] payload: ${JSON.stringify(payload)}`);
          const answer = result?.answer || result?.response || '[empty]';
          appendLog('info', `[Gemma RAG] answer: ${answer}`);
        } catch (e) {
          const err = e.err || e;
          const elapsed = e.elapsed || null;
          appendLog('error', `[GEMMA RAG] failed${elapsed ? ` after ${formatMs(elapsed)}` : ''}: ${err?.message || err}`);
        }
        return;
      }
      if (mode === 'chat') {
        appendLog('info', `Gemma chat → "${question}" (max_tokens=${maxTokens})`);
        try {
          const payload = {
            messages: [{ role: 'user', content: question }],
            max_tokens: maxTokens,
            temperature: 0.7,
          };
          const { result, elapsed } = await runWithTiming(
            () => apiClient.post('/gemma/chat', payload),
            'gemma-chat'
          );
          state.lastGemmaResponse = result;
          appendLog(
            'success',
            `[GEMMA chat] ${result?.tokens_generated ?? result?.usage?.completion_tokens ?? 'n/a'} tokens • ${result?.mode || 'n/a'} in ${formatMs(elapsed)}`,
            state.raw ? result : undefined
          );
        } catch (e) {
          const err = e.err || e;
          const elapsed = e.elapsed || null;
          appendLog('error', `[GEMMA chat] failed${elapsed ? ` after ${formatMs(elapsed)}` : ''}: ${err?.message || err}`);
        }
        return;
      }
      appendLog('warn', 'Unknown gemma command; try: gemma rag "question"');
    }

    function extractContextLines(item) {
      const lines = [];
      const before = Array.isArray(item.context_before) ? item.context_before : [];
      const after = Array.isArray(item.context_after) ? item.context_after : [];
      const fmt = (seg) => `${seg.speaker || (seg.metadata || {}).speaker || '?'}: ${seg.text || seg.content || ''}`;
      before.slice(-10).forEach((seg) => lines.push(fmt(seg)));
      lines.push(fmt(item));
      after.slice(0, 5).forEach((seg) => lines.push(fmt(seg)));
      return lines;
    }

    function parseEmotionConfidence(item) {
      const md = item.metadata || {};
      return item.emotion_confidence ?? md.emotion_confidence ?? null;
    }

    async function handleEmotion(tokens) {
      if (!tokens.length) {
        appendLog('warn', 'Usage: emotion analyze "<query>" | emotion summary');
        return;
      }
      const sub = tokens[0];
      if (sub === 'analyze') {
        const question = buildQueryFromTokens(tokens, 1);
        if (!question) {
          appendLog('warn', 'Provide a query for emotion analyze.');
          return;
        }
        const speakers = parseSpeakers(tokens);
        const periodFlag = parsePeriod(tokens);
        const tw = periodFlag || parseTimeWindowFromText(question);
        const emotions = parseEmotionsFromText(question);
        appendLog('info', `[Emotion] analyze → "${question}" (speakers=${speakers.length ? speakers.join(',') : 'all'}, emotions=${emotions.join(',') || 'any'}${tw?.label ? `, ${tw.label}` : ''}, conf>=${EMO_CONF_THRESH})`);
        let items = [];
        try {
          const payload = {
            keywords: question,
            search_type: 'keyword',
            limit: 5,
            context_lines: 10,
            speakers: speakers.length ? speakers : DEFAULT_SPEAKER_FOCUS,
            start_date: tw?.start_date || null,
            end_date: tw?.end_date || null,
            emotions: emotions.length ? emotions : undefined,
          };
          const { result, elapsed } = await runWithTiming(() => apiClient.post('/transcripts/query', payload), 'emotion-query');
          items = Array.isArray(result?.items) ? result.items : [];
          appendLog('info', `[Emotion] fetched ${items.length} segments in ${formatMs(elapsed)}`, state.raw ? items : undefined);
        } catch (e) {
          const err = e.err || e;
          appendLog('error', `[Emotion] query failed: ${err?.message || err}`);
          return;
        }
        if (!items.length) {
          appendLog('warn', '[Emotion] no segments returned for this filter.');
          return;
        }
        const above = items.filter((it) => {
          const conf = parseEmotionConfidence(it);
          return conf !== null && conf >= EMO_CONF_THRESH;
        });
        const selected = above.length ? above.slice(0, 5) : items.slice(0, 5);
        if (!above.length) {
          const maxConf = Math.max(...items.map((it) => parseEmotionConfidence(it) || 0));
          appendLog('warn', `[Emotion] no segments >= ${EMO_CONF_THRESH}; falling back to top by confidence (max=${maxConf.toFixed(2)})`);
        }
        const results = [];
        for (let idx = 0; idx < selected.length; idx += 1) {
          const item = selected[idx];
          const segId = item.segment_id || (item.metadata || {}).segment_id || 'unknown';
          const speaker = item.speaker || (item.metadata || {}).speaker || 'unknown';
          const emotion = item.emotion || (item.metadata || {}).emotion || 'unknown';
          const conf = parseEmotionConfidence(item);
          const contextLines = extractContextLines(item).join('\n');
          const prompt = [
            'You are analyzing emotional transcripts. Return ONLY JSON with keys: explanation, confidence_0_100 (0-100 integer), segment_id, context_segment_ids (list).',
            `Emotion: ${emotion}`,
            `Segment ID: ${segId}`,
            `Speaker: ${speaker}`,
            `Segment text: ${item.text || item.content || ''}`,
            'Context (ordered):',
            contextLines,
          ].join('\n');
          try {
            const payload = { messages: [{ role: 'user', content: prompt }], max_tokens: 200, temperature: 0.4 };
            const { result, elapsed } = await runWithTiming(() => apiClient.post('/gemma/chat', payload), 'emotion-explain');
            const answer = result?.answer || result?.response || result?.message || '';
            let parsed = null;
            try {
              parsed = JSON.parse(answer);
            } catch {
              parsed = null;
            }
            results.push({ segId, speaker, emotion, conf, raw: answer, parsed });
            const exp = parsed?.explanation || answer;
            appendLog('success', `[Emotion] seg=${segId} speaker=${speaker} emotion=${emotion} conf=${conf ?? 'n/a'} :: ${exp?.slice(0, 200) || ''} (${formatMs(elapsed)})`, state.raw ? result : undefined);
          } catch (e) {
            const err = e.err || e;
            appendLog('error', `[Emotion] explain failed for seg=${segId}: ${err?.message || err}`);
          }
        }
        state.lastEmotionAnalyses = results;
        state.lastEmotionQuestion = question;
        appendLog('info', `[Emotion] analyzed ${results.length} segments.`);
        return;
      }
      if (sub === 'summary') {
        if (!state.lastEmotionAnalyses || !state.lastEmotionAnalyses.length) {
          appendLog('warn', 'No emotion analyses available. Run emotion analyze first.');
          return;
        }
        const lines = state.lastEmotionAnalyses.slice(0, 10).map((r) => {
          const summary = r.parsed?.explanation || r.raw || '';
          return `seg=${r.segId} speaker=${r.speaker} emotion=${r.emotion} conf=${r.conf ?? 'n/a'} :: ${summary.replace(/\n/g, ' ')}`;
        });
        const prompt = [
          'You are summarizing an emotion analysis set. Each line is one segment analysis with speaker, emotion, and explanation.',
          'Find common themes and concise conclusions about why negative emotions occur. If data is thin, say so.',
          '',
          'Entries:',
          lines.join('\n') || 'No entries found.',
          '',
          'Respond with a short summary of common causes/themes.',
        ].join('\n');
        try {
          const payload = { messages: [{ role: 'user', content: prompt }], max_tokens: 200, temperature: 0.4 };
          const { result, elapsed } = await runWithTiming(() => apiClient.post('/gemma/chat', payload), 'emotion-summary');
          const ans = result?.answer || result?.response || result?.message || '';
          appendLog('success', `[Emotion] summary (${formatMs(elapsed)}): ${ans}`, state.raw ? result : undefined);
        } catch (e) {
          const err = e.err || e;
          appendLog('error', `[Emotion] summary failed: ${err?.message || err}`);
        }
        return;
      }
      appendLog('warn', 'Unknown emotion command; try: emotion analyze "<text>" | emotion summary');
    }

    async function handleGpu() {
      try {
        const { result, elapsed } = await runWithTiming(() => apiClient.get('/gemma/stats'), 'gpu');
        appendLog(
          'info',
          `[GPU] model_on_gpu=${result?.model_on_gpu ? 'yes' : 'no'} • ctx=${result?.context_size || 'n/a'} • gpu_layers=${result?.gpu_layers || 'n/a'} in ${formatMs(elapsed)}`,
          state.raw ? result : undefined
        );
      } catch (e) {
        appendLog('error', `[GPU] failed: ${e?.err?.message || e?.message || e}`);
      }
    }

    async function handleWarmup() {
      try {
        const { result, elapsed } = await runWithTiming(() => apiClient.post('/gemma/warmup', {}), 'warmup');
        appendLog('info', `[Warmup] ${result?.status || 'ok'} • model_on_gpu=${result?.model_on_gpu} in ${formatMs(elapsed)}`, state.raw ? result : undefined);
      } catch (e) {
        appendLog('error', `[Warmup] failed: ${e?.err?.message || e?.message || e}`);
      }
    }

    function showContext() {
      if (!state.lastRagResults || !state.lastRagResults.length) {
        appendLog('warn', 'No RAG context available. Run a RAG query first.');
        return;
      }
      const lines = state.lastRagResults.slice(0, 8).map((r, idx) => {
        const md = r.metadata || {};
        const speaker = md.speaker || 'unknown';
        const ts = typeof md.start_time === 'number' ? `${Math.floor(md.start_time / 60)}:${String(Math.floor(md.start_time % 60)).padStart(2, '0')}` : '--:--';
        return `[${idx + 1}] ${ts} • ${speaker} • ${r.text?.slice(0, 140) || ''}`;
      });
      appendLog('info', lines.join('\n'));
    }

    function showResponse() {
      if (!state.lastGemmaResponse) {
        appendLog('warn', 'No Gemma response available. Run gemma rag/chat first.');
        return;
      }
      const answer = state.lastGemmaResponse.answer || state.lastGemmaResponse.response || state.lastGemmaResponse.text;
      appendLog('info', answer || '[empty]', state.raw ? state.lastGemmaResponse : undefined);
    }

    function printHelp() {
      const helpText = [
        'Commands:',
        '  help                    Show this help',
        '  clear                   Clear log',
        '  raw on|off              Toggle raw JSON logging (default: on)',
        '',
        '=== Core Commands ===',
        '  rag query "<text>" [--top-k N] [--last-n M] [--period today|week|month]',
        '  gemma rag "<question>" [--top-k N] [--max-tokens N]',
        '  gemma chat "<question>" [--max-tokens N]',
        '  emotion analyze "<text>"   Analyze emotion segments',
        '  emotion summary            Summarize last emotion analysis',
        '  show context               Show last RAG context',
        '  show response              Show last Gemma response',
        '  gpu                        Show Gemma GPU stats',
        '  warmup                     Warm GPU (Gemma)',
        '',
        '=== ML Profit Commands ===',
        '  ml test [member_id]        Run all ML endpoint tests',
        '  ml crosssell <member_id>   Test cross-sell propensity',
        '  ml churn <member_id>       Test churn prediction',
        '  ml pricing <member_id> <loan_type> <amount> <term> <score>',
        '                             Test loan pricing optimization',
        '  ml features <member_id>    View member ML features',
        '',
        '=== Enterprise Commands ===',
        '  qa stats                   Show QA statistics',
        '  qa review [pending|all]    List feedback for review',
        '  qa golden list             List golden answers',
        '',
        '  automation stats           Show automation statistics',
        '  automation rules           List automation rules',
        '  automation webhooks        List webhooks',
        '',
        '  knowledge stats            Show knowledge stats',
        '  knowledge articles         List recent articles',
        '  knowledge search "<query>" Search knowledge base',
        '',
        '  analytics stats            Show analytics stats',
        '  analytics reports          List saved reports',
        '',
        '  meetings stats             Show meeting stats',
        '  meetings list              List recent meetings',
        '  meetings search "<query>"  Search meetings',
        '',
        '=== Banking Hub Commands (SCU) ===',
        '  banking analyze <member_id>   Analyze member with role-based insights',
        '  banking dataset <member_id>   Fetch raw dataset from Fiserv connector',
        '  banking test                  Run full integration test',
      ].join('\n');
      appendLog('info', helpText);
    }

    // ===========================================
    // Enterprise Command Handlers
    // ===========================================

    async function handleQA(tokens) {
      const sub = tokens[0] || 'stats';
      if (sub === 'stats') {
        try {
          const { result, elapsed } = await runWithTiming(() => apiClient.get('/enterprise/qa/stats'), 'qa-stats');
          appendLog('success', `[QA] Stats in ${formatMs(elapsed)}`, result);
        } catch (e) {
          appendLog('error', `[QA] stats failed: ${e?.message || e}`);
        }
        return;
      }
      if (sub === 'review') {
        const status = tokens[1] || 'pending';
        try {
          const { result, elapsed } = await runWithTiming(
            () => apiClient.get(`/enterprise/qa/review?status=${status}&limit=10`),
            'qa-review'
          );
          const items = result?.items || result || [];
          appendLog('success', `[QA] Review queue (${status}): ${items.length} items in ${formatMs(elapsed)}`, items);
        } catch (e) {
          appendLog('error', `[QA] review failed: ${e?.message || e}`);
        }
        return;
      }
      if (sub === 'golden') {
        const action = tokens[1] || 'list';
        if (action === 'list') {
          try {
            const { result, elapsed } = await runWithTiming(
              () => apiClient.get('/enterprise/qa/golden?limit=20'),
              'qa-golden'
            );
            const items = result?.items || result || [];
            appendLog('success', `[QA] Golden answers: ${items.length} in ${formatMs(elapsed)}`, items);
          } catch (e) {
            appendLog('error', `[QA] golden list failed: ${e?.message || e}`);
          }
        }
        return;
      }
      appendLog('warn', 'Usage: qa stats | qa review [pending|all] | qa golden list');
    }

    async function handleAutomation(tokens) {
      const sub = tokens[0] || 'stats';
      if (sub === 'stats') {
        try {
          const { result, elapsed } = await runWithTiming(() => apiClient.get('/enterprise/automation/stats'), 'auto-stats');
          appendLog('success', `[Automation] Stats in ${formatMs(elapsed)}`, result);
        } catch (e) {
          appendLog('error', `[Automation] stats failed: ${e?.message || e}`);
        }
        return;
      }
      if (sub === 'rules') {
        try {
          const { result, elapsed } = await runWithTiming(() => apiClient.get('/enterprise/automation/rules'), 'auto-rules');
          const items = result?.rules || [];
          appendLog('success', `[Automation] Rules: ${items.length} in ${formatMs(elapsed)}`, items);
        } catch (e) {
          appendLog('error', `[Automation] rules failed: ${e?.message || e}`);
        }
        return;
      }
      if (sub === 'webhooks') {
        try {
          const { result, elapsed } = await runWithTiming(() => apiClient.get('/enterprise/automation/webhooks'), 'auto-webhooks');
          const items = result?.webhooks || [];
          appendLog('success', `[Automation] Webhooks: ${items.length} in ${formatMs(elapsed)}`, items);
        } catch (e) {
          appendLog('error', `[Automation] webhooks failed: ${e?.message || e}`);
        }
        return;
      }
      appendLog('warn', 'Usage: automation stats | automation rules | automation webhooks');
    }

    async function handleKnowledge(tokens) {
      const sub = tokens[0] || 'stats';
      if (sub === 'stats') {
        try {
          const { result, elapsed } = await runWithTiming(() => apiClient.get('/enterprise/knowledge/stats'), 'kb-stats');
          appendLog('success', `[Knowledge] Stats in ${formatMs(elapsed)}`, result);
        } catch (e) {
          appendLog('error', `[Knowledge] stats failed: ${e?.message || e}`);
        }
        return;
      }
      if (sub === 'articles') {
        try {
          const { result, elapsed } = await runWithTiming(() => apiClient.get('/enterprise/knowledge/articles?limit=10'), 'kb-articles');
          const items = result?.articles || result?.items || [];
          appendLog('success', `[Knowledge] Articles: ${items.length} in ${formatMs(elapsed)}`, items);
        } catch (e) {
          appendLog('error', `[Knowledge] articles failed: ${e?.message || e}`);
        }
        return;
      }
      if (sub === 'search') {
        const query = buildQueryFromTokens(tokens, 1);
        if (!query) {
          appendLog('warn', 'Usage: knowledge search "<query>"');
          return;
        }
        try {
          const { result, elapsed } = await runWithTiming(
            () => apiClient.get(`/enterprise/knowledge/search?query=${encodeURIComponent(query)}`),
            'kb-search'
          );
          appendLog('success', `[Knowledge] Search results in ${formatMs(elapsed)}`, result);
        } catch (e) {
          appendLog('error', `[Knowledge] search failed: ${e?.message || e}`);
        }
        return;
      }
      appendLog('warn', 'Usage: knowledge stats | knowledge articles | knowledge search "<query>"');
    }

    async function handleAnalytics(tokens) {
      const sub = tokens[0] || 'stats';
      if (sub === 'stats') {
        try {
          const { result, elapsed } = await runWithTiming(() => apiClient.get('/enterprise/analytics/stats'), 'analytics-stats');
          appendLog('success', `[Analytics] Stats in ${formatMs(elapsed)}`, result);
        } catch (e) {
          appendLog('error', `[Analytics] stats failed: ${e?.message || e}`);
        }
        return;
      }
      if (sub === 'reports') {
        try {
          const { result, elapsed } = await runWithTiming(() => apiClient.get('/enterprise/analytics/reports?limit=10'), 'analytics-reports');
          const items = result?.reports || result?.items || [];
          appendLog('success', `[Analytics] Reports: ${items.length} in ${formatMs(elapsed)}`, items);
        } catch (e) {
          appendLog('error', `[Analytics] reports failed: ${e?.message || e}`);
        }
        return;
      }
      appendLog('warn', 'Usage: analytics stats | analytics reports');
    }

    async function handleMeetings(tokens) {
      const sub = tokens[0] || 'stats';
      if (sub === 'stats') {
        try {
          const { result, elapsed } = await runWithTiming(() => apiClient.get('/enterprise/meetings/stats'), 'meetings-stats');
          appendLog('success', `[Meetings] Stats in ${formatMs(elapsed)}`, result);
        } catch (e) {
          appendLog('error', `[Meetings] stats failed: ${e?.message || e}`);
        }
        return;
      }
      if (sub === 'list') {
        try {
          const { result, elapsed } = await runWithTiming(() => apiClient.get('/enterprise/meetings?limit=10'), 'meetings-list');
          const items = result?.meetings || result?.items || [];
          appendLog('success', `[Meetings] List: ${items.length} in ${formatMs(elapsed)}`, items);
        } catch (e) {
          appendLog('error', `[Meetings] list failed: ${e?.message || e}`);
        }
        return;
      }
      if (sub === 'search') {
        const query = buildQueryFromTokens(tokens, 1);
        if (!query) {
          appendLog('warn', 'Usage: meetings search "<query>"');
          return;
        }
        try {
          const { result, elapsed } = await runWithTiming(
            () => apiClient.get(`/enterprise/meetings/search?query=${encodeURIComponent(query)}`),
            'meetings-search'
          );
          appendLog('success', `[Meetings] Search results in ${formatMs(elapsed)}`, result);
        } catch (e) {
          appendLog('error', `[Meetings] search failed: ${e?.message || e}`);
        }
        return;
      }
      appendLog('warn', 'Usage: meetings stats | meetings list | meetings search "<query>"');
    }

    // ===========================================
    // ML Profit Maximization Command Handlers
    // ===========================================

    async function handleML(tokens) {
      const sub = tokens[0] || 'test';
      const memberId = tokens[1] || 'M12345';

      // Helper to make ML API calls through gateway
      async function mlFetch(method, path, body = null) {
        const url = `/fiserv/api/v1/ml${path}`;
        const options = {
          method,
          headers: { 'Content-Type': 'application/json' },
        };
        if (body) options.body = JSON.stringify(body);
        const res = await fetch(url, options);
        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`${res.status}: ${errText}`);
        }
        return res.json();
      }

      if (sub === 'test') {
        appendLog('info', `[ML] Running all tests for member: ${memberId}`);
        let passed = 0;
        let failed = 0;

        // Test Cross-Sell
        try {
          const { result, elapsed } = await runWithTiming(
            () => mlFetch('POST', '/cross-sell/predict', { member_id: memberId }),
            'ml-crosssell'
          );
          const recCount = result?.recommendations?.length || 0;
          appendLog('success', `[ML] ✓ Cross-sell: ${recCount} recommendations in ${formatMs(elapsed)}`, state.raw ? result : undefined);
          passed++;
        } catch (e) {
          appendLog('error', `[ML] ✗ Cross-sell failed: ${e.message}`);
          failed++;
        }

        // Test Churn
        try {
          const { result, elapsed } = await runWithTiming(
            () => mlFetch('POST', '/churn/predict', { member_id: memberId }),
            'ml-churn'
          );
          appendLog('success', `[ML] ✓ Churn: ${result?.risk_level || 'N/A'} risk (${(result?.churn_probability * 100).toFixed(1)}%) in ${formatMs(elapsed)}`, state.raw ? result : undefined);
          passed++;
        } catch (e) {
          appendLog('error', `[ML] ✗ Churn failed: ${e.message}`);
          failed++;
        }

        // Test Pricing
        try {
          const { result, elapsed } = await runWithTiming(
            () => mlFetch('POST', '/pricing/optimize', {
              member_id: memberId,
              loan_type: 'auto_new',
              amount: 25000,
              term_months: 60,
              credit_score: 720
            }),
            'ml-pricing'
          );
          appendLog('success', `[ML] ✓ Pricing: ${result?.recommended_rate}% rate in ${formatMs(elapsed)}`, state.raw ? result : undefined);
          passed++;
        } catch (e) {
          appendLog('error', `[ML] ✗ Pricing failed: ${e.message}`);
          failed++;
        }

        appendLog('info', `[ML] Test Summary: ${passed} passed, ${failed} failed`);
        return;
      }

      if (sub === 'crosssell') {
        try {
          const { result, elapsed } = await runWithTiming(
            () => mlFetch('POST', '/cross-sell/predict', { member_id: memberId, top_n: 5 }),
            'ml-crosssell'
          );
          const recs = result?.recommendations || [];
          if (recs.length === 0) {
            appendLog('warn', `[ML Cross-Sell] No recommendations for ${memberId} (may already have all products)`);
          } else {
            recs.forEach((r, idx) => {
              appendLog('success', `[${idx + 1}] ${r.product_name}: ${(r.propensity_score * 100).toFixed(0)}% (${r.confidence})`);
              if (r.reasons?.length) {
                r.reasons.forEach(reason => appendLog('info', `    → ${reason}`));
              }
            });
          }
          appendLog('info', `[ML Cross-Sell] Completed in ${formatMs(elapsed)}`, state.raw ? result : undefined);
        } catch (e) {
          appendLog('error', `[ML Cross-Sell] Failed: ${e.message}`);
        }
        return;
      }

      if (sub === 'churn') {
        try {
          const { result, elapsed } = await runWithTiming(
            () => mlFetch('POST', '/churn/predict', { member_id: memberId }),
            'ml-churn'
          );
          const prob = (result?.churn_probability * 100).toFixed(1);
          const level = result?.risk_level || 'N/A';
          const priority = result?.retention_priority || 0;
          appendLog('success', `[ML Churn] Risk: ${level} (${prob}%) | Priority: ${priority}/100`);

          if (result?.top_risk_factors?.length) {
            appendLog('info', '[ML Churn] Risk Factors:');
            result.top_risk_factors.forEach(f => {
              appendLog('warn', `    ⚠ ${f.description} (${f.severity})`);
            });
          }

          if (result?.recommended_actions?.length) {
            appendLog('info', '[ML Churn] Recommended Actions:');
            result.recommended_actions.forEach((a, idx) => {
              appendLog('info', `    ${idx + 1}. ${a}`);
            });
          }
          appendLog('info', `[ML Churn] Completed in ${formatMs(elapsed)}`, state.raw ? result : undefined);
        } catch (e) {
          appendLog('error', `[ML Churn] Failed: ${e.message}`);
        }
        return;
      }

      if (sub === 'pricing') {
        const loanType = tokens[2] || 'auto_new';
        const amount = parseFloat(tokens[3]) || 25000;
        const term = parseInt(tokens[4]) || 60;
        const score = parseInt(tokens[5]) || 720;

        try {
          const { result, elapsed } = await runWithTiming(
            () => mlFetch('POST', '/pricing/optimize', {
              member_id: memberId,
              loan_type: loanType,
              amount: amount,
              term_months: term,
              credit_score: score
            }),
            'ml-pricing'
          );

          appendLog('success', `[ML Pricing] Recommended Rate: ${result?.recommended_rate}%`);
          appendLog('info', `    Loan: $${amount.toLocaleString()} ${loanType} @ ${term} months`);
          appendLog('info', `    Risk Tier: ${result?.risk_tier} (Score: ${score})`);
          appendLog('info', `    Range: ${result?.rate_range?.floor}% - ${result?.rate_range?.ceiling}%`);
          appendLog('info', `    Monthly Payment: $${result?.monthly_payment}`);
          appendLog('info', `    Total Interest: $${result?.total_interest}`);
          appendLog('info', `    Win Probability: ${(result?.win_probability * 100).toFixed(0)}%`);

          if (result?.rate_breakdown?.relationship_adjustments?.length) {
            appendLog('info', '[ML Pricing] Adjustments:');
            result.rate_breakdown.relationship_adjustments.forEach(adj => {
              const sign = adj.amount > 0 ? '+' : '';
              appendLog('info', `    ${sign}${adj.amount}%: ${adj.name}`);
            });
          }

          const rec = result?.approval_recommendation;
          if (rec) {
            appendLog(rec.recommendation === 'APPROVE' ? 'success' : 'warn',
              `[ML Pricing] ${rec.recommendation} (${rec.confidence}): ${rec.notes}`);
          }
          appendLog('info', `[ML Pricing] Completed in ${formatMs(elapsed)}`, state.raw ? result : undefined);
        } catch (e) {
          appendLog('error', `[ML Pricing] Failed: ${e.message}`);
        }
        return;
      }

      if (sub === 'features') {
        try {
          const { result, elapsed } = await runWithTiming(
            () => mlFetch('GET', `/features/${memberId}`),
            'ml-features'
          );
          appendLog('success', `[ML Features] Retrieved ${Object.keys(result).length} features in ${formatMs(elapsed)}`);

          // Display key features in organized groups
          const demo = ['age', 'age_bracket', 'tenure_years', 'tenure_months'];
          const products = ['has_checking', 'has_savings', 'has_auto_loan', 'has_credit_card', 'has_heloc', 'has_mortgage', 'product_count'];
          const balances = ['total_balance', 'avg_balance'];
          const activity = ['transaction_count_30d', 'has_direct_deposit', 'days_since_last_tx'];

          appendLog('info', '[Demographics]');
          demo.forEach(k => result[k] !== undefined && appendLog('info', `    ${k}: ${result[k]}`));

          appendLog('info', '[Products]');
          products.forEach(k => result[k] !== undefined && appendLog('info', `    ${k}: ${result[k]}`));

          appendLog('info', '[Balances]');
          balances.forEach(k => result[k] !== undefined && appendLog('info', `    ${k}: $${Number(result[k]).toLocaleString()}`));

          appendLog('info', '[Activity]');
          activity.forEach(k => result[k] !== undefined && appendLog('info', `    ${k}: ${result[k]}`));

          if (state.raw) appendLog('info', '', result);
        } catch (e) {
          appendLog('error', `[ML Features] Failed: ${e.message}`);
        }
        return;
      }

      appendLog('warn', 'Usage: ml test | ml crosssell <id> | ml churn <id> | ml pricing <id> <type> <amt> <term> <score> | ml features <id>');
    }

    // ===========================================
    // Banking Hub Command Handler (Phase 1: SCU Integration)
    // ===========================================

    async function handleBanking(tokens) {
      const sub = tokens[0] || 'analyze';
      const memberId = tokens[1] || '12345';

      if (sub === 'analyze') {
        appendLog('info', `[Banking] Analyzing member ${memberId}...`);
        try {
          const { result, elapsed } = await runWithTiming(
            () => apiClient.post(`/api/v1/banking/analyze/${memberId}`, {}),
            'banking-analyze'
          );

          appendLog('success', `[Banking] Analysis complete in ${formatMs(elapsed)}`);
          appendLog('info', `[Banking] Member: ${result?.member_id} | Role: ${result?.role_context}`);

          const insights = result?.insights || {};

          // Display Stats if available
          if (insights.stats && !insights.stats.error) {
            appendLog('success', `[Banking] ✓ Statistical Summary loaded`);
            if (state.raw) appendLog('info', '', insights.stats);
          } else if (insights.stats?.error) {
            appendLog('warn', `[Banking] ⚠ Statistics: ${insights.stats.error}`);
          }

          // Display Cash Flow if available
          if (insights.cash_flow && !insights.cash_flow.error) {
            appendLog('success', `[Banking] ✓ Cash Flow analysis loaded`);
            if (state.raw) appendLog('info', '', insights.cash_flow);
          } else if (insights.cash_flow?.error) {
            appendLog('warn', `[Banking] ⚠ Cash Flow: ${insights.cash_flow.error}`);
          }

          // Display Spend Pattern if available (Loan Officer role)
          if (insights.spend_pattern && !insights.spend_pattern.error) {
            appendLog('success', `[Banking] ✓ Spend Pattern analysis loaded`);
            if (state.raw) appendLog('info', '', insights.spend_pattern);
          }

          // Display LTV if available (Loan Officer role)
          if (insights.ltv && !insights.ltv.error) {
            appendLog('success', `[Banking] ✓ Customer LTV calculated`);
            if (state.raw) appendLog('info', '', insights.ltv);
          }

          // Display Anomalies if available (Fraud Analyst role)
          if (insights.anomalies && !insights.anomalies.error) {
            appendLog('success', `[Banking] ✓ Anomaly Detection complete`);
            if (state.raw) appendLog('info', '', insights.anomalies);
          }

          appendLog('info', `[Banking] Full response:`, state.raw ? result : undefined);
        } catch (e) {
          const err = e.err || e;
          appendLog('error', `[Banking] Analysis failed: ${err?.message || err}`);
        }
        return;
      }

      if (sub === 'dataset') {
        appendLog('info', `[Banking] Fetching dataset for member ${memberId}...`);
        try {
          // Call Fiserv service directly for dataset view
          const { result, elapsed } = await runWithTiming(
            () => fetch(`/fiserv/api/v1/datasets/member/${memberId}`).then(r => r.json()),
            'banking-dataset'
          );
          appendLog('success', `[Banking] Dataset fetched in ${formatMs(elapsed)}`);
          appendLog('info', `[Banking] Accounts: ${result?.accounts?.length || 0}`);
          appendLog('info', `[Banking] Transactions: ${result?.transactions?.length || 0}`);
          if (state.raw) appendLog('info', '', result);
        } catch (e) {
          const err = e.err || e;
          appendLog('error', `[Banking] Dataset fetch failed: ${err?.message || err}`);
        }
        return;
      }

      if (sub === 'test') {
        appendLog('info', `[Banking] Running full integration test...`);
        let passed = 0, failed = 0;

        // Test 1: Dataset endpoint
        try {
          const { result, elapsed } = await runWithTiming(
            () => fetch(`/fiserv/api/v1/datasets/member/${memberId}`).then(r => r.json()),
            'banking-test-dataset'
          );
          if (result?.member?.member_id) {
            appendLog('success', `[Banking] ✓ Dataset endpoint OK (${formatMs(elapsed)})`);
            passed++;
          } else {
            throw new Error('Invalid dataset response');
          }
        } catch (e) {
          appendLog('error', `[Banking] ✗ Dataset endpoint failed: ${e.message}`);
          failed++;
        }

        // Test 2: Analyze endpoint
        try {
          const { result, elapsed } = await runWithTiming(
            () => apiClient.post(`/api/v1/banking/analyze/${memberId}`, {}),
            'banking-test-analyze'
          );
          if (result?.member_id && result?.insights) {
            appendLog('success', `[Banking] ✓ Analyze endpoint OK (${formatMs(elapsed)})`);
            passed++;
          } else {
            throw new Error('Invalid analyze response');
          }
        } catch (e) {
          appendLog('error', `[Banking] ✗ Analyze endpoint failed: ${e.message}`);
          failed++;
        }

        appendLog('info', `[Banking] Test Summary: ${passed} passed, ${failed} failed`);
        return;
      }

      appendLog('warn', 'Usage: banking analyze <member_id> | banking dataset <member_id> | banking test');
    }

    async function dispatch(line) {
      const trimmed = (line || '').trim();
      if (!trimmed) return;
      appendLog('debug', `> ${trimmed}`);
      const tokens = tokenize(trimmed);
      if (!tokens.length) return;
      const [cmd, ...rest] = tokens;
      switch (cmd) {
        case 'help':
        case '?':
          printHelp();
          break;
        case 'clear':
          clearLog();
          break;
        case 'raw':
          setRaw((rest[0] || '').toLowerCase() === 'on');
          break;
        case 'rag':
          await handleRag(rest);
          break;
        case 'gemma':
          await handleGemma(rest);
          break;
        case 'emotion':
          await handleEmotion(rest);
          break;
        case 'show':
          if ((rest[0] || '').toLowerCase() === 'context') showContext();
          else if ((rest[0] || '').toLowerCase() === 'response') showResponse();
          else appendLog('warn', 'Usage: show context | show response');
          break;
        case 'gpu':
          await handleGpu();
          break;
        case 'warmup':
          await handleWarmup();
          break;
        // Enterprise commands
        case 'qa':
          await handleQA(rest);
          break;
        case 'automation':
          await handleAutomation(rest);
          break;
        case 'knowledge':
        case 'kb':
          await handleKnowledge(rest);
          break;
        case 'analytics':
          await handleAnalytics(rest);
          break;
        case 'meetings':
        case 'meeting':
          await handleMeetings(rest);
          break;
        // ML Profit commands
        case 'ml':
          await handleML(rest);
          break;
        // Banking Hub commands (SCU Integration)
        case 'banking':
        case 'bank':
          await handleBanking(rest);
          break;
        default:
          appendLog('warn', `Unknown command: ${cmd}. Type "help" for options.`);
      }
    }

    function handleHistoryNavigation(evt) {
      if (!state.history.length) return;
      if (evt.key === 'ArrowUp') {
        evt.preventDefault();
        if (state.historyIndex < 0) state.historyIndex = state.history.length - 1;
        else state.historyIndex = Math.max(0, state.historyIndex - 1);
        inputEl.value = state.history[state.historyIndex] || '';
      } else if (evt.key === 'ArrowDown') {
        evt.preventDefault();
        if (state.historyIndex >= state.history.length - 1) {
          state.historyIndex = -1;
          inputEl.value = '';
        } else {
          state.historyIndex = Math.min(state.history.length - 1, state.historyIndex + 1);
          inputEl.value = state.history[state.historyIndex] || '';
        }
      }
    }

    async function handleSubmit() {
      if (!inputEl) return;
      const value = inputEl.value.trim();
      if (!value) return;
      state.history.push(value);
      state.historyIndex = -1;
      inputEl.value = '';
      await dispatch(value);
    }

    if (runBtn) runBtn.addEventListener('click', handleSubmit);
    if (inputEl) {
      inputEl.addEventListener('keydown', async (evt) => {
        if (evt.key === 'Enter') {
          evt.preventDefault();
          await handleSubmit();
        } else if (evt.key === 'ArrowUp' || evt.key === 'ArrowDown') {
          handleHistoryNavigation(evt);
        }
      });
    }
    if (clearBtn) clearBtn.addEventListener('click', clearLog);
    if (helpBtn) helpBtn.addEventListener('click', printHelp);
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        root.classList.toggle('active');
        if (typeof onToggle === 'function') onToggle(root.classList.contains('active'));
        if (root.classList.contains('active') && inputEl) {
          setTimeout(() => inputEl.focus(), 50);
        }
      });
    }

    appendLog('info', 'RAG CLI ready. Type "help" for commands.');
    setSessionChip();
    return {
      state,
      clear: clearLog,
      log: appendLog,
    };
  }

  global.initRagCliShell = initRagCliShell;
})(typeof window !== 'undefined' ? window : globalThis);
