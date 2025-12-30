# Gemma Database Quality Analysis Feature

## Overview

This document details the implementation of the Gemma-powered database quality scoring feature, which analyzes uploaded CSV/Excel files and generates a new "insights" database with Gemma's quality scores appended to each row.

---

## Feature Summary

**Goal**: When a user uploads a database and clicks "Analyze", Gemma evaluates the data quality and creates a NEW database file containing:
- All original data columns
- 7 new score columns: `Q1_anomaly`, `Q2_business`, `Q3_validity`, `Q4_complete`, `Q5_consistent`, `overall`, `findings`

The original database is **never modified** — an `_insights.csv` file is created instead.

---

## Files Modified

### Backend (ML Service)

#### [database_scoring.py](file:///home/pruittcolon/Desktop/Nemo_Server/services/ml-service/src/routers/database_scoring.py)

| Function | Changes |
|----------|---------|
| `_generate_insights_csv()` | **Rewrote** to load original CSV, add score columns to EACH ROW based on chunk assignment, save as `{filename}_insights.csv` |
| `_format_chunk_for_scoring()` | Reduced from 10 rows × 15 cols to 3 rows × 8 cols for faster inference |
| `_save_results()` | Updated summary keys from `completeness/accuracy` to `Q1/Q2/Q3/Q4/Q5` format |
| `get_job_results()` | Fixed KeyError by updating to use `Q1-Q5` keys |
| Module-level auth | Added `ServiceAuth` initialization to avoid circular imports |

### Backend (Gemma Service)

#### [main.py](file:///home/pruittcolon/Desktop/Nemo_Server/services/gemma-service/src/main.py)

| Location | Changes |
|----------|---------|
| Line ~316 | Added `/score-chunk` to exempt paths in `ServiceAuthMiddleware` |
| Line ~1094 | Rewrote prompt to use **strict JSON format** with explicit integer schema |
| Line ~1127 | Replaced regex parser with **JSON parser** + regex fallback |
| Line ~1121 | Set `max_tokens=80`, `temperature=0.1` for consistent output |

### Frontend

#### [gemma.html](file:///home/pruittcolon/Desktop/Nemo_Server/frontend/gemma.html)

| Function | Changes |
|----------|---------|
| `displayScoringResults()` | Updated score card dimensions to `Q1-Q5` format |
| `displayScoringResults()` | Added auto-switch to insights database after scoring completes |
| `loadSavedDatabases()` | Called after scoring to refresh file list |

---

## Issues Encountered & Solutions

### Issue 1: 401 Unauthorized on `/score-chunk`

**Symptom**: ML service received 401 when calling Gemma's `/score-chunk` endpoint.

**Root Cause**: Gemma's `ServiceAuthMiddleware` only exempted `/health` and `/settings`.

**Solution**: Added `/score-chunk` to exempt paths in `gemma-service/src/main.py` line ~318:
```python
if request.url.path in ["/health", "/settings", "/score-chunk"]:
```

**Additional Fix**: Container required full rebuild (`docker compose build --no-cache gemma-service`) because code is copied at build time, not mounted.

---

### Issue 2: Circular Import in ML Service

**Symptom**: `ImportError` when importing `service_auth` from `src.main` in `database_scoring.py`.

**Solution**: Initialize `ServiceAuth` at module level using `load_service_jwt_keys()`:
```python
from shared.security.service_auth import ServiceAuth, load_service_jwt_keys

_scoring_service_auth: ServiceAuth | None = None
try:
    keys = load_service_jwt_keys("ml-service")
    _scoring_service_auth = ServiceAuth(service_name="ml-service", jwt_keys=keys)
except Exception:
    pass
```

---

### Issue 3: KeyError 'completeness' on Results Endpoint

**Symptom**: 500 error when fetching `/database-scoring/results/{job_id}`.

**Root Cause**: Code used old key names (`completeness`, `accuracy`) but Gemma returns `Q1`, `Q2`, etc.

**Solution**: Updated both `_save_results()` and `get_job_results()` to use `Q1-Q5` keys:
```python
"summary": {
    "avg_Q1": _avg_score(job.chunks, "Q1"),
    "avg_Q2": _avg_score(job.chunks, "Q2"),
    ...
}
```

---

### Issue 4: Gemma Outputting Only 1 Token

**Symptom**: Scores always defaulted to 5 because Gemma generated minimal output.

**Root Cause**: Prompt was too vague, didn't clearly specify expected output format.

**Solution**: Rewrote prompt to use strict JSON format with explicit schema:
```python
scoring_prompt = f"""You are a data quality analyzer. Score this data chunk on a scale of 1-10 (integers only).

DATA:
{request.chunk_content}

Respond ONLY with valid JSON. No other text allowed.

JSON Schema:
{{"Q1": integer 1-10, "Q2": integer 1-10, ...}}

Example valid response:
{{"Q1": 7, "Q2": 8, "Q3": 9, "Q4": 6, "Q5": 8, "findings": "column X has outliers"}}

Your JSON response:"""
```

---

### Issue 5: Slow Scoring (26+ seconds per chunk)

**Symptom**: Each chunk took 20-30 seconds to score.

**Root Causes**:
1. Prompt tokens too high (~1300 tokens from chunk data)
2. GPU coordinator added ~5s delay per request

**Solutions**:
1. Reduced chunk format: 3 rows × 8 cols (was 10 × 15)
2. Reduced `max_tokens` from 256 to 80
3. Set `temperature=0.1` for faster, more consistent output

**Expected Performance**: ~8-10 seconds per chunk

---

### Issue 6: UI Not Showing Score Columns

**Symptom**: After scoring, database viewer still showed original file without score columns.

**Root Cause**: UI didn't switch to the new `_insights.csv` file.

**Solution**: Added auto-switch logic to `displayScoringResults()`:
```javascript
setTimeout(async () => {
    const dbData = await fetch('/databases').then(r => r.json());
    const insightsDb = dbData.databases.find(db => db.filename.includes('_insights'));
    if (insightsDb) {
        selectDatabase(insightsDb.filename, insightsDb.display_name);
    }
}, 500);
```

---

## How to Test

1. **Start Services**:
   ```bash
   cd docker && docker compose up -d
   ```

2. **Navigate to UI**:
   - Go to `http://localhost:8000/ui/gemma.html`
   - Click "Databases" tab

3. **Upload & Analyze**:
   - Upload a CSV file (or use existing)
   - Click "Use" on a database
   - Click "Analyze" button

4. **Verify Output**:
   - Watch console for: `📊 Scoring status: 1/3 → 3/3`
   - After completion: `📊 Auto-switching to insights database`
   - Check that viewer shows new columns: `Q1_anomaly`, `Q2_business`, etc.

5. **Check Logs**:
   ```bash
   docker logs refactored_gemma 2>&1 | grep SCORE-CHUNK | tail -20
   docker logs refactored_ml_service 2>&1 | grep SCORING | tail -20
   ```

---

## Insights CSV Output Format

| Original Columns | Q1_anomaly | Q2_business | Q3_validity | Q4_complete | Q5_consistent | overall | findings |
|------------------|------------|-------------|-------------|-------------|---------------|---------|----------|
| (all original data) | 7 | 8 | 9 | 6 | 8 | 7.6 | "brief issue text" |

Each row gets scores from the chunk it belongs to. For example:
- Rows 0-16 (chunk 1) all get chunk 1's scores
- Rows 17-33 (chunk 2) all get chunk 2's scores
- etc.

---

## Rebuilding After Changes

If you modify Gemma service code:
```bash
cd docker
docker compose build gemma-service
docker compose up -d gemma-service
```

If you modify ML service code:
```bash
cd docker
docker compose build ml-service
docker compose up -d ml-service
```

**Important**: Simple `docker compose restart` does NOT pick up code changes — you must rebuild!

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMMA_URL` | `http://gemma-service:8001` | Gemma service URL |
| `CHUNK_SIZE` | 20 | Rows per chunk |
| `TEST_MODE` | false | Score only first chunk |

### Scoring Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_tokens` | 80 | JSON response is compact |
| `temperature` | 0.1 | Consistent, deterministic output |
| Chunk rows | 3 | Speed (fewer tokens) |
| Chunk cols | 8 | Speed (fewer tokens) |

---

## Score Definitions

| Score | Question | Meaning |
|-------|----------|---------|
| Q1 | Anomaly Detection | 1=many anomalies, 10=no anomalies |
| Q2 | Business Reasonableness | 1=unrealistic values, 10=very reasonable |
| Q3 | Data Type Validity | 1=many invalid types, 10=all valid |
| Q4 | Completeness | 1=many missing values, 10=complete |
| Q5 | Consistency | 1=inconsistent, 10=consistent |

---

## Architecture Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Frontend   │────▶│ ML Service  │────▶│   Gemma     │
│ gemma.html  │     │ database_   │     │  /score-    │
│             │     │ scoring.py  │     │  chunk      │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  _insights  │
                    │    .csv     │
                    └─────────────┘
```

1. User clicks Analyze → Frontend calls `/database-scoring/score/{filename}`
2. ML Service chunks the data → Calls Gemma `/score-chunk` for each chunk
3. Gemma returns JSON scores → ML Service parses and stores
4. ML Service creates `{filename}_insights.csv` with original data + scores
5. Frontend auto-switches to show insights file

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| 401 on score-chunk | Verify `/score-chunk` in exempt paths, rebuild Gemma |
| 500 on results | Check for KeyError in logs, verify Q1-Q5 keys used |
| Scores all 5.0 | Check Gemma logs for JSON parse errors |
| Slow scoring | Reduce chunk size, check GPU coordinator logs |
| No insights file | Check ML service logs for pandas errors |
| UI not switching | Check console for "Auto-switching" message |
