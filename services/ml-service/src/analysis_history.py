"""
Analysis History Persistence Service
=====================================
Stores all engine runs to SQLite for multi-run comparison.
Enables "Test Again?" feature and run navigation.

Author: Nemo AI Platform
Created: December 2025
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Database location - use environment variable or fallback to local path
# In Docker: /app/data/databases/ (mounted volume)
# Local: relative to project root
def _get_db_path():
    """Get database path from environment or compute reasonable default."""
    # Check for explicit environment variable first
    if os.environ.get('ANALYSIS_HISTORY_DB'):
        return Path(os.environ['ANALYSIS_HISTORY_DB'])
    
    # Check for common Docker paths
    docker_paths = [
        Path('/app/data/databases/analysis_history.db'),
        Path('/data/databases/analysis_history.db'),
    ]
    
    for p in docker_paths:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except PermissionError:
            continue
    
    # Fallback: use temp directory that always exists
    tmp_path = Path('/tmp/nemo_analysis_history.db')
    logger.warning(f"Using temporary database path: {tmp_path}")
    return tmp_path

DB_PATH = _get_db_path()


class AnalysisHistoryService:
    """
    Manages persistent storage of analysis runs.
    
    Features:
    - Store unlimited sessions with up to MAX_RUNS_PER_ENGINE runs each
    - Track used target columns for "Test Again?" exclusion
    - Query runs by session, engine, or specific index
    - Auto-cleanup of oldest runs when limit exceeded
    """
    
    MAX_RUNS_PER_ENGINE = 20
    
    def __init__(self, dbPath: Optional[Path] = None):
        """Initialize the history service with optional custom DB path."""
        self.dbPath = dbPath or DB_PATH
        self._ensureDb()
    
    def _ensureDb(self):
        """Create database and tables if they don't exist."""
        self.dbPath.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.dbPath) as conn:
            # Main runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    engine_name TEXT NOT NULL,
                    run_index INTEGER NOT NULL,
                    target_column TEXT,
                    feature_columns TEXT,
                    results TEXT NOT NULL,
                    gemma_summary TEXT,
                    config TEXT,
                    score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, engine_name, run_index)
                )
            """)
            
            # Index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_engine 
                ON analysis_runs(session_id, engine_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id
                ON analysis_runs(session_id)
            """)
            
            # Sessions table for metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    session_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    columns TEXT,
                    row_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info(f"Analysis history database initialized at {self.dbPath}")
    
    def saveSession(
        self, 
        sessionId: str, 
        filename: str, 
        columns: List[str],
        rowCount: Optional[int] = None
    ) -> bool:
        """
        Create or update a session record.
        
        Args:
            sessionId: Unique session identifier
            filename: Original uploaded filename
            columns: List of column names in the dataset
            rowCount: Number of rows in the dataset
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.dbPath) as conn:
                conn.execute("""
                    INSERT INTO analysis_sessions (session_id, filename, columns, row_count, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        filename = excluded.filename,
                        columns = excluded.columns,
                        row_count = excluded.row_count,
                        updated_at = excluded.updated_at
                """, (sessionId, filename, json.dumps(columns), rowCount, datetime.now()))
                conn.commit()
            logger.info(f"Session saved: {sessionId} ({filename}, {len(columns)} columns)")
            return True
        except Exception as e:
            logger.error(f"Failed to save session {sessionId}: {e}")
            return False
    
    def saveRun(
        self,
        sessionId: str,
        engineName: str,
        runIndex: int,
        results: Dict[str, Any],
        targetColumn: Optional[str] = None,
        featureColumns: Optional[List[str]] = None,
        gemmaSummary: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None
    ) -> bool:
        """
        Save an analysis run, enforcing MAX_RUNS_PER_ENGINE limit.
        
        When limit is exceeded, the oldest run is deleted to make room.
        
        Args:
            sessionId: Session identifier
            engineName: Engine name (titan, oracle, chronos, etc.)
            runIndex: 0-based run index
            results: Full results dictionary
            targetColumn: Target column used (for exclusion tracking)
            featureColumns: Features used
            gemmaSummary: Gemma's analysis summary
            config: Engine configuration used
            score: Primary score/metric
            
        Returns:
            True if successful
        """
        try:
            # Check run count limit
            existingCount = self.getRunCount(sessionId, engineName)
            if existingCount >= self.MAX_RUNS_PER_ENGINE:
                self._deleteOldestRun(sessionId, engineName)
                logger.info(f"Deleted oldest run for {sessionId}/{engineName} (limit: {self.MAX_RUNS_PER_ENGINE})")
            
            with sqlite3.connect(self.dbPath) as conn:
                conn.execute("""
                    INSERT INTO analysis_runs 
                    (session_id, engine_name, run_index, target_column, feature_columns, 
                     results, gemma_summary, config, score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, engine_name, run_index) DO UPDATE SET
                        results = excluded.results,
                        gemma_summary = excluded.gemma_summary,
                        config = excluded.config,
                        score = excluded.score
                """, (
                    sessionId,
                    engineName,
                    runIndex,
                    targetColumn,
                    json.dumps(featureColumns) if featureColumns else None,
                    json.dumps(results),
                    gemmaSummary,
                    json.dumps(config) if config else None,
                    score
                ))
                conn.commit()
            
            logger.info(f"Run saved: {sessionId}/{engineName}/run_{runIndex} (target: {targetColumn})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save run {sessionId}/{engineName}/{runIndex}: {e}")
            return False
    
    def getRuns(self, sessionId: str, engineName: str) -> List[Dict[str, Any]]:
        """
        Get all runs for a session/engine combination.
        
        Args:
            sessionId: Session identifier
            engineName: Engine name
            
        Returns:
            List of run dictionaries, ordered by run_index ASC
        """
        try:
            with sqlite3.connect(self.dbPath) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM analysis_runs
                    WHERE session_id = ? AND engine_name = ?
                    ORDER BY run_index ASC
                """, (sessionId, engineName))
                rows = cursor.fetchall()
            
            return [self._rowToDict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get runs for {sessionId}/{engineName}: {e}")
            return []
    
    def getRun(
        self, 
        sessionId: str, 
        engineName: str, 
        runIndex: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific run by index.
        
        Args:
            sessionId: Session identifier
            engineName: Engine name
            runIndex: 0-based run index
            
        Returns:
            Run dictionary or None if not found
        """
        try:
            with sqlite3.connect(self.dbPath) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM analysis_runs
                    WHERE session_id = ? AND engine_name = ? AND run_index = ?
                """, (sessionId, engineName, runIndex))
                row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._rowToDict(row)
            
        except Exception as e:
            logger.error(f"Failed to get run {sessionId}/{engineName}/{runIndex}: {e}")
            return None
    
    def getRunCount(self, sessionId: str, engineName: str) -> int:
        """Get number of runs for a session/engine."""
        try:
            with sqlite3.connect(self.dbPath) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM analysis_runs
                    WHERE session_id = ? AND engine_name = ?
                """, (sessionId, engineName))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get run count: {e}")
            return 0
    
    def getNextRunIndex(self, sessionId: str, engineName: str) -> int:
        """Get the next available run index (max + 1)."""
        try:
            with sqlite3.connect(self.dbPath) as conn:
                cursor = conn.execute("""
                    SELECT COALESCE(MAX(run_index), -1) + 1 FROM analysis_runs
                    WHERE session_id = ? AND engine_name = ?
                """, (sessionId, engineName))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get next run index: {e}")
            return 0
    
    def getUsedTargets(self, sessionId: str, engineName: str) -> List[str]:
        """
        Get list of target columns already used in previous runs.
        Used for "Test Again?" exclusion.
        """
        try:
            with sqlite3.connect(self.dbPath) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT target_column FROM analysis_runs
                    WHERE session_id = ? AND engine_name = ? AND target_column IS NOT NULL
                """, (sessionId, engineName))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get used targets: {e}")
            return []
    
    def getSessionColumns(self, sessionId: str) -> List[str]:
        """Get all columns for a session."""
        try:
            with sqlite3.connect(self.dbPath) as conn:
                cursor = conn.execute("""
                    SELECT columns FROM analysis_sessions WHERE session_id = ?
                """, (sessionId,))
                row = cursor.fetchone()
            
            if row and row[0]:
                return json.loads(row[0])
            return []
        except Exception as e:
            logger.error(f"Failed to get session columns: {e}")
            return []
    
    def getSession(self, sessionId: str) -> Optional[Dict[str, Any]]:
        """Get session metadata."""
        try:
            with sqlite3.connect(self.dbPath) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM analysis_sessions WHERE session_id = ?
                """, (sessionId,))
                row = cursor.fetchone()
            
            if not row:
                return None
            
            return {
                "session_id": row["session_id"],
                "filename": row["filename"],
                "columns": json.loads(row["columns"]) if row["columns"] else [],
                "row_count": row["row_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def getAllSessions(self) -> List[Dict[str, Any]]:
        """Get all sessions with run counts per engine."""
        try:
            with sqlite3.connect(self.dbPath) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT s.*, 
                           (SELECT COUNT(*) FROM analysis_runs r WHERE r.session_id = s.session_id) as total_runs,
                           (SELECT GROUP_CONCAT(DISTINCT engine_name) FROM analysis_runs r WHERE r.session_id = s.session_id) as engines_used
                    FROM analysis_sessions s
                    ORDER BY s.updated_at DESC
                """)
                rows = cursor.fetchall()
            
            return [
                {
                    "session_id": row["session_id"],
                    "filename": row["filename"],
                    "columns": json.loads(row["columns"]) if row["columns"] else [],
                    "row_count": row["row_count"],
                    "total_runs": row["total_runs"],
                    "engines_used": row["engines_used"].split(",") if row["engines_used"] else [],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get all sessions: {e}")
            return []
    
    def getSessionRunSummary(self, sessionId: str) -> Dict[str, int]:
        """Get run counts per engine for a session."""
        try:
            with sqlite3.connect(self.dbPath) as conn:
                cursor = conn.execute("""
                    SELECT engine_name, COUNT(*) as run_count
                    FROM analysis_runs
                    WHERE session_id = ?
                    GROUP BY engine_name
                """, (sessionId,))
                return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get session run summary: {e}")
            return {}
    
    def _deleteOldestRun(self, sessionId: str, engineName: str):
        """Delete the oldest run for this session/engine."""
        try:
            with sqlite3.connect(self.dbPath) as conn:
                conn.execute("""
                    DELETE FROM analysis_runs
                    WHERE id = (
                        SELECT id FROM analysis_runs
                        WHERE session_id = ? AND engine_name = ?
                        ORDER BY run_index ASC
                        LIMIT 1
                    )
                """, (sessionId, engineName))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to delete oldest run: {e}")
    
    def deleteRun(self, sessionId: str, engineName: str, runIndex: int) -> bool:
        """Delete a specific run."""
        try:
            with sqlite3.connect(self.dbPath) as conn:
                conn.execute("""
                    DELETE FROM analysis_runs
                    WHERE session_id = ? AND engine_name = ? AND run_index = ?
                """, (sessionId, engineName, runIndex))
                conn.commit()
            logger.info(f"Deleted run: {sessionId}/{engineName}/{runIndex}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete run: {e}")
            return False
    
    def deleteSession(self, sessionId: str) -> bool:
        """Delete a session and all its runs."""
        try:
            with sqlite3.connect(self.dbPath) as conn:
                conn.execute("DELETE FROM analysis_runs WHERE session_id = ?", (sessionId,))
                conn.execute("DELETE FROM analysis_sessions WHERE session_id = ?", (sessionId,))
                conn.commit()
            logger.info(f"Deleted session: {sessionId}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def _rowToDict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary."""
        return {
            "id": row["id"],
            "run_index": row["run_index"],
            "target_column": row["target_column"],
            "feature_columns": json.loads(row["feature_columns"]) if row["feature_columns"] else [],
            "results": json.loads(row["results"]),
            "gemma_summary": row["gemma_summary"],
            "config": json.loads(row["config"]) if row["config"] else None,
            "score": row["score"],
            "created_at": row["created_at"]
        }


# Global instance
historyService = AnalysisHistoryService()
