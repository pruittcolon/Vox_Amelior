"""
Correlation Engine
------------------
Analyzes quantitative scores from Gemma to find trends and anomalies.
Currently focuses on internal transcript metrics, preparing for external DB joins.
"""

import sqlite3
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os

@dataclass
class HealthTrend:
    date: str
    avg_health_score: float
    avg_deadline_stress: float
    avg_emotional_conflict: float
    segment_count: int

@dataclass
class Anomaly:
    date: str
    metric: str
    value: float
    threshold: float
    severity: str  # "HIGH", "MEDIUM", "LOW"

class CorrelationEngine:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to relative path from src/
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.db_path = os.path.join(base_dir, "data", "transcript_scores.db")
        else:
            self.db_path = db_path
            
    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def get_health_trends(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[HealthTrend]:
        """
        Aggregates scores by date to show health/stress trends.
        """
        query = """
            SELECT 
                recording_date,
                AVG(health_score) as avg_health,
                AVG(deadline_stress) as avg_stress,
                AVG(emotional_conflict) as avg_conflict,
                COUNT(*) as count
            FROM transcript_scores
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND recording_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND recording_date <= ?"
            params.append(end_date)
            
        query += " GROUP BY recording_date ORDER BY recording_date ASC"
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        trends = []
        for _, row in df.iterrows():
            trends.append(HealthTrend(
                date=row['recording_date'] or "Unknown",
                avg_health_score=row['avg_health'],
                avg_deadline_stress=row['avg_stress'],
                avg_emotional_conflict=row['avg_conflict'],
                segment_count=row['count']
            ))
            
        return trends

    def detect_anomalies(self, stress_threshold: float = 7.0, health_threshold: float = 4.0) -> List[Anomaly]:
        """
        Identifies dates with unusually high stress or low health.
        """
        trends = self.get_health_trends()
        anomalies = []
        
        for t in trends:
            # High Stress Anomaly
            if t.avg_deadline_stress >= stress_threshold:
                anomalies.append(Anomaly(
                    date=t.date,
                    metric="deadline_stress",
                    value=t.avg_deadline_stress,
                    threshold=stress_threshold,
                    severity="HIGH" if t.avg_deadline_stress > 8.5 else "MEDIUM"
                ))
            
            # Low Health Anomaly
            if t.avg_health_score <= health_threshold:
                anomalies.append(Anomaly(
                    date=t.date,
                    metric="health_score",
                    value=t.avg_health_score,
                    threshold=health_threshold,
                    severity="HIGH" if t.avg_health_score < 3.0 else "MEDIUM"
                ))
                
        return anomalies

    async def correlate_with_salesforce(self, sf_data: List[Dict]) -> Dict[str, float]:
        """
        Placeholder for Salesforce correlation.
        Compares Transcript Health vs Deal Close Rate.
        """
        # TODO: Implement actual join logic once Salesforce integration is active
        return {"correlation_coefficient": 0.0}
