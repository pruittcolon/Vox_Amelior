
import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.correlation_engine import CorrelationEngine
from src.gemma_scorer import get_score_database

def test_correlation_engine():
    print("Testing Correlation Engine...")
    
    # Use the real DB (which has mock data)
    db_path = "/mnt/Nemo_Server/services/ml-service/data/transcript_scores.db"
    
    engine = CorrelationEngine(db_path=db_path)
    
    # 1. Test Trends
    print("\n1. Health Trends:")
    trends = engine.get_health_trends()
    for t in trends:
        print(f"  Date: {t.date}, Health: {t.avg_health_score:.2f}, Stress: {t.avg_deadline_stress:.2f}, Count: {t.segment_count}")
        
    if not trends:
        print("  No trends found (DB might be empty of valid dates)")
        
    # 2. Test Anomalies
    print("\n2. Anomalies (Stress > 4.0 for testing):")
    anomalies = engine.detect_anomalies(stress_threshold=4.0) 
    for a in anomalies:
        print(f"  [{a.severity}] {a.date}: {a.metric} = {a.value:.2f}")

    if not anomalies:
        print("  No anomalies found.")
        
    print("\nCorrelation Engine Test Complete.")

if __name__ == "__main__":
    test_correlation_engine()
