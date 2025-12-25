#!/usr/bin/env python3
"""
Test script for Analytics Engines against all Nemo Server databases

Tests:
- StatisticalEngine against rag.db
- UniversalGraphEngine against rag.db, users.db, email.db
- Verifies 10+ graphs generated for each database
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engines.statistical_engine import StatisticalEngine
from engines.universal_graph_engine import UniversalGraphEngine


def test_statistical_engine_on_rag_db():
    """Test Statistical Engine on rag.db"""
    print("\n" + "=" * 60)
    print("TEST 1: Statistical Engine on rag.db")
    print("=" * 60)

    # Get Nemo_Server root: from test file location, go up 3 levels (tests -> ml-service -> services -> Nemo_Server)
    test_file = Path(__file__).absolute()
    nemo_root = test_file.parent.parent.parent.parent
    rag_db_path = nemo_root / "docker" / "rag_instance" / "rag.db"

    print(f"   Nemo root: {nemo_root}")
    print(f"   Looking for rag.db at: {rag_db_path}")

    if not rag_db_path.exists():
        print(f"❌ rag.db not found at {rag_db_path}")
        return False

    conn = sqlite3.connect(rag_db_path)

    # Test on transcript_segments table
    print("\n📊 Analyzing transcript_segments table...")
    df = pd.read_sql_query("SELECT * FROM transcript_segments LIMIT 1000", conn)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    engine = StatisticalEngine()

    # Run descriptive statistics
    result = engine.analyze(df, {"analysis_types": ["descriptive", "correlation", "distribution"]})

    print("\n✅ Analysis completed:")
    print(f"   - Numeric columns analyzed: {len(result.get('descriptive', {}).get('numeric', {}))}")
    print(f"   - Categorical columns analyzed: {len(result.get('descriptive', {}).get('categorical', {}))}")
    print(f"   - Visualizations generated: {len(result.get('visualizations', []))}")

    if result.get("correlation"):
        print(f"   - Correlation analysis: {len(result['correlation'].get('columns', []))} columns")

    conn.close()
    return True


def test_universal_graph_on_database(db_name: str, db_path: Path, table_name: str):
    """Test Universal Graph Generator on a specific database"""
    print(f"\n📊 Testing {db_name} - Table: {table_name}")

    if not db_path.exists():
        print(f"   ⚠️  Database not found at {db_path}")
        return False

    conn = sqlite3.connect(db_path)

    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1000", conn)
        print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

        engine = UniversalGraphEngine()
        result = engine.generate_graphs(df)

        graphs = result["graphs"]
        print(f"   ✅ Generated {len(graphs)} visualizations")

        # Print graph types
        graph_types = {}
        for graph in graphs:
            graph_type = graph["type"]
            graph_types[graph_type] = graph_types.get(graph_type, 0) + 1

        print("   Graph types:")
        for gtype, count in sorted(graph_types.items()):
            print(f"     - {gtype}: {count}")

        # Verify we got 10+ graphs
        if len(graphs) >= 10:
            print(f"   ✅ SUCCESS: Generated {len(graphs)} >= 10 graphs")
            return True
        else:
            print(f"   ❌ FAILED: Only {len(graphs)} graphs (need 10+)")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        conn.close()


def main():
    """Run all tests"""
    print("=" * 60)
    print("ANALYTICS ENGINES DATABASE TESTS")
    print("=" * 60)

    results = []

    # Test 1: Statistical Engine
    results.append(("Statistical Engine on rag.db", test_statistical_engine_on_rag_db()))

    # Test 2: Universal Graph Generator on rag.db
    print("\n" + "=" * 60)
    print("TEST 2: Universal Graph Generator on rag.db")
    print("=" * 60)

    test_file = Path(__file__).absolute()
    nemo_root = test_file.parent.parent.parent.parent
    rag_db_path = nemo_root / "docker" / "rag_instance" / "rag.db"
    results.append(
        (
            "UniversalGraph on rag.db/transcript_segments",
            test_universal_graph_on_database("rag.db", rag_db_path, "transcript_segments"),
        )
    )
    results.append(
        (
            "UniversalGraph on rag.db/transcript_records",
            test_universal_graph_on_database("rag.db", rag_db_path, "transcript_records"),
        )
    )

    # Test 3: Universal Graph Generator on users.db
    print("\n" + "=" * 60)
    print("TEST 3: Universal Graph Generator on users.db")
    print("=" * 60)

    users_db_path = nemo_root / "services" / "api-gateway" / "instance" / "users.db.container"
    results.append(
        ("UniversalGraph on users.db/users", test_universal_graph_on_database("users.db", users_db_path, "users"))
    )

    # Test 4: Universal Graph Generator on email.db
    print("\n" + "=" * 60)
    print("TEST 4: Universal Graph Generator on email.db")
    print("=" * 60)

    email_db_path = nemo_root / "docker" / "rag_instance" / "email.db"
    results.append(
        ("UniversalGraph on email.db", test_universal_graph_on_database("email.db", email_db_path, "emails"))
    )

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
