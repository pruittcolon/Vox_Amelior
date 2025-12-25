#!/usr/bin/env python3
"""
Quick test: Universal Graph Generator on users.db

Verifies the Universal Graph Engine can generate 10+ graphs for any dataset
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engines.universal_graph_engine import UniversalGraphEngine


def main():
    # Connect to users.db
    db_path = (
        Path(__file__).parent.parent.parent.parent / "services" / "api-gateway" / "instance" / "users.db.container"
    )

    print("=" * 60)
    print("UNIVERSAL GRAPH GENERATOR - QUICK TEST")
    print("=" * 60)
    print(f"\nDatabase: {db_path}")
    print(f"Exists: {db_path.exists()}\n")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}\n")

    # Run Universal Graph Generator
    engine = UniversalGraphEngine()
    result = engine.generate_graphs(df)

    graphs = result["graphs"]
    print(f"✅ Generated {len(graphs)} visualizations\n")

    # List all graphs
    print("Graphs generated:")
    for i, graph in enumerate(graphs, 1):
        print(f"  {i}. {graph['type']}: {graph['title']} (priority: {graph.get('priority', 'N/A')})")

    # Verify requirement
    if len(graphs) >= 10:
        print(f"\n🎉 SUCCESS: Generated {len(graphs)} >= 10 graphs!")
        print("\n✅ Universal Graph Generator is working correctly")
        return 0
    else:
        print(f"\n❌ FAILED: Only {len(graphs)} graphs (need 10+)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
