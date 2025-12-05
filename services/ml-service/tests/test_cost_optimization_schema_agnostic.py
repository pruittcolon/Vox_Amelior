"""
Test Schema-Agnostic Cost Optimization Engine

Tests the refactored engine with diverse dataset structures.

Author: Nemo Server ML Team
Date: 2025-11-27
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from engines.cost_optimization_engine import CostOptimizationEngine


class TestSchemaAgnosticCostOptimization(unittest.TestCase):
    """Test cost optimization with various dataset structures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = CostOptimizationEngine()
    
    def test_financial_dataset(self):
        """Test with financial dataset (standard column names)."""
        np.random.seed(42)
        df = pd.DataFrame({
            'expense_category': np.random.choice(['IT', 'Marketing', 'Operations', 'HR'], 100),
            'total_spend_usd': np.random.uniform(1000, 10000, 100),
            'transaction_date': pd.date_range('2024-01-01', periods=100),
            'department': np.random.choice(['Engineering', 'Sales', 'Finance'], 100)
        })
        
        result = self.engine.analyze(df)
        
        # Should succeed
        self.assertNotIn('error', result)
        
        # Should have detected cost column
        self.assertIn('column_mappings', result)
        self.assertIn('cost', result['column_mappings'])
        self.assertEqual(result['column_mappings']['cost']['column'], 'total_spend_usd')
        
        # Should have total cost
        self.assertIn('total_cost', result['summary'])
        self.assertGreater(result['summary']['total_cost'], 0)
        
        # Should have graphs
        self.assertGreater(len(result['graphs']), 0)
        
        # Should use profiling
        self.assertTrue(result.get('profiling_used', False))
        
        print("✓ Financial dataset test passed")
        print(f"  - Detected cost column: {result['column_mappings']['cost']['column']}")
        print(f"  - Total cost: ${result['summary']['total_cost']:,.2f}")
        print(f"  - Mapping confidence: {result['summary']['mapping_confidence']:.2%}")
    
    def test_operations_dataset(self):
        """Test with operations dataset (different naming convention)."""
        np.random.seed(43)
        df = pd.DataFrame({
            'dept': np.random.choice(['Warehouse', 'Logistics', 'Maintenance'], 80),
            'operational_cost': np.random.uniform(500, 5000, 80),
            'resource_type': np.random.choice(['Equipment', 'Labor', 'Materials'], 80)
        })
        
        result = self.engine.analyze(df)
        
        # Should succeed
        self.assertNotIn('error', result)
        
        # Should detect operational_cost
        self.assertIn('cost', result['column_mappings'])
        self.assertEqual(result['column_mappings']['cost']['column'], 'operational_cost')
        
        # Should have grouping
        self.assertIn('grouping', result['column_mappings'])
        self.assertIn(result['column_mappings']['grouping']['column'], ['dept', 'resource_type'])
        
        print("✓ Operations dataset test passed")
        print(f"  - Detected cost column: {result['column_mappings']['cost']['column']}")
        print(f"  - Detected grouping: {result['column_mappings']['grouping']['column']}")
    
    def test_retail_dataset(self):
        """Test with retail dataset (price instead of cost)."""
        np.random.seed(44)
        df = pd.DataFrame({
            'product_type': np.random.choice(['Electronics', 'Clothing', 'Food'], 120),
            'wholesale_price': np.random.uniform(20, 200, 120),
            'supplier': np.random.choice(['SupplierA', 'SupplierB', 'SupplierC'], 120),
            'quantity': np.random.randint(1, 100, 120)
        })
        
        result = self.engine.analyze(df)
        
        # Should succeed
        self.assertNotIn('error', result)
        
        # Should detect wholesale_price as cost
        self.assertIn('cost', result['column_mappings'])
        self.assertEqual(result['column_mappings']['cost']['column'], 'wholesale_price')
        
        # Should have Pareto analysis (since grouping exists)
        if 'pareto' in result:
            self.assertIn('vital_few_count', result['pareto'])
            self.assertIn('graph', result['pareto'])
        
        print("✓ Retail dataset test passed")
        print(f"  - Uses 'wholesale_price' as cost metric")
    
    def test_minimal_dataset(self):
        """Test with minimal dataset (just numeric column)."""
        df = pd.DataFrame({
            'amount': np.random.uniform(100, 1000, 30),
            'id': range(30)
        })
        
        result = self.engine.analyze(df)
        
        # Should succeed even without grouping
        self.assertNotIn('error', result)
        
        # Should detect amount
        self.assertIn('cost', result['column_mappings'])
        
        # Should have distribution analysis
        self.assertIn('distribution', result)
        
        print("✓ Minimal dataset test passed")
    
    def test_missing_requirements(self):
        """Test dataset without numeric columns."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'category': ['A', 'B', 'C']
        })
        
        result = self.engine.analyze(df)
        
        # Should fail gracefully
        self.assertIn('error', result)
        self.assertIn('missing_requirements', result)
        
        print("✓ Missing requirements test passed (correctly rejected)")
    
    def test_engine_requirements(self):
        """Test that engine requirements are properly defined."""
        requirements = self.engine.get_requirements()
        
        self.assertIsNotNone(requirements)
        self.assertGreater(len(requirements.required_semantics), 0)
        self.assertEqual(requirements.min_rows, 20)
        
        print("✓ Engine requirements test passed")
    
    def test_mixed_confidence_scenario(self):
        """Test with ambiguous column names."""
        df = pd.DataFrame({
            'cost1': np.random.uniform(100, 500, 50),
            'cost2': np.random.uniform(200, 600, 50),
            'type': np.random.choice(['X', 'Y'], 50)
        })
        
        result = self.engine.analyze(df)
        
        # Should succeed and pick one
        self.assertNotIn('error', result)
        self.assertIn('cost', result['column_mappings'])
        self.assertIn(result['column_mappings']['cost']['column'], ['cost1', 'cost2'])
        
        # Confidence might be lower due to ambiguity
        confidence = result['summary'].get('mapping_confidence', 1.0)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        print(f"✓ Ambiguous scenario test passed (confidence: {confidence:.2%})")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Schema-Agnostic Cost Optimization Engine")
    print("=" * 70)
    unittest.main(verbosity=2)
