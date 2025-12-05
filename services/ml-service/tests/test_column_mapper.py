"""
Unit Tests for Column Mapper

Tests for smart column mapping functionality.

Author: Nemo Server ML Team
Date: 2025-11-27
"""

import unittest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from core.schema_intelligence import ColumnProfiler, SemanticType, BusinessEntity
from core.applicability_scorer import EngineRequirements, AnalyticsEngine
from core.column_mapper import ColumnMapper, auto_map_columns


class TestColumnMapper(unittest.TestCase):
    """Test cases for ColumnMapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mapper = ColumnMapper()
        self.profiler = ColumnProfiler()
    
    def test_exact_semantic_match(self):
        """Test mapping with exact semantic type match."""
        # Create dataset with clear semantic types
        df = pd.DataFrame({
            'total_cost': np.random.uniform(100, 1000, 50),
            'category': np.random.choice(['A', 'B', 'C'], 50)
        })
        
        profiles = self.profiler.profile_dataset(df)
        
        requirements = EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={'grouping': [SemanticType.CATEGORICAL]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10
        )
        
        result = self.mapper.map_columns(profiles, requirements)
        
        self.assertTrue(result.success)
        self.assertIn('numeric_continuous', result.mappings)
        self.assertIn('grouping', result.mappings)
    
    def test_business_entity_match(self):
        """Test mapping by business entity."""
        df = pd.DataFrame({
            'product_cost': np.random.uniform(10, 100, 50),
            'item_name': ['Item' + str(i) for i in range(50)]
        })
        
        profiles = self.profiler.profile_dataset(df)
        
        requirements = EngineRequirements(
            required_semantics=[],
            optional_semantics={},
            required_entities=[BusinessEntity.COST],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10
        )
        
        result = self.mapper.map_columns(profiles, requirements)
        
        self.assertTrue(result.success)
        self.assertIn('cost', result.mappings)
        self.assertEqual(result.mappings['cost'].column_name, 'product_cost')
    
    def test_fuzzy_name_matching(self):
        """Test fuzzy column name matching."""
        df = pd.DataFrame({
            'total_revenue_usd': np.random.uniform(1000, 5000, 50),
            'customer_segment': np.random.choice(['Premium', 'Standard'], 50)
        })
        
        profiles = self.profiler.profile_dataset(df)
        
        # Even without exact entity match, fuzzy matching should work
        requirements = EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10
        )
        
        result = self.mapper.map_columns(profiles, requirements)
        
        self.assertTrue(result.success)
        # Should map to total_revenue_usd
        self.assertIn('numeric_continuous', result.mappings)
    
    def test_missing_required_field(self):
        """Test handling of missing required fields."""
        df = pd.DataFrame({
            'description': ['text' + str(i) for i in range(50)]
        })
        
        profiles = self.profiler.profile_dataset(df)
        
        requirements = EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[BusinessEntity.COST],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10
        )
        
        result = self.mapper.map_columns(profiles, requirements)
        
        self.assertFalse(result.success)
        self.assertGreater(len(result.missing_required), 0)
        self.assertIn('numeric_continuous', result.missing_required)
    
    def test_ambiguous_mapping(self):
        """Test handling of ambiguous column mappings."""
        df = pd.DataFrame({
            'cost1': np.random.uniform(100, 1000, 50),
            'cost2': np.random.uniform(100, 1000, 50),
            'category': np.random.choice(['A', 'B'], 50)
        })
        
        profiles = self.profiler.profile_dataset(df)
        
        requirements = EngineRequirements(
            required_semantics=[],
            optional_semantics={'numeric': [SemanticType.NUMERIC_CONTINUOUS]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10
        )
        
        result = self.mapper.map_columns(profiles, requirements)
        
        # Should have ambiguity (two numeric columns)
        self.assertTrue(len(result.ambiguous) > 0 or 'numeric' in result.mappings)
        # But should still pick one
        if 'numeric' in result.mappings:
            self.assertIn(result.mappings['numeric'].column_name, ['cost1', 'cost2'])
    
    def test_ambiguity_resolution(self):
        """Test user-driven ambiguity resolution."""
        df = pd.DataFrame({
            'revenue': np.random.uniform(1000, 5000, 50),
            'cost': np.random.uniform(500, 2000, 50)
        })
        
        profiles = self.profiler.profile_dataset(df)
        
        requirements = EngineRequirements(
            required_semantics=[],
            optional_semantics={'value': [SemanticType.NUMERIC_CONTINUOUS]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10
        )
        
        result = self.mapper.map_columns(profiles, requirements)
        
        # User resolves to 'revenue'
        resolved = self.mapper.resolve_ambiguity(result, {'value': 'revenue'})
        
        self.assertEqual(resolved.mappings['value'].column_name, 'revenue')
        self.assertEqual(resolved.mappings['value'].confidence, 1.0)
    
    def test_auto_map_convenience(self):
        """Test auto_map_columns convenience function."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'sales': np.random.uniform(1000, 5000, 50),
            'region': np.random.choice(['North', 'South'], 50)
        })
        
        profiles = self.profiler.profile_dataset(df)
        
        requirements = EngineRequirements(
            required_semantics=[SemanticType.TEMPORAL, SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={'grouping': [SemanticType.CATEGORICAL]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10
        )
        
        mapping = auto_map_columns(profiles, requirements)
        
        self.assertIsInstance(mapping, dict)
        self.assertIn('temporal', mapping)
        self.assertIn('numeric_continuous', mapping)
        self.assertEqual(mapping['temporal'], 'date')
    
    def test_cost_optimization_realistic(self):
        """Test realistic cost optimization scenario."""
        # Realistic dataset with various naming conventions
        df = pd.DataFrame({
            'expense_category': np.random.choice(['IT', 'Marketing', 'Operations'], 100),
            'total_spend_usd': np.random.uniform(1000, 10000, 100),
            'transaction_date': pd.date_range('2024-01-01', periods=100),
            'vendor_name': ['Vendor' + str(i % 20) for i in range(100)]
        })
        
        profiles = self.profiler.profile_dataset(df)
        
        # Cost optimization requirements
        requirements = EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={'grouping': [SemanticType.CATEGORICAL]},
            required_entities=[],  # Flexible - can detect cost or not
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=50
        )
        
        result = self.mapper.map_columns(profiles, requirements)
        
        self.assertTrue(result.success)
        # Should map total_spend_usd to numeric requirement
        self.assertIn('numeric_continuous', result.mappings)
        self.assertEqual(result.mappings['numeric_continuous'].column_name, 'total_spend_usd')
        
        # Should find categorical grouping
        if 'grouping' in result.mappings:
            self.assertIn(result.mappings['grouping'].column_name, ['expense_category', 'vendor_name'])
    
    def test_confidence_scoring(self):
        """Test confidence scoring mechanism."""
        df = pd.DataFrame({
            'amount': np.random.uniform(100, 1000, 50),
            'type': np.random.choice(['A', 'B'], 50)
        })
        
        profiles = self.profiler.profile_dataset(df)
        
        requirements = EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={'category': [SemanticType.CATEGORICAL]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10
        )
        
        result = self.mapper.map_columns(profiles, requirements)
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # All mappings should have confidence scores
        for mapping in result.mappings.values():
            self.assertGreaterEqual(mapping.confidence, 0.0)
            self.assertLessEqual(mapping.confidence, 1.0)


if __name__ == '__main__':
    unittest.main()
