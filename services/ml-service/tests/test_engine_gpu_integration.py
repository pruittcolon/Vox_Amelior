"""
Integration Tests for Engine GPU Support

Tests that engines correctly coordinate GPU access and 
fall back to CPU when GPU is unavailable.

Author: Enterprise Analytics Team
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing"""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.choice(['A', 'B', 'C'], n_samples)
    })


@pytest.fixture
def mock_gpu_client():
    """Create a mock GPU client"""
    client = MagicMock()
    client.request_gpu = AsyncMock(return_value=MagicMock(
        acquired=True,
        task_id="ml-test-12345678",
        error=None
    ))
    client.release_gpu = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_gpu_client_unavailable():
    """Create a mock GPU client that fails to acquire GPU"""
    client = MagicMock()
    client.request_gpu = AsyncMock(return_value=MagicMock(
        acquired=False,
        task_id=None,
        error="GPU unavailable"
    ))
    client.release_gpu = AsyncMock(return_value=True)
    return client


# ============================================================================
# Mirror Engine GPU Tests
# ============================================================================

class TestMirrorEngineGPU:
    """Tests for Mirror Engine GPU support"""
    
    def test_mirror_engine_initialization_with_gpu_client(self, mock_gpu_client):
        """Test Mirror engine accepts GPU client"""
        from engines.mirror_engine import MirrorEngine
        
        engine = MirrorEngine(gpu_client=mock_gpu_client)
        assert engine._gpu_client is mock_gpu_client
    
    def test_mirror_engine_initialization_without_gpu_client(self):
        """Test Mirror engine works without GPU client"""
        from engines.mirror_engine import MirrorEngine
        
        engine = MirrorEngine()
        assert engine._gpu_client is None
    
    @pytest.mark.asyncio
    async def test_mirror_engine_gpu_acquisition(self, sample_dataframe, mock_gpu_client):
        """Test Mirror engine acquires GPU when available"""
        from engines.mirror_engine import MirrorEngine, CUDA_AVAILABLE
        
        engine = MirrorEngine(gpu_client=mock_gpu_client)
        
        if CUDA_AVAILABLE:
            acquired = await engine._acquire_gpu()
            assert acquired is True
            mock_gpu_client.request_gpu.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mirror_engine_gpu_release(self, mock_gpu_client):
        """Test Mirror engine releases GPU after use"""
        from engines.mirror_engine import MirrorEngine
        
        engine = MirrorEngine(gpu_client=mock_gpu_client)
        engine._gpu_acquired = True
        
        await engine._release_gpu()
        
        mock_gpu_client.release_gpu.assert_called_once()
        assert engine._gpu_acquired is False
    
    def test_mirror_engine_analyze_returns_used_gpu(self, sample_dataframe):
        """Test that analyze returns used_gpu field"""
        from engines.mirror_engine import MirrorEngine
        
        # Use minimal config for faster test
        engine = MirrorEngine()
        config = {'epochs': 5, 'num_rows': 10, 'use_gpu': False}
        
        result = engine.analyze(sample_dataframe, config)
        
        # Should have used_gpu field (will be False since no GPU client)
        assert 'used_gpu' in result


# ============================================================================
# Galileo Engine GPU Tests
# ============================================================================

class TestGalileoEngineGPU:
    """Tests for Galileo Engine GPU support"""
    
    def test_galileo_engine_initialization_with_gpu_client(self, mock_gpu_client):
        """Test Galileo engine accepts GPU client"""
        from engines.galileo_engine import GalileoEngine
        
        engine = GalileoEngine(gpu_client=mock_gpu_client)
        assert engine._gpu_client is mock_gpu_client
    
    def test_galileo_engine_device_default_cpu(self):
        """Test Galileo engine defaults to CPU device"""
        from engines.galileo_engine import GalileoEngine
        import torch
        
        engine = GalileoEngine()
        assert engine._device == torch.device("cpu")
    
    @pytest.mark.asyncio
    async def test_galileo_engine_gpu_acquisition_updates_device(self, mock_gpu_client):
        """Test GPU acquisition updates device to CUDA"""
        from engines.galileo_engine import GalileoEngine, CUDA_AVAILABLE
        import torch
        
        engine = GalileoEngine(gpu_client=mock_gpu_client)
        
        if CUDA_AVAILABLE:
            acquired = await engine._acquire_gpu()
            assert acquired is True
            assert engine._device == torch.device("cuda")
    
    @pytest.mark.asyncio
    async def test_galileo_engine_gpu_release_resets_device(self, mock_gpu_client):
        """Test GPU release resets device to CPU"""
        from engines.galileo_engine import GalileoEngine
        import torch
        
        engine = GalileoEngine(gpu_client=mock_gpu_client)
        engine._gpu_acquired = True
        engine._device = torch.device("cuda")
        
        await engine._release_gpu()
        
        assert engine._device == torch.device("cpu")


# ============================================================================
# Titan Engine GPU Tests
# ============================================================================

class TestTitanEngineGPU:
    """Tests for Titan Engine GPU support"""
    
    def test_titan_engine_initialization_with_gpu_client(self, mock_gpu_client):
        """Test Titan engine accepts GPU client"""
        from engines.titan_engine import TitanEngine
        
        engine = TitanEngine(gpu_client=mock_gpu_client)
        assert engine._gpu_client is mock_gpu_client
    
    def test_titan_config_includes_use_gpu(self):
        """Test Titan config schema includes use_gpu option"""
        from engines.titan_engine import TITAN_CONFIG_SCHEMA
        
        assert 'use_gpu' in TITAN_CONFIG_SCHEMA
        assert TITAN_CONFIG_SCHEMA['use_gpu']['type'] == 'bool'
        assert TITAN_CONFIG_SCHEMA['use_gpu']['default'] is True
    
    def test_titan_model_configs_with_gpu(self):
        """Test model configs include XGBoost GPU when use_gpu=True"""
        from engines.titan_engine import _create_model_configs, XGBOOST_AVAILABLE
        
        if XGBOOST_AVAILABLE:
            configs = _create_model_configs(use_gpu=True)
            
            # Should have XGBoost-GPU config
            xgb_configs = [c for c in configs if 'XGBoost' in c['name']]
            assert len(xgb_configs) >= 1
            
            xgb_gpu = [c for c in xgb_configs if c.get('gpu', False)]
            if XGBOOST_AVAILABLE:
                assert len(xgb_gpu) >= 1
    
    def test_titan_model_configs_without_gpu(self):
        """Test model configs use CPU XGBoost when use_gpu=False"""
        from engines.titan_engine import _create_model_configs, XGBOOST_AVAILABLE
        
        configs = _create_model_configs(use_gpu=False)
        
        # Check no configs have gpu=True
        gpu_configs = [c for c in configs if c.get('gpu', False)]
        assert len(gpu_configs) == 0


# ============================================================================
# GPU Fallback Tests
# ============================================================================

class TestGPUFallback:
    """Tests for graceful GPU fallback to CPU"""
    
    @pytest.mark.asyncio
    async def test_mirror_falls_back_to_cpu(self, sample_dataframe, mock_gpu_client_unavailable):
        """Test Mirror engine falls back to CPU when GPU unavailable"""
        from engines.mirror_engine import MirrorEngine
        
        engine = MirrorEngine(gpu_client=mock_gpu_client_unavailable)
        
        # Should not raise error
        result = await engine.analyze_async(sample_dataframe, {'epochs': 5, 'num_rows': 10})
        
        # Should indicate CPU was used
        assert result.get('used_gpu', False) is False
    
    @pytest.mark.asyncio
    async def test_galileo_falls_back_to_cpu(self, sample_dataframe, mock_gpu_client_unavailable):
        """Test Galileo engine falls back to CPU when GPU unavailable"""
        from engines.galileo_engine import GalileoEngine
        
        engine = GalileoEngine(gpu_client=mock_gpu_client_unavailable)
        
        result = await engine.analyze_async(sample_dataframe, {'epochs': 5})
        
        assert result.get('used_gpu', False) is False


# ============================================================================
# GPU Cleanup Tests
# ============================================================================

class TestGPUCleanup:
    """Tests for proper GPU resource cleanup"""
    
    @pytest.mark.asyncio
    async def test_mirror_releases_gpu_on_success(self, sample_dataframe, mock_gpu_client):
        """Test Mirror engine releases GPU after successful analysis"""
        from engines.mirror_engine import MirrorEngine, CUDA_AVAILABLE
        
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        
        engine = MirrorEngine(gpu_client=mock_gpu_client)
        
        # Run analysis
        await engine.analyze_async(sample_dataframe, {'epochs': 5, 'num_rows': 10})
        
        # GPU should have been released
        if mock_gpu_client.request_gpu.called:
            mock_gpu_client.release_gpu.assert_called()
    
    @pytest.mark.asyncio
    async def test_mirror_releases_gpu_on_error(self, mock_gpu_client):
        """Test Mirror engine releases GPU even on error"""
        from engines.mirror_engine import MirrorEngine, CUDA_AVAILABLE
        
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        
        engine = MirrorEngine(gpu_client=mock_gpu_client)
        
        # Create invalid data that will cause error
        bad_df = pd.DataFrame()
        
        try:
            await engine.analyze_async(bad_df, {'epochs': 5})
        except Exception:
            pass
        
        # State should be reset regardless
        assert engine._gpu_acquired is False


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
