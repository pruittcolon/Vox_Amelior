"""
Galileo Graph Neural Network Engine - Relational Learning

Discovers patterns in graph-structured data using Graph Neural Networks.

Core technique: Graph Convolutional Networks (GCN)
- Learns from node features and graph structure
- Captures relational patterns
- Applications: social networks, molecular analysis, recommendation systems

GPU Support:
- GCN training can leverage CUDA for faster computation
- Automatically coordinates GPU access via GPU Coordinator

Author: Enterprise Analytics Team
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import asyncio
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Optional torch_geometric import - may not be installed
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    GCNConv = None
    Data = None

from sklearn.preprocessing import LabelEncoder, StandardScaler

from core import (
    ColumnProfiler,
    ColumnMapper,
    SemanticType,
    BusinessEntity,
    EngineRequirements,
    DatasetClassifier
)

# Import premium models
from core.premium_models import (
    PremiumResult, Variant, PlainEnglishSummary, TechnicalExplanation,
    ExplanationStep, FeatureImportance, ConfigParameter, TaskType, Confidence
)

# GPU Client for coordinated GPU access
try:
    from core.gpu_client import get_gpu_client, GPUClient
    GPU_CLIENT_AVAILABLE = True
except ImportError:
    GPU_CLIENT_AVAILABLE = False

GALILEO_CONFIG_SCHEMA = {
    "hidden_dim": {"type": "int", "min": 8, "max": 128, "default": 16, "description": "Hidden layer dimension"},
    "epochs": {"type": "int", "min": 50, "max": 500, "default": 200, "description": "Training epochs"},
    "similarity_threshold": {"type": "float", "min": 0.1, "max": 0.9, "default": 0.5, "description": "Edge creation threshold"},
    "use_gpu": {"type": "bool", "default": True, "description": "Use GPU acceleration if available"},
}


class GCN(torch.nn.Module):
    """Simple 2-layer Graph Convolutional Network"""
    def __init__(self, num_features, num_classes, hidden_dim=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GalileoEngine:
    """
    Galileo Graph Neural Network Engine: Relational Pattern Discovery
    
    Uses Graph Convolutional Networks to learn from both node features
    and graph structure simultaneously.
    
    Key insight: Relationships matter as much as attributes
    
    GPU Support:
    - Automatically requests GPU access via GPU Coordinator
    - Falls back to CPU if GPU unavailable
    - GPU significantly accelerates GNN training
    """
    
    def __init__(self, gpu_client: Optional['GPUClient'] = None):
        self.name = "Galileo Graph Neural Network Engine"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        self._gpu_client = gpu_client
        self._gpu_acquired = False
        self._device = torch.device("cpu")  # Default to CPU
        
    @property
    def gpu_client(self) -> Optional['GPUClient']:
        """Get GPU client, initializing from singleton if not provided"""
        if self._gpu_client is None and GPU_CLIENT_AVAILABLE:
            try:
                self._gpu_client = get_gpu_client()
            except Exception:
                pass
        return self._gpu_client
        
    def get_requirements(self) -> EngineRequirements:
        """Define what this engine needs to run"""
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=['graph_learning', 'node_classification'],
            min_rows=20
        )
    
    async def _acquire_gpu(self) -> bool:
        """
        Request GPU access from the GPU Coordinator.
        
        Returns:
            bool: True if GPU was acquired, False if falling back to CPU
        """
        if not CUDA_AVAILABLE:
            return False
            
        if self.gpu_client is None:
            return False
            
        try:
            result = await self.gpu_client.request_gpu(engine_name="galileo")
            self._gpu_acquired = result.acquired
            if result.acquired:
                self._device = torch.device("cuda")
            return result.acquired
        except Exception as e:
            print(f"GPU acquisition failed, using CPU: {e}")
            return False
    
    async def _release_gpu(self) -> None:
        """Release GPU back to the coordinator"""
        if self._gpu_acquired and self.gpu_client is not None:
            try:
                await self.gpu_client.release_gpu()
            except Exception:
                pass
            finally:
                self._gpu_acquired = False
                self._device = torch.device("cpu")
    
    def analyze(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main analysis method (synchronous)
        
        Args:
            df: Input dataset
            config: {
                'target_column': Optional[str] - node labels
                'edge_column': Optional[str] - defines edges
                'epochs': int - training epochs (default: 50)
                'use_gpu': bool - whether to attempt GPU acceleration (default: True)
            }
        
        Returns:
            {
                'accuracy': float,
                'node_embeddings': np.ndarray,
                'insights': [...]
            }
        """
        config = config or {}
        
        # Check if torch_geometric is available
        if not TORCH_GEOMETRIC_AVAILABLE:
            return {
                'engine': self.name,
                'status': 'not_available',
                'error': 'torch_geometric is not installed - GNN functionality unavailable',
                'message': 'torch_geometric is not installed - GNN functionality unavailable',
                'insights': ['📊 **Galileo Engine**: torch_geometric package not installed.'],
                'used_gpu': False
            }
        
        # For sync context, use CPU (GPU requires async coordination)
        gpu_acquired = False
        
        # 1. Detect target column
        target_col = self._detect_target_column(df, config.get('target_column'))
        
        if not target_col:
            return {
                'engine': self.name,
                'status': 'not_applicable',
                'insights': ['📊 **Galileo Engine - Not Applicable**: No suitable target column detected.'],
                'used_gpu': False
            }
        
        # 2. Build graph from data
        graph_data, feature_names = self._build_graph(df, target_col)
        
        if graph_data is None:
            return {
                'error': 'Graph construction failed',
                'message': 'Could not build graph from data',
                'insights': [],
                'used_gpu': False
            }
        
        # 3. Train GNN
        epochs = config.get('epochs', 50)
        model, accuracy, embeddings = self._train_gnn(graph_data, epochs, gpu_acquired)
        
        # 4. Generate insights
        insights = self._generate_insights(
            df, target_col, feature_names, accuracy, embeddings, epochs, gpu_used=gpu_acquired
        )
        
        return {
            'engine': self.name,
            'nodes': graph_data.num_nodes,
            'edges': graph_data.num_edges,
            'features': len(feature_names),
            'accuracy': accuracy,
            'epochs_trained': epochs,
            'node_embeddings': embeddings,
            'insights': insights,
            'used_gpu': gpu_acquired
        }
    
    async def analyze_async(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Async main analysis method with GPU coordination
        
        Args:
            df: Input dataset
            config: {
                'target_column': Optional[str] - node labels
                'edge_column': Optional[str] - defines edges
                'epochs': int - training epochs (default: 50)
                'use_gpu': bool - whether to attempt GPU acceleration (default: True)
            }
        
        Returns:
            {
                'accuracy': float,
                'node_embeddings': np.ndarray,
                'insights': [...],
                'used_gpu': bool
            }
        """
        config = config or {}
        use_gpu = config.get('use_gpu', True)
        
        # Check if torch_geometric is available
        if not TORCH_GEOMETRIC_AVAILABLE:
            return {
                'engine': self.name,
                'status': 'not_available',
                'error': 'torch_geometric is not installed - GNN functionality unavailable',
                'message': 'torch_geometric is not installed - GNN functionality unavailable',
                'insights': ['📊 **Galileo Engine**: torch_geometric package not installed. Install with: pip install torch-geometric'],
                'used_gpu': False
            }
        
        # Attempt GPU acquisition if requested
        gpu_acquired = False
        if use_gpu and CUDA_AVAILABLE:
            gpu_acquired = await self._acquire_gpu()
        
        try:
            # 1. Detect target column
            target_col = self._detect_target_column(df, config.get('target_column'))
            
            if not target_col:
                return {
                    'engine': self.name,
                    'status': 'not_applicable',
                    'insights': ['📊 **Galileo Engine - Not Applicable**: This engine requires a target column (labels) for node classification. No suitable target column was detected.'],
                    'used_gpu': False
                }
            
            # 2. Build graph from data
            graph_data, feature_names = self._build_graph(df, target_col)
            
            if graph_data is None:
                return {
                    'error': 'Graph construction failed',
                    'message': 'Could not build graph from data',
                    'insights': [],
                    'used_gpu': False
                }
            
            # 3. Move data to GPU if available
            if gpu_acquired:
                graph_data = graph_data.to(self._device)
            
            # 4. Train GNN
            epochs = config.get('epochs', 50)
            model, accuracy, embeddings = self._train_gnn(graph_data, epochs, gpu_acquired)
            
            # 5. Generate insights
            insights = self._generate_insights(
                df, target_col, feature_names, accuracy, embeddings, epochs, gpu_used=gpu_acquired
            )
            
            return {
                'engine': self.name,
                'nodes': graph_data.num_nodes,
                'edges': graph_data.num_edges,
                'features': len(feature_names),
                'accuracy': accuracy,
                'epochs_trained': epochs,
                'node_embeddings': embeddings,
                'insights': insights,
                'used_gpu': gpu_acquired
            }
        finally:
            if gpu_acquired:
                await self._release_gpu()
    
    def _detect_target_column(self, df: pd.DataFrame, hint: Optional[str]) -> Optional[str]:
        """Detect target column"""
        if hint and hint in df.columns:
            return hint
        
        # Common target keywords
        target_keywords = ['label', 'class', 'category', 'type', 'species']
        for col in df.columns:
            if any(kw in col.lower() for kw in target_keywords):
                return col
        
        # Last column if categorical
        last_col = df.columns[-1]
        if df[last_col].dtype == 'object' or df[last_col].nunique() < 20:
            return last_col
        
        return None
    
    def _build_graph(self, df: pd.DataFrame, target_col: str) -> tuple:
        """
        Build graph from tabular data
        Creates similarity-based edges from feature vectors
        """
        # Separate features and labels
        feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        
        if len(feature_cols) == 0:
            return None, []
        
        # Prepare features
        X = df[feature_cols].fillna(df[feature_cols].median()).values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Prepare labels - always encode to [0, n_classes) for node classification
        y = df[target_col].values
        le = LabelEncoder()
        y = le.fit_transform(y)  # Always encode - GCN requires labels in [0, num_classes)
        self._label_encoder = le  # Store for later decoding if needed
        self._num_classes = len(le.classes_)
        
        # Create edges based on k-nearest neighbors (simple similarity graph)
        from sklearn.neighbors import kneighbors_graph
        k = min(5, len(X) - 1)  # 5 neighbors or less
        knn_graph = kneighbors_graph(X, k, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
        
        # Create PyG Data object
        x = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Create train/test masks (80/20 split)
        num_nodes = len(X)
        train_size = int(0.8 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        indices = torch.randperm(num_nodes)
        train_mask[indices[:train_size]] = True
        test_mask[indices[train_size:]] = True
        
        data.train_mask = train_mask
        data.test_mask = test_mask
        
        return data, feature_cols
    
    def _train_gnn(self, data: Data, epochs: int, use_gpu: bool = False) -> tuple:
        """
        Train Graph Convolutional Network
        
        Args:
            data: PyTorch Geometric Data object
            epochs: Number of training epochs
            use_gpu: Whether GPU is being used (for device placement)
        """
        device = self._device
        
        # Move data to device
        data = data.to(device)
        
        num_features = data.num_features
        num_classes = len(torch.unique(data.y))
        
        # Create model and move to device
        model = GCN(num_features, num_classes, hidden_dim=16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=1)
            
            # Test accuracy
            test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            test_acc = int(test_correct) / int(data.test_mask.sum())
            
            # Get node embeddings (before final layer) - move to CPU for numpy
            embeddings = model.conv1(data.x, data.edge_index).cpu().detach().numpy()
        
        return model, test_acc, embeddings
    
    def _generate_insights(
        self, df: pd.DataFrame, target_col: str, features: List[str],
        accuracy: float, embeddings: np.ndarray, epochs: int,
        gpu_used: bool = False
    ) -> List[str]:
        """Generate business-friendly insights"""
        insights = []
        
        # Summary with GPU indicator
        gpu_indicator = " 🚀 (GPU-accelerated)" if gpu_used else ""
        insights.append(
            f"📊 **Galileo GNN Analysis Complete{gpu_indicator}**: Trained Graph Convolutional Network "
            f"for {epochs} epochs on graph with {len(df)} nodes."
        )
        
        # Graph structure
        avg_neighbors = 5  # k-NN parameter
        insights.append(
            f"🕸️ **Graph Structure**: Built similarity graph with ~{avg_neighbors} neighbors per node. "
            f"GNN learns from both node features ({len(features)}) and graph connections."
        )
        
        # Model performance
        if accuracy >= 0.9:
            grade = "Excellent"
            emoji = "✅"
        elif accuracy >= 0.7:
            grade = "Good"
            emoji = "👍"
        elif accuracy >= 0.5:
            grade = "Fair"
            emoji = "⚠️"
        else:
            grade = "Poor"
            emoji = "❌"
        
        insights.append(
            f"{emoji} **Node Classification Accuracy** ({grade}): {accuracy*100:.1f}%. "
            f"GNN successfully learned relational patterns in graph structure."
        )
        
        # Embedding quality
        embedding_dim = embeddings.shape[1]
        insights.append(
            f"🔬 **Node Embeddings**: Generated {embedding_dim}-dimensional representations "
            f"capturing both node features and graph topology. Use for clustering, "
            f"visualization, or downstream ML tasks."
        )
        
        # Business applications
        insights.append(
            "💡 **Use Cases**:"
        )
        insights.append("   • Social Network Analysis: Friend recommendations, community detection")
        insights.append("   • Fraud Detection: Identify suspicious transaction rings")
        insights.append("   • Molecular Analysis: Drug discovery, chemical property prediction")
        insights.append("   • Recommendation Systems: collaborative filtering with graph structure")
        
        # Strategic insight
        insights.append(
            "🎯 **Strategic Insight**: GNNs capture relational patterns that tabular ML misses. "
            "When entities are connected (users, molecules, documents), graph-based learning "
            "leverages these relationships for superior predictions."
        )
        
        return insights

    # =========================================================================
    # PREMIUM OUTPUT
    # =========================================================================
    
    def run_premium(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> PremiumResult:
        """Run Galileo GNN and return PremiumResult."""
        import time
        start_time = time.time()
        
        config = config or {}
        raw = self.analyze(df, config)
        
        if 'error' in raw:
            return self._error_to_premium_result(raw, df, config, start_time)
        
        accuracy = raw.get('accuracy', 0.0)
        nodes = raw.get('nodes', 0)
        edges = raw.get('edges', 0)
        used_gpu = raw.get('used_gpu', False)
        gpu_indicator = " (GPU)" if used_gpu else " (CPU)"
        
        variants = [Variant(
            rank=1, gemma_score=int(accuracy * 100), cv_score=accuracy,
            variant_type="graph_model", model_name=f"Graph Convolutional Network{gpu_indicator}",
            features_used=raw.get('feature_names', [])[:5],
            interpretation=f"GNN on {nodes} nodes, {edges} edges - {accuracy:.1%} accuracy",
            details={'nodes': nodes, 'edges': edges, 'accuracy': accuracy, 'used_gpu': used_gpu}
        )]
        
        features = [FeatureImportance(
            name=f, stability=80.0, importance=0.5, impact="positive",
            explanation=f"{f} used as node feature in GNN"
        ) for f in raw.get('feature_names', [])[:10]]
        
        # Add GPU note to training step
        training_step = "Trained 2-layer GCN"
        if used_gpu:
            training_step += " (GPU-accelerated)"
        
        return PremiumResult(
            engine_name="galileo", engine_display_name="Galileo Graph AI", engine_icon="🌐",
            task_type=TaskType.GRAPH_ANALYSIS, target_column=raw.get('target_column'),
            columns_analyzed=raw.get('feature_names', []), row_count=len(df),
            variants=variants, best_variant=variants[0], feature_importance=features,
            summary=PlainEnglishSummary(
                f"Graph learning achieved {accuracy:.1%} accuracy",
                f"Built graph with {nodes} nodes and {edges} edges. GNN captures relational patterns between records.",
                "Use node embeddings for downstream tasks.",
                Confidence.HIGH if accuracy > 0.7 else Confidence.MEDIUM
            ),
            explanation=TechnicalExplanation(
                "Graph Convolutional Network (GCN)",
                "https://pytorch-geometric.readthedocs.io/",
                [ExplanationStep(1, "Graph Construction", f"Created {edges} edges from feature similarity"),
                 ExplanationStep(2, "Feature Encoding", "Normalized node features"),
                 ExplanationStep(3, "GNN Training", training_step),
                 ExplanationStep(4, "Embedding", "Generated node representations")],
                ["Requires defining graph structure", "May not scale to very large graphs"]
            ),
            holdout=None, config_used=config,
            config_schema=[ConfigParameter(k, v['type'], v['default'], 
                          [v.get('min'), v.get('max')] if 'min' in v else None, 
                          v.get('description', '')) for k, v in GALILEO_CONFIG_SCHEMA.items()],
            execution_time_seconds=time.time() - start_time, warnings=[]
        )
    
    def _error_to_premium_result(self, raw, df, config, start):
        import time
        return PremiumResult(
            engine_name="galileo", engine_display_name="Galileo Graph AI", engine_icon="🌐",
            task_type=TaskType.GRAPH_ANALYSIS, target_column=None, columns_analyzed=list(df.columns),
            row_count=len(df), variants=[],
            best_variant=Variant(1, 0, 0.0, "error", "None", [], raw.get('message', 'Error'), {}),
            feature_importance=[],
            summary=PlainEnglishSummary("Graph analysis failed", raw.get('message', 'Error'), 
                                       "Check data format.", Confidence.LOW),
            explanation=TechnicalExplanation("GCN", None, [], ["Analysis failed"]),
            holdout=None, config_used=config, config_schema=[],
            execution_time_seconds=time.time() - start, warnings=[raw.get('message', '')]
        )


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python galileo_engine.py <csv_file> [--target COLUMN] [--epochs N]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    target_col = None
    epochs = 50
    
    if '--target' in sys.argv:
        target_idx = sys.argv.index('--target')
        if len(sys.argv) > target_idx + 1:
            target_col = sys.argv[target_idx + 1]
    
    if '--epochs' in sys.argv:
        epochs_idx = sys.argv.index('--epochs')
        if len(sys.argv) > epochs_idx + 1:
            epochs = int(sys.argv[epochs_idx + 1])
    
    df = pd.read_csv(csv_file)
    
    engine = GalileoEngine()
    config = {
        'target_column': target_col,
        'epochs': epochs
    }
    
    print(f"\n{'='*60}")
    print(f"GALILEO ENGINE: Training GNN for {epochs} epochs...")
    print(f"{'='*60}\n")
    
    result = engine.analyze(df, config)
    
    print(f"\n{'='*60}")
    print(f"GALILEO ENGINE RESULTS: {csv_file}")
    print(f"{'='*60}\n")
    
    if 'error' in result:
        print(f"❌ Error: {result['message']}")
        for insight in result.get('insights', []):
            print(f"   {insight}")
    else:
        for insight in result['insights']:
            print(insight)
        
        print(f"\n{'='*60}")
        print("GNN MODEL SUMMARY:")
        print(f"{'='*60}\n")
        print(f"Nodes: {result['nodes']}")
        print(f"Edges: {result['edges']}")
        print(f"Features: {result['features']}")
        print(f"Test Accuracy: {result['accuracy']*100:.1f}%")
        print(f"Embedding Dimensions: {result['node_embeddings'].shape[1]}")
