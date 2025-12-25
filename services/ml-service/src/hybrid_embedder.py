"""
Hybrid Embedding Generator
==========================

Creates multi-dimensional embeddings combining:
- Semantic embeddings (sentence-transformers on raw text)
- Contextual embeddings (sentence-transformers on Gemma-generated descriptions)
- Temporal embeddings (date/time cyclical features)

This hybrid approach captures:
1. What was said (raw semantic)
2. What it means in context (Gemma interpretation)
3. When it happened (temporal awareness)

Author: NeMo Server Team
Version: 1.0.0
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# Lazy imports to avoid loading heavy models at import time
_sentence_transformer_model = None
_model_name = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

logger = logging.getLogger(__name__)


def get_sentence_transformer():
    """Lazy load sentence-transformer model."""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformer model: {_model_name}")
            _sentence_transformer_model = SentenceTransformer(_model_name)
            logger.info("Sentence-transformer model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    return _sentence_transformer_model


@dataclass
class EmbeddingConfig:
    """Configuration for hybrid embedding generation."""
    
    # Weights for combining different embedding types
    raw_weight: float = 0.35          # Direct text meaning
    context_weight: float = 0.55       # Gemma contextual interpretation
    temporal_weight: float = 0.10      # Temporal features
    
    # Dimensions
    raw_embedding_dim: int = 384       # all-MiniLM-L6-v2 output
    context_embedding_dim: int = 384   # Same model for context
    temporal_embedding_dim: int = 32   # From TemporalEncoder
    
    # Options
    normalize_embeddings: bool = True
    include_metadata_in_context: bool = True
    
    @property
    def total_dim(self) -> int:
        """Total dimension of hybrid embedding."""
        return (
            self.raw_embedding_dim +
            self.context_embedding_dim +
            self.temporal_embedding_dim
        )
    
    @property
    def weighted_total_dim(self) -> int:
        """Dimension when using weighted average (not concatenation)."""
        # When using weighted average, we match to largest component
        return max(
            self.raw_embedding_dim,
            self.context_embedding_dim
        )


@dataclass
class Section:
    """
    A section of transcription with associated metadata.
    
    This represents a chunk of text that will be vectorized
    along with its contextual enrichments.
    """
    
    index: int
    text: str
    gemma_context: str = ""           # Gemma-generated context description
    speaker: str = "unknown"
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    
    # Generated embeddings (populated by HybridEmbedder)
    raw_embedding: np.ndarray | None = None
    context_embedding: np.ndarray | None = None
    temporal_embedding: np.ndarray | None = None
    hybrid_embedding: np.ndarray | None = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "index": self.index,
            "text": self.text,
            "gemma_context": self.gemma_context,
            "speaker": self.speaker,
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
            "has_hybrid_embedding": self.hybrid_embedding is not None,
            "metadata": self.metadata,
        }


class HybridEmbedder:
    """
    Generates hybrid embeddings combining semantic and temporal features.
    
    The hybrid embedding captures:
    - Raw semantic meaning from the transcription text
    - Rich contextual meaning from Gemma's interpretation
    - Temporal patterns from when the recording was made
    
    Usage:
        embedder = HybridEmbedder()
        
        # Generate embedding for a section
        hybrid = embedder.generate_hybrid_embedding(
            raw_text="We need to accelerate the Q4 timeline",
            gemma_context="Discussion of acceleration strategy for Q4 deliverables, 
                           expressing urgency due to competitive pressure from recent 
                           market entrant. Decision to reallocate resources from R&D.",
            temporal_features=temporal_encoder.encode(context)
        )
    """
    
    def __init__(self, config: EmbeddingConfig | None = None):
        """
        Initialize embedder with configuration.
        
        Args:
            config: EmbeddingConfig instance. If None, uses defaults.
        """
        self.config = config or EmbeddingConfig()
        self._model = None  # Lazy loaded
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = get_sentence_transformer()
        return self._model
    
    def generate_hybrid_embedding(
        self,
        raw_text: str,
        gemma_context: str,
        temporal_features: np.ndarray,
        combination_mode: str = "concatenate"
    ) -> np.ndarray:
        """
        Generate a combined embedding vector.
        
        Args:
            raw_text: Original transcription text
            gemma_context: Gemma-generated contextual description
            temporal_features: Pre-computed temporal feature vector
            combination_mode: How to combine embeddings:
                - "concatenate": Stack all features (800d output)
                - "weighted_average": Weighted combination (384d output)
                - "hybrid": Concatenate semantic, add temporal (768d + 32d)
        
        Returns:
            numpy array of combined embedding
        """
        # Generate semantic embeddings
        raw_embedding = self._encode_text(raw_text)
        
        # Generate context embedding (from Gemma's interpretation)
        if gemma_context and len(gemma_context.strip()) > 10:
            context_embedding = self._encode_text(gemma_context)
        else:
            # Fallback: use raw text if no Gemma context
            context_embedding = raw_embedding.copy()
        
        # Ensure temporal features are correct size
        if len(temporal_features) != self.config.temporal_embedding_dim:
            temporal_features = self._resize_features(
                temporal_features,
                self.config.temporal_embedding_dim
            )
        
        # Combine based on mode
        if combination_mode == "concatenate":
            combined = self._concatenate_embeddings(
                raw_embedding,
                context_embedding,
                temporal_features
            )
        elif combination_mode == "weighted_average":
            combined = self._weighted_average_embeddings(
                raw_embedding,
                context_embedding,
                temporal_features
            )
        elif combination_mode == "hybrid":
            combined = self._hybrid_combine(
                raw_embedding,
                context_embedding,
                temporal_features
            )
        else:
            raise ValueError(f"Unknown combination_mode: {combination_mode}")
        
        # Optionally normalize
        if self.config.normalize_embeddings:
            combined = self._normalize(combined)
        
        return combined
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using sentence-transformer."""
        if not text or len(text.strip()) < 3:
            return np.zeros(self.config.raw_embedding_dim, dtype=np.float32)
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False  # We normalize at the end
        )
        
        return embedding.astype(np.float32)
    
    def _concatenate_embeddings(
        self,
        raw_emb: np.ndarray,
        ctx_emb: np.ndarray,
        temp_emb: np.ndarray
    ) -> np.ndarray:
        """
        Concatenate all embeddings into a single vector.
        
        Output: (raw_dim + context_dim + temporal_dim) dimensions
        Default: 384 + 384 + 32 = 800 dimensions
        """
        # Apply weights before concatenation for relative importance
        weighted_raw = raw_emb * self.config.raw_weight
        weighted_ctx = ctx_emb * self.config.context_weight
        weighted_temp = temp_emb * self.config.temporal_weight
        
        return np.concatenate([weighted_raw, weighted_ctx, weighted_temp])
    
    def _weighted_average_embeddings(
        self,
        raw_emb: np.ndarray,
        ctx_emb: np.ndarray,
        temp_emb: np.ndarray
    ) -> np.ndarray:
        """
        Compute weighted average of semantic embeddings.
        
        Temporal features are projected and added.
        Output: 384 dimensions (matching sentence-transformer)
        """
        # Weighted average of semantic embeddings
        semantic_combined = (
            raw_emb * self.config.raw_weight +
            ctx_emb * self.config.context_weight
        )
        
        # Project temporal to match semantic dimension
        temp_projected = np.zeros(len(raw_emb), dtype=np.float32)
        temp_projected[:len(temp_emb)] = temp_emb * self.config.temporal_weight
        
        # Combine
        return semantic_combined + temp_projected
    
    def _hybrid_combine(
        self,
        raw_emb: np.ndarray,
        ctx_emb: np.ndarray,
        temp_emb: np.ndarray
    ) -> np.ndarray:
        """
        Hybrid: weighted semantic + concatenated temporal.
        
        Output: 384 + 32 = 416 dimensions
        """
        # Weighted average of semantic embeddings
        semantic_weight_sum = self.config.raw_weight + self.config.context_weight
        semantic_combined = (
            raw_emb * (self.config.raw_weight / semantic_weight_sum) +
            ctx_emb * (self.config.context_weight / semantic_weight_sum)
        )
        
        # Concatenate temporal
        return np.concatenate([semantic_combined, temp_emb])
    
    def _resize_features(
        self,
        features: np.ndarray,
        target_size: int
    ) -> np.ndarray:
        """Resize feature vector to target size."""
        if len(features) == target_size:
            return features
        elif len(features) > target_size:
            return features[:target_size]
        else:
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(features)] = features
            return padded
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def embed_section(
        self,
        section: Section,
        temporal_features: np.ndarray,
        combination_mode: str = "concatenate"
    ) -> Section:
        """
        Generate hybrid embedding for a section in-place.
        
        Args:
            section: Section object with text and gemma_context
            temporal_features: Pre-computed temporal features
            combination_mode: How to combine embeddings
            
        Returns:
            Same section with embeddings populated
        """
        # Generate component embeddings
        section.raw_embedding = self._encode_text(section.text)
        section.context_embedding = self._encode_text(section.gemma_context)
        section.temporal_embedding = temporal_features.copy()
        
        # Generate hybrid embedding
        section.hybrid_embedding = self.generate_hybrid_embedding(
            raw_text=section.text,
            gemma_context=section.gemma_context,
            temporal_features=temporal_features,
            combination_mode=combination_mode
        )
        
        return section
    
    def embed_sections(
        self,
        sections: list[Section],
        temporal_features: np.ndarray,
        combination_mode: str = "concatenate"
    ) -> list[Section]:
        """
        Generate hybrid embeddings for multiple sections.
        
        Uses batched encoding for efficiency.
        
        Args:
            sections: List of Section objects
            temporal_features: Shared temporal features for all sections
            combination_mode: How to combine embeddings
            
        Returns:
            List of sections with embeddings populated
        """
        if not sections:
            return []
        
        # Batch encode all raw texts
        raw_texts = [s.text for s in sections]
        raw_embeddings = self.model.encode(
            raw_texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False
        )
        
        # Batch encode all Gemma contexts
        context_texts = [
            s.gemma_context if s.gemma_context else s.text
            for s in sections
        ]
        context_embeddings = self.model.encode(
            context_texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False
        )
        
        # Combine for each section
        for i, section in enumerate(sections):
            section.raw_embedding = raw_embeddings[i].astype(np.float32)
            section.context_embedding = context_embeddings[i].astype(np.float32)
            section.temporal_embedding = temporal_features.copy()
            
            # Generate hybrid
            if combination_mode == "concatenate":
                combined = self._concatenate_embeddings(
                    section.raw_embedding,
                    section.context_embedding,
                    temporal_features
                )
            elif combination_mode == "weighted_average":
                combined = self._weighted_average_embeddings(
                    section.raw_embedding,
                    section.context_embedding,
                    temporal_features
                )
            elif combination_mode == "hybrid":
                combined = self._hybrid_combine(
                    section.raw_embedding,
                    section.context_embedding,
                    temporal_features
                )
            else:
                combined = self._concatenate_embeddings(
                    section.raw_embedding,
                    section.context_embedding,
                    temporal_features
                )
            
            if self.config.normalize_embeddings:
                combined = self._normalize(combined)
            
            section.hybrid_embedding = combined
        
        return sections


# =============================================================================
# Similarity Functions for Hybrid Embeddings
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def find_similar_sections(
    query_embedding: np.ndarray,
    section_embeddings: list[np.ndarray],
    top_k: int = 5,
    metric: str = "cosine"
) -> list[tuple[int, float]]:
    """
    Find the most similar sections to a query.
    
    Args:
        query_embedding: Query vector
        section_embeddings: List of section vectors
        top_k: Number of results to return
        metric: Similarity metric ("cosine" or "euclidean")
        
    Returns:
        List of (index, score) tuples, sorted by similarity
    """
    scores = []
    
    for i, section_emb in enumerate(section_embeddings):
        if metric == "cosine":
            score = cosine_similarity(query_embedding, section_emb)
        else:
            # For euclidean, lower is better, so we negate
            score = -euclidean_distance(query_embedding, section_emb)
        
        scores.append((i, score))
    
    # Sort by score (higher is better)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_k]


# =============================================================================
# Singleton Embedder Instance
# =============================================================================

_hybrid_embedder: HybridEmbedder | None = None


def get_hybrid_embedder(config: EmbeddingConfig | None = None) -> HybridEmbedder:
    """Get or create the hybrid embedder singleton."""
    global _hybrid_embedder
    if _hybrid_embedder is None:
        _hybrid_embedder = HybridEmbedder(config)
    return _hybrid_embedder
