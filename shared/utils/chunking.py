"""
Document Chunking Utilities for RAG.

Provides multiple chunking strategies following 2024 best practices:
- Fixed-size with overlap
- Semantic chunking
- Recursive/hierarchical chunking
- Context-enriched chunking

References:
- https://stackoverflow.blog/chunking-strategies-rag
- https://medium.com/rag-chunking-best-practices
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Chunk:
    """A document chunk with metadata."""

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Character count of chunk content."""
        return len(self.content)


# =============================================================================
# FIXED-SIZE CHUNKING
# =============================================================================


def fixed_size_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    min_chunk_size: int = 50,
) -> list[Chunk]:
    """
    Split text into fixed-size chunks with overlap.

    Best for homogeneous datasets with consistent formatting.

    Args:
        text: Input text to chunk
        chunk_size: Target size per chunk in characters
        overlap: Number of overlapping characters between chunks
        min_chunk_size: Minimum size for final chunk

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size

        # Don't split in the middle of a word
        if end < len(text):
            # Find the last space before end
            space_pos = text.rfind(" ", start, end)
            if space_pos > start:
                end = space_pos

        chunk_content = text[start:end].strip()

        if len(chunk_content) >= min_chunk_size:
            chunks.append(
                Chunk(
                    content=chunk_content,
                    index=index,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1

        # Move start position with overlap
        start = end - overlap if end - overlap > start else end

        # Avoid infinite loop
        if start >= len(text) - min_chunk_size:
            break

    return chunks


# =============================================================================
# SENTENCE-BASED CHUNKING
# =============================================================================


def sentence_chunk(
    text: str,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Split text by sentences, grouping into chunks.

    Best for narrative content where sentence boundaries matter.

    Args:
        text: Input text
        sentences_per_chunk: Number of sentences per chunk
        overlap_sentences: Sentences to overlap between chunks

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []

    # Split into sentences (handles ., !, ?)
    sentence_pattern = r"(?<=[.!?])\s+"
    sentences = re.split(sentence_pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    start_idx = 0
    chunk_index = 0

    while start_idx < len(sentences):
        end_idx = min(start_idx + sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[start_idx:end_idx]
        chunk_content = " ".join(chunk_sentences)

        # Calculate character positions
        start_char = text.find(chunk_sentences[0]) if chunk_sentences else 0
        end_char = (
            text.find(chunk_sentences[-1]) + len(chunk_sentences[-1])
            if chunk_sentences
            else len(text)
        )

        chunks.append(
            Chunk(
                content=chunk_content,
                index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata={"sentence_count": len(chunk_sentences)},
            )
        )
        chunk_index += 1

        # Move with overlap
        start_idx = end_idx - overlap_sentences
        if start_idx >= end_idx:
            start_idx = end_idx

    return chunks


# =============================================================================
# PARAGRAPH-BASED CHUNKING
# =============================================================================


def paragraph_chunk(
    text: str,
    max_paragraphs: int = 3,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """
    Split text by paragraphs.

    Best for well-structured documents with clear paragraph breaks.

    Args:
        text: Input text
        max_paragraphs: Maximum paragraphs per chunk
        max_chunk_size: Maximum character size per chunk

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []

    # Split on double newlines or multiple newlines
    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_size = 0
    chunk_index = 0
    start_char = 0

    for para in paragraphs:
        para_size = len(para)

        # Check if adding this paragraph exceeds limits
        if (
            len(current_chunk) >= max_paragraphs
            or current_size + para_size > max_chunk_size
        ):
            if current_chunk:
                content = "\n\n".join(current_chunk)
                chunks.append(
                    Chunk(
                        content=content,
                        index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(content),
                        metadata={"paragraph_count": len(current_chunk)},
                    )
                )
                chunk_index += 1
                start_char += len(content) + 2  # Account for \n\n

            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size

    # Add remaining paragraphs
    if current_chunk:
        content = "\n\n".join(current_chunk)
        chunks.append(
            Chunk(
                content=content,
                index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(content),
                metadata={"paragraph_count": len(current_chunk)},
            )
        )

    return chunks


# =============================================================================
# RECURSIVE CHUNKING
# =============================================================================


def recursive_chunk(
    text: str,
    separators: Optional[list[str]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """
    Recursively split text using multiple separators.

    Tries larger separators first, then falls back to smaller ones.
    Best for structured documents with varying levels of organization.

    Args:
        text: Input text
        separators: List of separators in order of preference
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        List of Chunk objects
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if not text or not text.strip():
        return []

    def _split_recursive(
        text: str, seps: list[str], depth: int = 0
    ) -> list[str]:
        if not text:
            return []

        if len(text) <= chunk_size:
            return [text]

        if not seps:
            # No more separators, force split
            return fixed_size_chunk(text, chunk_size, chunk_overlap)

        sep = seps[0]
        remaining_seps = seps[1:]

        # Split by current separator
        parts = text.split(sep)

        result = []
        current = ""

        for part in parts:
            if len(current) + len(part) + len(sep) <= chunk_size:
                current = current + sep + part if current else part
            else:
                if current:
                    result.extend(_split_recursive(current, remaining_seps, depth + 1))
                current = part

        if current:
            result.extend(_split_recursive(current, remaining_seps, depth + 1))

        return result

    text_chunks = _split_recursive(text, separators)

    # Convert to Chunk objects
    chunks = []
    current_pos = 0

    for i, content in enumerate(text_chunks):
        if isinstance(content, Chunk):
            chunks.append(content)
        else:
            content = content.strip() if isinstance(content, str) else str(content)
            if content:
                chunks.append(
                    Chunk(
                        content=content,
                        index=i,
                        start_char=current_pos,
                        end_char=current_pos + len(content),
                    )
                )
                current_pos += len(content)

    return chunks


# =============================================================================
# CONTEXT-ENRICHED CHUNKING
# =============================================================================


def context_enriched_chunk(
    text: str,
    document_context: str,
    chunk_fn: Callable[[str], list[Chunk]] = None,
    context_template: str = "Document: {context}\n\n{content}",
) -> list[Chunk]:
    """
    Prepend document-level context to each chunk.

    Improves retrieval by including document metadata in each chunk.

    Args:
        text: Input text
        document_context: Summary or metadata about the document
        chunk_fn: Chunking function to use (defaults to fixed_size)
        context_template: Template for prepending context

    Returns:
        List of Chunk objects with context prepended
    """
    if chunk_fn is None:
        chunk_fn = lambda t: fixed_size_chunk(t, 400, 50)  # Smaller to leave room

    base_chunks = chunk_fn(text)

    enriched = []
    for chunk in base_chunks:
        enriched_content = context_template.format(
            context=document_context, content=chunk.content
        )
        enriched.append(
            Chunk(
                content=enriched_content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "has_context": True,
                    "original_length": chunk.length,
                },
            )
        )

    return enriched


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_optimal_chunk_size(text: str, target_chunks: int = 10) -> int:
    """
    Calculate optimal chunk size to get approximately target_chunks.

    Args:
        text: Input text
        target_chunks: Desired number of chunks

    Returns:
        Recommended chunk size
    """
    if not text:
        return 500

    text_length = len(text)
    return max(100, text_length // target_chunks)


def estimate_token_count(text: str, avg_chars_per_token: float = 4.0) -> int:
    """
    Estimate token count for LLM context limits.

    Args:
        text: Input text
        avg_chars_per_token: Average characters per token (GPT ~4)

    Returns:
        Estimated token count
    """
    return int(len(text) / avg_chars_per_token)
