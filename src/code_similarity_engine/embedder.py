# Code Similarity Engine - Find and analyze duplicate code patterns
# Copyright (C) 2025  Jonathan Louis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Code embedder - converts code chunks into semantic vectors.

Uses Qwen3-Embedding via llama-cpp-python for fully offline operation.
Caches embeddings in SQLite to avoid re-embedding unchanged chunks.
"""

from typing import List, Optional
from pathlib import Path
import numpy as np

from .cache import (
    init_cache,
    get_cached_embedding,
    cache_embedding,
    get_chunk_hash,
)


def _get_embedding_model_from_manager() -> Optional[Path]:
    """Get embedding model path from model_manager."""
    try:
        from .model_manager import get_model_path
        return get_model_path("embedding")
    except ImportError:
        return None


def find_embedding_model(model_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find GGUF embedding model file.

    Args:
        model_path: Explicit path to model file

    Returns:
        Path to model file or None if not found
    """
    if model_path and Path(model_path).exists():
        return Path(model_path)

    # Use model_manager
    return _get_embedding_model_from_manager()


def embed_chunks(
    chunks: List["CodeChunk"],
    model_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    batch_size: int = 32,
    verbose: bool = False,
    quiet: bool = False,
) -> np.ndarray:
    """
    Embed code chunks using Qwen3-Embedding GGUF model.

    Uses SQLite cache to avoid re-embedding unchanged chunks.

    Args:
        chunks: List of CodeChunk objects
        model_path: Path to GGUF embedding model (auto-detected if None)
        project_root: Root directory for caching (defaults to cwd)
        batch_size: Chunks per progress update
        verbose: Print progress
        quiet: Suppress model loading messages

    Returns:
        NumPy array of shape (n_chunks, embedding_dim)

    Raises:
        ImportError: If llama-cpp-python is not installed
        FileNotFoundError: If no GGUF model found
    """
    from .models import CodeChunk

    # Find model first (needed for cache key even if all cached)
    model_file = find_embedding_model(model_path)
    if not model_file:
        raise FileNotFoundError(
            "No embedding model found. Download with:\n"
            "  cse --download-models"
        )

    # Model ID for cache invalidation when model changes
    model_id = model_file.stem  # e.g. "qwen3-embedding-0.6b-q8_0"

    # Initialize cache
    if project_root is None:
        project_root = Path.cwd()
    cache_conn = init_cache(project_root)

    # Check cache for each chunk
    all_embeddings = [None] * len(chunks)
    uncached_indices = []
    cache_hits = 0

    for i, chunk in enumerate(chunks):
        chunk_hash = get_chunk_hash(chunk.content)
        cached = get_cached_embedding(
            cache_conn,
            file_path=str(chunk.file_path),
            chunk_hash=chunk_hash,
            model_id=model_id,
        )
        if cached is not None:
            all_embeddings[i] = cached
            cache_hits += 1
        else:
            uncached_indices.append(i)

    if verbose:
        print(f"   Cache: {cache_hits}/{len(chunks)} hits ({len(uncached_indices)} to embed)")

    # If all cached, return early without loading model
    if not uncached_indices:
        if verbose and not quiet:
            print(f"   All embeddings loaded from cache!")
        embeddings = np.array(all_embeddings, dtype=np.float32)
        cache_conn.close()
        return embeddings

    # Load model only if we need to embed new chunks
    try:
        from llama_cpp import Llama
    except ImportError:
        cache_conn.close()
        raise ImportError(
            "llama-cpp-python not installed. Install with:\n"
            "  pip install 'code-similarity-engine[llm]'"
        )

    if verbose and not quiet:
        print(f"   Loading embedding model: {model_file.name}")

    # Load model in embedding mode
    llm = Llama(
        model_path=str(model_file),
        embedding=True,      # Enable embedding extraction
        n_ctx=512,           # Context size for embeddings
        n_threads=4,         # CPU threads
        n_gpu_layers=-1,     # Use GPU if available (Metal on Mac)
        verbose=False,       # Suppress llama.cpp output
    )

    if verbose and not quiet:
        print(f"   Model loaded successfully")

    # Embed only uncached chunks
    for progress_idx, i in enumerate(uncached_indices):
        chunk = chunks[i]
        text = _prepare_chunk_text(chunk)

        # Get embedding - llama.cpp returns list of floats
        embedding = np.array(llm.embed(text), dtype=np.float32)

        # L2 normalize individual embedding
        norm = np.linalg.norm(embedding)
        if norm > 1e-9:
            embedding = embedding / norm

        all_embeddings[i] = embedding

        # Save to cache
        chunk_hash = get_chunk_hash(chunk.content)
        cache_embedding(
            cache_conn,
            file_path=str(chunk.file_path),
            chunk_hash=chunk_hash,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            mtime=chunk.file_path.stat().st_mtime if chunk.file_path.exists() else 0,
            embedding=embedding,
            model_id=model_id,
        )

        # Progress reporting
        if verbose and (progress_idx + 1) % batch_size == 0:
            print(f"   Embedded {progress_idx + 1}/{len(uncached_indices)} new chunks...")

    if verbose and len(uncached_indices) % batch_size != 0:
        print(f"   Embedded {len(uncached_indices)}/{len(uncached_indices)} new chunks")

    cache_conn.close()

    # Stack all embeddings (cached embeddings are already normalized)
    embeddings = np.array(all_embeddings, dtype=np.float32)

    return embeddings


def _prepare_chunk_text(chunk: "CodeChunk") -> str:
    """
    Prepare a code chunk for embedding.

    Include metadata prefix to help the model understand context.
    """
    parts = []

    # Context hint
    if chunk.name:
        parts.append(f"# {chunk.language} {chunk.chunk_type}: {chunk.name}")
    else:
        parts.append(f"# {chunk.language} {chunk.chunk_type}")

    # Normalized code
    code = _normalize_code(chunk.content)
    parts.append(code)

    return "\n".join(parts)


def _normalize_code(code: str) -> str:
    """Normalize code for better embedding quality."""
    lines = code.split("\n")

    # Strip trailing whitespace
    lines = [line.rstrip() for line in lines]

    # Collapse runs of blank lines
    normalized = []
    prev_blank = False
    for line in lines:
        is_blank = len(line.strip()) == 0
        if is_blank and prev_blank:
            continue
        normalized.append(line)
        prev_blank = is_blank

    result = "\n".join(normalized)

    # Truncate if too long
    max_chars = 512 * 4  # ~512 tokens
    if len(result) > max_chars:
        result = result[:max_chars] + "\n# ... (truncated)"

    return result
