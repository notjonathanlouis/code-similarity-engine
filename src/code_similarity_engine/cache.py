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
Embedding cache for code-similarity-engine.

This module provides persistent caching of embeddings to avoid re-embedding
unchanged code chunks. Uses SQLite for storage and SHA-256 hashing for
content-based cache invalidation.
"""

import sqlite3
import hashlib
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np


CACHE_DIR = ".cse_cache"
CACHE_DB = "embeddings.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    file_path TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    mtime REAL,
    embedding BLOB,
    model_id TEXT,
    created_at REAL,
    PRIMARY KEY (file_path, chunk_hash)
);

CREATE INDEX IF NOT EXISTS idx_file_path ON embeddings(file_path);
CREATE INDEX IF NOT EXISTS idx_model_id ON embeddings(model_id);
"""


def get_cache_path(project_root: Path = None) -> Path:
    """
    Get path to cache database. Creates .cse_cache/ directory if needed.

    Args:
        project_root: Root directory for cache. Defaults to current working directory.

    Returns:
        Path to the cache database file.
    """
    if project_root is None:
        project_root = Path.cwd()

    cache_dir = project_root / CACHE_DIR
    cache_dir.mkdir(exist_ok=True)

    return cache_dir / CACHE_DB


def init_cache(project_root: Path = None) -> sqlite3.Connection:
    """
    Initialize cache database with schema.

    Args:
        project_root: Root directory for cache. Defaults to current working directory.

    Returns:
        SQLite connection object with initialized schema.
    """
    cache_path = get_cache_path(project_root)
    conn = sqlite3.connect(str(cache_path))

    # Execute schema creation
    conn.executescript(SCHEMA)
    conn.commit()

    return conn


def get_cached_embedding(
    conn: sqlite3.Connection,
    file_path: str,
    chunk_hash: str,
    model_id: str,
) -> Optional[np.ndarray]:
    """
    Get cached embedding if it exists and the model matches.

    Args:
        conn: SQLite connection object.
        file_path: Path to the source file.
        chunk_hash: SHA-256 hash of the chunk content.
        model_id: Identifier for the embedding model.

    Returns:
        Cached embedding as numpy array, or None if not found or model mismatch.
    """
    cursor = conn.execute(
        """
        SELECT embedding FROM embeddings
        WHERE file_path = ? AND chunk_hash = ? AND model_id = ?
        """,
        (file_path, chunk_hash, model_id)
    )

    row = cursor.fetchone()
    if row is None:
        return None

    # Deserialize embedding from bytes
    embedding_bytes = row[0]
    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

    return embedding


def cache_embedding(
    conn: sqlite3.Connection,
    file_path: str,
    chunk_hash: str,
    start_line: int,
    end_line: int,
    mtime: float,
    embedding: np.ndarray,
    model_id: str,
):
    """
    Store embedding in cache.

    Args:
        conn: SQLite connection object.
        file_path: Path to the source file.
        chunk_hash: SHA-256 hash of the chunk content.
        start_line: Starting line number of the chunk.
        end_line: Ending line number of the chunk.
        mtime: File modification time (Unix timestamp).
        embedding: Embedding vector as numpy array.
        model_id: Identifier for the embedding model.
    """
    # Serialize embedding to bytes
    embedding_bytes = embedding.astype(np.float32).tobytes()
    created_at = time.time()

    conn.execute(
        """
        INSERT OR REPLACE INTO embeddings
        (file_path, chunk_hash, start_line, end_line, mtime, embedding, model_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (file_path, chunk_hash, start_line, end_line, mtime,
         embedding_bytes, model_id, created_at)
    )
    conn.commit()


def get_chunk_hash(content: str) -> str:
    """
    Hash chunk content for cache key using SHA-256.

    Args:
        content: Text content of the code chunk.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def clear_cache(project_root: Path = None) -> bool:
    """
    Delete the .cse_cache/ directory and all its contents.

    Args:
        project_root: Root directory for cache. Defaults to current working directory.

    Returns:
        True if cache directory was deleted, False if it didn't exist.
    """
    if project_root is None:
        project_root = Path.cwd()

    cache_dir = project_root / CACHE_DIR

    if not cache_dir.exists():
        return False

    shutil.rmtree(cache_dir)
    return True


def get_cache_stats(project_root: Path = None) -> dict:
    """
    Return statistics about the cache.

    Args:
        project_root: Root directory for cache. Defaults to current working directory.

    Returns:
        Dictionary with keys:
        - total_entries: Total number of cached embeddings
        - total_size_mb: Size of cache database in megabytes
        - models_used: List of unique model IDs in cache
    """
    cache_path = get_cache_path(project_root)

    if not cache_path.exists():
        return {
            "total_entries": 0,
            "total_size_mb": 0.0,
            "models_used": []
        }

    # Get file size
    size_bytes = cache_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Query database for stats
    conn = sqlite3.connect(str(cache_path))

    # Total entries
    cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
    total_entries = cursor.fetchone()[0]

    # Unique models
    cursor = conn.execute("SELECT DISTINCT model_id FROM embeddings ORDER BY model_id")
    models_used = [row[0] for row in cursor.fetchall()]

    conn.close()

    return {
        "total_entries": total_entries,
        "total_size_mb": round(size_mb, 2),
        "models_used": models_used
    }


def invalidate_file(
    conn: sqlite3.Connection,
    file_path: str,
) -> int:
    """
    Remove all cached embeddings for a specific file.

    Useful when a file has been modified or deleted.

    Args:
        conn: SQLite connection object.
        file_path: Path to the source file.

    Returns:
        Number of embeddings removed from cache.
    """
    cursor = conn.execute(
        "DELETE FROM embeddings WHERE file_path = ?",
        (file_path,)
    )
    conn.commit()
    return cursor.rowcount


def invalidate_model(
    conn: sqlite3.Connection,
    model_id: str,
) -> int:
    """
    Remove all cached embeddings for a specific model.

    Useful when switching embedding models or when a model is updated.

    Args:
        conn: SQLite connection object.
        model_id: Identifier for the embedding model.

    Returns:
        Number of embeddings removed from cache.
    """
    cursor = conn.execute(
        "DELETE FROM embeddings WHERE model_id = ?",
        (model_id,)
    )
    conn.commit()
    return cursor.rowcount


def vacuum_cache(conn: sqlite3.Connection):
    """
    Optimize the cache database by reclaiming unused space.

    Should be called periodically after invalidation operations.

    Args:
        conn: SQLite connection object.
    """
    conn.execute("VACUUM")
    conn.commit()
