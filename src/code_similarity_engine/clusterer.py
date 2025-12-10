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
Vector clusterer - groups similar code embeddings.

Uses Agglomerative Clustering with cosine distance for
deterministic, density-aware clustering.
"""

from typing import List, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from .models import CodeChunk, Cluster


def cluster_vectors(
    vectors: np.ndarray,
    chunks: List[CodeChunk],
    threshold: float = 0.80,
    min_cluster_size: int = 2,
    verbose: bool = False,
) -> List[Cluster]:
    """
    Cluster code vectors by semantic similarity.

    Args:
        vectors: Embedding vectors (n_chunks, 384)
        chunks: Corresponding CodeChunk objects
        threshold: Similarity threshold (0.0-1.0)
        min_cluster_size: Minimum members per cluster
        verbose: Print progress

    Returns:
        List of Cluster objects, sorted by similarity (highest first)
    """
    if len(vectors) < 2:
        return []

    # Convert similarity threshold to distance threshold
    # Cosine distance = 1 - cosine similarity
    distance_threshold = 1.0 - threshold

    # Run agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",  # Average linkage handles varying cluster sizes well
    )

    labels = clustering.fit_predict(vectors)

    # Group chunks by cluster label
    cluster_map: dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:  # Noise (shouldn't happen with agglomerative)
            continue
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(idx)

    # Filter by minimum size
    cluster_map = {k: v for k, v in cluster_map.items() if len(v) >= min_cluster_size}

    if verbose:
        print(f"   Raw clusters: {len(set(labels))}, filtered: {len(cluster_map)}")

    # Build Cluster objects with similarity scores
    clusters = []
    for cluster_id, indices in cluster_map.items():
        cluster_chunks = [chunks[i] for i in indices]
        cluster_vectors = vectors[indices]

        # Calculate average pairwise similarity
        similarity = _calculate_cluster_similarity(cluster_vectors)

        # Find centroid (most representative chunk)
        centroid_idx = _find_centroid_idx(cluster_vectors)

        clusters.append(Cluster(
            id=cluster_id,
            chunks=cluster_chunks,
            similarity_score=similarity,
            centroid_idx=centroid_idx,
        ))

    # Sort by similarity (most similar first)
    clusters.sort(key=lambda c: c.similarity_score, reverse=True)

    # Renumber cluster IDs to be sequential
    for i, cluster in enumerate(clusters):
        cluster.id = i + 1

    return clusters


def _calculate_cluster_similarity(vectors: np.ndarray) -> float:
    """Calculate average pairwise cosine similarity within a cluster."""
    if len(vectors) < 2:
        return 1.0

    # Compute pairwise similarities
    sim_matrix = cosine_similarity(vectors)

    # Get upper triangle (excluding diagonal)
    n = len(vectors)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]

    return float(np.mean(upper_tri))


def _find_centroid_idx(vectors: np.ndarray) -> int:
    """Find the vector closest to the centroid (most representative)."""
    if len(vectors) == 1:
        return 0

    # Compute centroid
    centroid = np.mean(vectors, axis=0, keepdims=True)

    # Find closest vector to centroid
    similarities = cosine_similarity(centroid, vectors)[0]
    return int(np.argmax(similarities))
