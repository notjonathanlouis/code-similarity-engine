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
Reranker module for improving cluster quality using Qwen3-Reranker.

This module uses a cross-encoder reranking model to score similarity between
chunk pairs and filter out low-quality cluster members.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import logging

if TYPE_CHECKING:
    from .models import Cluster

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result of reranking a cluster."""
    original_size: int
    filtered_size: int
    kept_indices: List[int]
    scores: List[float]

    @property
    def filter_rate(self) -> float:
        """Percentage of chunks filtered out."""
        if self.original_size == 0:
            return 0.0
        return (1.0 - self.filtered_size / self.original_size) * 100


# Default model paths to check
DEFAULT_MODEL_PATHS = [
    Path.home() / ".cache" / "cse" / "models" / "Qwen3-Reranker-0.6B-q4_k_m.gguf",
    Path("/Volumes/APPLE-STORAGE/Tether/Tether/Resources/ML Models/Qwen3-Reranker-0.6B-q4_k_m.gguf"),
]


def find_reranker_model(model_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find reranker model file.

    Args:
        model_path: Explicit path to model file. If None, searches default locations.

    Returns:
        Path to model file if found, None otherwise.
    """
    # Check explicit path first
    if model_path is not None:
        if isinstance(model_path, str):
            model_path = Path(model_path)
        if model_path.exists():
            return model_path
        logger.warning(f"Specified model path does not exist: {model_path}")
        return None

    # Search default locations
    for path in DEFAULT_MODEL_PATHS:
        if path.exists():
            logger.info(f"Found reranker model at: {path}")
            return path

    logger.warning("No reranker model found in default locations:")
    for path in DEFAULT_MODEL_PATHS:
        logger.warning(f"  - {path}")

    return None


def _score_pair(llm: "Llama", query: str, document: str) -> float:
    """
    Score a query-document pair using the reranker.

    Args:
        llm: Loaded Llama model (reranker)
        query: Query text (representative chunk)
        document: Document text (chunk to score)

    Returns:
        Relevance score between 0.0 and 1.0
    """
    # Qwen3-Reranker prompt format
    prompt = (
        f"<|im_start|>user\n"
        f"Query: {query}\n"
        f"Document: {document}\n"
        f"Is this document relevant to the query?<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        # Generate response with logprobs to extract confidence
        output = llm(
            prompt,
            max_tokens=16,
            temperature=0.0,
            logprobs=1,
            echo=False,
        )

        # Extract first token and its probability
        if output and "choices" in output and len(output["choices"]) > 0:
            choice = output["choices"][0]

            # Try to parse logprobs for confidence
            if "logprobs" in choice and choice["logprobs"]:
                logprobs = choice["logprobs"]
                if "top_logprobs" in logprobs and len(logprobs["top_logprobs"]) > 0:
                    top = logprobs["top_logprobs"][0]

                    # Look for yes/no tokens and their probabilities
                    yes_tokens = ["yes", "Yes", "YES", "relevant", "Relevant"]
                    no_tokens = ["no", "No", "NO", "irrelevant", "Irrelevant"]

                    yes_score = 0.0
                    no_score = 0.0

                    for token, logprob in top.items():
                        prob = 2.718281828 ** logprob  # e^logprob
                        token_lower = token.lower().strip()

                        if any(yt.lower() in token_lower for yt in yes_tokens):
                            yes_score = max(yes_score, prob)
                        elif any(nt.lower() in token_lower for nt in no_tokens):
                            no_score = max(no_score, prob)

                    # Normalize to 0-1 range
                    if yes_score + no_score > 0:
                        return yes_score / (yes_score + no_score)

            # Fallback: parse text response
            text = choice.get("text", "").lower().strip()
            if any(word in text for word in ["yes", "relevant", "similar"]):
                return 0.75  # High confidence
            elif any(word in text for word in ["no", "irrelevant", "different"]):
                return 0.25  # Low confidence

        # Default to neutral score if parsing fails
        logger.debug("Could not parse reranker output, defaulting to 0.5")
        return 0.5

    except Exception as e:
        logger.warning(f"Error scoring pair: {e}")
        return 0.5


def rerank_cluster(
    cluster: "Cluster",
    model_path: Optional[Path] = None,
    threshold: float = 0.5,
    verbose: bool = False,
) -> Tuple["Cluster", RerankResult]:
    """
    Rerank cluster members by similarity to representative chunk.

    Args:
        cluster: Cluster object with chunks
        model_path: Path to reranker GGUF. If None, searches default locations.
        threshold: Minimum score to keep (0.0-1.0)
        verbose: Print progress

    Returns:
        Tuple of (refined_cluster, rerank_result)

    Raises:
        ImportError: If llama-cpp-python is not installed
        FileNotFoundError: If model file cannot be found
    """
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python is required for reranking. "
            "Install with: pip install llama-cpp-python"
        )

    # Find model
    model_path = find_reranker_model(model_path)
    if model_path is None:
        raise FileNotFoundError(
            "Reranker model not found. Please download Qwen3-Reranker-0.6B-q4_k_m.gguf"
            f"to one of: {DEFAULT_MODEL_PATHS}"
        )

    if verbose:
        print(f"Loading reranker model: {model_path}")

    # Load model
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        logits_all=True,  # Required for logprobs in _score_pair
        verbose=False,
    )

    # Get representative chunk as query
    rep_chunk = cluster.representative
    query = rep_chunk.content

    if verbose:
        print(f"Reranking cluster with {len(cluster.chunks)} chunks")

    # Score each chunk
    scores = []
    for i, chunk in enumerate(cluster.chunks):
        if chunk is rep_chunk:
            # Representative chunk always gets perfect score
            scores.append(1.0)
        else:
            score = _score_pair(llm, query, chunk.content)
            scores.append(score)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Scored {i + 1}/{len(cluster.chunks)} chunks")

    # Filter by threshold, but keep at least 2 chunks
    original_size = len(cluster.chunks)
    kept_indices = [i for i, score in enumerate(scores) if score >= threshold]

    # Ensure minimum cluster size
    if len(kept_indices) < 2:
        # Keep top 2 scoring chunks
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        kept_indices = sorted_indices[:2]

    # Create filtered cluster (preserve id and similarity from original)
    from .models import Cluster
    filtered_chunks = [cluster.chunks[i] for i in kept_indices]
    filtered_cluster = Cluster(
        id=cluster.id,
        chunks=filtered_chunks,
        centroid_idx=0,  # Reset centroid to first chunk
        similarity_score=cluster.similarity_score,
    )

    # Create result
    result = RerankResult(
        original_size=original_size,
        filtered_size=len(kept_indices),
        kept_indices=kept_indices,
        scores=scores,
    )

    if verbose:
        print(f"  Kept {result.filtered_size}/{result.original_size} chunks "
              f"({result.filter_rate:.1f}% filtered)")
        print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")

    return filtered_cluster, result


def rerank_clusters(
    clusters: List["Cluster"],
    model_path: Optional[Path] = None,
    threshold: float = 0.5,
    verbose: bool = False,
    quiet: bool = False,
) -> List[Tuple["Cluster", RerankResult]]:
    """
    Rerank multiple clusters. Load model once for efficiency.

    Args:
        clusters: List of Cluster objects
        model_path: Path to reranker GGUF. If None, searches default locations.
        threshold: Minimum score to keep (0.0-1.0)
        verbose: Print progress
        quiet: Suppress model loading messages

    Returns:
        List of (refined_cluster, rerank_result) tuples

    Raises:
        ImportError: If llama-cpp-python is not installed
        FileNotFoundError: If model file cannot be found
    """
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python is required for reranking. "
            "Install with: pip install llama-cpp-python"
        )

    # Find model
    model_path = find_reranker_model(model_path)
    if model_path is None:
        raise FileNotFoundError(
            "Reranker model not found. Please download Qwen3-Reranker-0.6B-q4_k_m.gguf"
            f"to one of: {DEFAULT_MODEL_PATHS}"
        )

    if verbose and not quiet:
        print(f"Loading reranker model: {model_path}")

    # Load model once
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        logits_all=True,  # Required for logprobs in _score_pair
        verbose=False,
    )

    results = []

    for cluster_idx, cluster in enumerate(clusters):
        if verbose:
            print(f"\nCluster {cluster_idx + 1}/{len(clusters)}: {len(cluster.chunks)} chunks")

        # Get representative chunk as query
        rep_chunk = cluster.representative
        query = rep_chunk.content

        # Score each chunk
        scores = []
        for chunk in cluster.chunks:
            if chunk is rep_chunk:
                scores.append(1.0)
            else:
                score = _score_pair(llm, query, chunk.content)
                scores.append(score)

        # Filter by threshold, but keep at least 2 chunks
        original_size = len(cluster.chunks)
        kept_indices = [i for i, score in enumerate(scores) if score >= threshold]

        # Ensure minimum cluster size
        if len(kept_indices) < 2:
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            kept_indices = sorted_indices[:2]

        # Create filtered cluster (preserve id and similarity from original)
        from .models import Cluster
        filtered_chunks = [cluster.chunks[i] for i in kept_indices]
        filtered_cluster = Cluster(
            id=cluster.id,
            chunks=filtered_chunks,
            centroid_idx=0,  # Reset centroid to first chunk
            similarity_score=cluster.similarity_score,
        )

        # Create result
        result = RerankResult(
            original_size=original_size,
            filtered_size=len(kept_indices),
            kept_indices=kept_indices,
            scores=scores,
        )

        if verbose:
            print(f"  Kept {result.filtered_size}/{result.original_size} chunks "
                  f"({result.filter_rate:.1f}% filtered)")
            print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")

        results.append((filtered_cluster, result))

    return results


def print_rerank_summary(results: List[Tuple["Cluster", RerankResult]]) -> None:
    """
    Print summary statistics for reranking results.

    Args:
        results: List of (cluster, rerank_result) tuples
    """
    if not results:
        print("No reranking results to summarize")
        return

    total_original = sum(r.original_size for _, r in results)
    total_filtered = sum(r.filtered_size for _, r in results)
    avg_filter_rate = sum(r.filter_rate for _, r in results) / len(results)

    print("\n" + "=" * 60)
    print("RERANKING SUMMARY")
    print("=" * 60)
    print(f"Total clusters:     {len(results)}")
    print(f"Total chunks:       {total_original} → {total_filtered}")
    print(f"Average filter:     {avg_filter_rate:.1f}%")
    print(f"Chunks removed:     {total_original - total_filtered}")
    print("=" * 60)
