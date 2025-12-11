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
LLM-based cluster analyzer - explains similarities and suggests abstractions.

Uses llama-cpp-python with Qwen3-0.6B GGUF for fully offline inference.
"""

from typing import List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
import json
import os
import contextlib


@contextlib.contextmanager
def _suppress_stderr():
    """Suppress stderr at OS level to catch C library output like ggml Metal init."""
    stderr_fd = 2
    saved = os.dup(stderr_fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)


@dataclass
class ClusterAnalysis:
    """Analysis results for a code cluster."""
    commonality: str          # What the regions share
    abstraction: str          # How to refactor (or why not to)
    suggested_name: str       # Name for extracted function/utility
    complexity: str           # low/medium/high effort to refactor
    worth_refactoring: bool = True  # False if LLM recommends leaving as-is


# Default model paths (legacy, prefer model_manager)
DEFAULT_MODEL_PATHS = [
    Path.home() / ".cache" / "cse" / "models" / "Qwen3-0.6B-Q4_K_M.gguf",
    Path("/Volumes/APPLE-STORAGE/Tether/Tether/Resources/ML Models/Qwen3-0.6B-Q4_K_M.gguf"),
]


def _get_llm_model_from_manager() -> Optional[Path]:
    """Try to get LLM model path from model_manager."""
    try:
        from .model_manager import get_model_path
        return get_model_path("llm")
    except ImportError:
        return None


def analyze_cluster(
    cluster: "Cluster",
    model_path: Optional[Path] = None,
    verbose: bool = False,
) -> Optional[ClusterAnalysis]:
    """
    Analyze a cluster using a local LLM.

    Args:
        cluster: Cluster object with similar code chunks
        model_path: Path to GGUF model file
        verbose: Print progress

    Returns:
        ClusterAnalysis object or None if model unavailable
    """
    from .models import Cluster

    # Find model
    model_file = _find_model(model_path)
    if not model_file:
        if verbose:
            print("   ⚠️  No LLM model found - skipping analysis")
            print("   💡 Download with: cse --download-model")
        return None

    try:
        with _suppress_stderr():
            from llama_cpp import Llama
    except ImportError:
        if verbose:
            print("   ⚠️  llama-cpp-python not installed")
        return None

    # Load model (lazy singleton would be better for multiple clusters)
    if verbose:
        print(f"   Loading LLM: {model_file.name}")

    with _suppress_stderr():
        llm = Llama(
            model_path=str(model_file),
            n_ctx=8192,       # Larger context for detailed prompts
            n_threads=4,
            n_gpu_layers=-1,  # Use GPU if available
            verbose=False,
        )

    # Build prompt
    prompt = _build_analysis_prompt(cluster)

    # Generate response
    response = llm(
        prompt,
        max_tokens=400,   # Reduced - JSON response is ~200 tokens
        temperature=0.5,
        stop=["<|im_end|>", "\n\n\n", "</think>"],
    )

    output = response["choices"][0]["text"].strip()

    # Parse response
    return _parse_analysis(output)


def analyze_clusters(
    clusters: List["Cluster"],
    model_path: Optional[Path] = None,
    max_clusters: Optional[int] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> dict[int, ClusterAnalysis]:
    """
    Analyze multiple clusters.

    Args:
        clusters: List of Cluster objects
        model_path: Path to GGUF model
        max_clusters: Maximum clusters to analyze (LLM is slow)
        verbose: Print progress
        quiet: Suppress model loading messages

    Returns:
        Dict mapping cluster ID to analysis
    """
    from .models import Cluster

    model_file = _find_model(model_path)
    if not model_file:
        if verbose:
            print("   ⚠️  No LLM model found - skipping analysis")
        return {}

    try:
        with _suppress_stderr():
            from llama_cpp import Llama
    except ImportError:
        if verbose:
            print("   ⚠️  llama-cpp-python not installed")
        return {}

    # Load model once
    if verbose and not quiet:
        print(f"   Loading LLM: {model_file.name}")

    with _suppress_stderr():
        llm = Llama(
            model_path=str(model_file),
            n_ctx=8192,       # Larger context for detailed prompts
            n_threads=4,
            n_gpu_layers=-1,  # Use GPU if available
            verbose=False,
        )

    results = {}
    # If max_clusters is None, analyze all clusters
    clusters_to_analyze = clusters if max_clusters is None else clusters[:max_clusters]

    for i, cluster in enumerate(clusters_to_analyze):
        if verbose:
            print(f"   Analyzing cluster {i+1}/{len(clusters_to_analyze)}...")

        prompt = _build_analysis_prompt(cluster)

        try:
            response = llm(
                prompt,
                max_tokens=400,   # Reduced - JSON response is ~200 tokens
                temperature=0.5,
                stop=["<|im_end|>", "\n\n\n", "</think>"],
            )

            output = response["choices"][0]["text"].strip()
            analysis = _parse_analysis(output)

            if analysis:
                results[cluster.id] = analysis

        except Exception as e:
            if verbose:
                print(f"   ⚠️  Failed to analyze cluster {cluster.id}: {e}")

    return results


def analyze_clusters_incremental(
    clusters: List["Cluster"],
    existing_analyses: dict[int, ClusterAnalysis] = None,
    on_analysis_complete: callable = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    model_path: Optional[Path] = None,
    max_clusters: Optional[int] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> tuple[dict[int, ClusterAnalysis], List[int]]:
    """
    Analyze clusters incrementally with checkpoint support.

    Args:
        clusters: List of Cluster objects
        existing_analyses: Already-completed analyses to skip
        on_analysis_complete: Callback(cluster_id, analysis) called after each success
        on_progress: Optional callback(current, total, message) for progress updates
        model_path: Path to GGUF model
        max_clusters: Maximum clusters to analyze
        verbose: Print progress
        quiet: Suppress model loading messages

    Returns:
        (analyses_dict, failed_cluster_ids)
    """
    from .models import Cluster

    existing = existing_analyses or {}
    results = dict(existing)  # Start with existing
    failed = []

    model_file = _find_model(model_path)
    if not model_file:
        if verbose:
            print("   ⚠️  No LLM model found - skipping analysis")
        return results, failed

    try:
        with _suppress_stderr():
            from llama_cpp import Llama
    except ImportError:
        if verbose:
            print("   ⚠️  llama-cpp-python not installed")
        return results, failed

    # Filter to clusters we haven't analyzed yet
    clusters_to_analyze = clusters if max_clusters is None else clusters[:max_clusters]
    pending = [c for c in clusters_to_analyze if c.id not in existing]
    total_to_analyze = len(clusters_to_analyze)

    # Progress callback: Loading model
    if on_progress:
        on_progress(0, total_to_analyze, "Loading LLM model...")

    # Load model once
    if verbose and not quiet:
        print(f"   Loading LLM: {model_file.name}")

    with _suppress_stderr():
        llm = Llama(
            model_path=str(model_file),
            n_ctx=8192,
            n_threads=4,
            n_gpu_layers=-1,
            verbose=False,
    )

    if verbose and existing:
        print(f"   Resuming: {len(existing)} already done, {len(pending)} remaining")

    for i, cluster in enumerate(pending):
        total_done = len(existing) + i

        # Progress callback: Analyzing cluster
        if on_progress:
            on_progress(total_done, total_to_analyze, f"Analyzing cluster {total_done + 1}/{total_to_analyze}")

        if verbose:
            print(f"   Analyzing cluster {total_done + 1}/{total_to_analyze} (ID: {cluster.id})...")

        prompt = _build_analysis_prompt(cluster)

        try:
            response = llm(
                prompt,
                max_tokens=400,   # Reduced from 1024 - JSON response is ~200 tokens
                temperature=0.5,
                stop=["<|im_end|>", "\n\n\n", "</think>"],  # Stop on thinking end too
            )

            output = response["choices"][0]["text"].strip()
            analysis = _parse_analysis(output)

            if analysis:
                results[cluster.id] = analysis

                # Checkpoint callback
                if on_analysis_complete:
                    on_analysis_complete(cluster.id, analysis)

                if verbose:
                    status = "✓" if analysis.worth_refactoring else "⚠️ skip"
                    print(f"      {status} {analysis.suggested_name}")
            else:
                failed.append(cluster.id)
                if verbose:
                    print(f"      ✗ Parse failed")

        except Exception as e:
            failed.append(cluster.id)
            if verbose:
                print(f"   ⚠️  Failed cluster {cluster.id}: {e}")

    # Progress callback: Complete
    if on_progress:
        on_progress(total_to_analyze, total_to_analyze, f"Analyzed {total_to_analyze} clusters")

    return results, failed


def _find_model(model_path: Optional[Path]) -> Optional[Path]:
    """Find a valid model file."""
    if model_path and model_path.exists():
        return model_path

    # Try model_manager first (preferred)
    manager_path = _get_llm_model_from_manager()
    if manager_path:
        return manager_path

    # Fallback to legacy paths
    for path in DEFAULT_MODEL_PATHS:
        if path.exists():
            return path

    return None


def _build_analysis_prompt(cluster: "Cluster") -> str:
    """Build the analysis prompt for a cluster."""
    # System prompt with explicit instructions and few-shot examples
    system_prompt = """You are a senior developer reviewing code for refactoring opportunities. Analyze similar code regions and determine if they should be abstracted.

For each analysis, you MUST provide:
1. **commonality**: A 1-2 sentence explanation of WHAT these code regions do and WHY they're similar. Be specific about the functionality.
2. **worth_refactoring**: true if abstraction would genuinely improve the codebase, false if the similarity is coincidental or abstraction would harm readability.
3. **abstraction**: If worth_refactoring is true: A 2-3 sentence recommendation for HOW to refactor. If false: Explain WHY abstraction isn't recommended.
4. **suggested_name**: A clear name for the extracted utility (use "n/a" if not worth refactoring).
5. **complexity**: "low" (< 1 hour), "medium" (1-4 hours), or "high" (> 4 hours). Use "n/a" if not worth refactoring.

IMPORTANT: Not all similar code should be abstracted! Say "worth_refactoring": false when:
- The similarity is coincidental (similar syntax but different purposes)
- Abstraction would make the code harder to understand
- The code is boilerplate that's clearer when explicit (e.g., switch cases, protocol conformance)
- The regions are already using a shared abstraction appropriately
- Over-abstraction would create tight coupling

EXAMPLES of GOOD analysis:

Example 1 - Worth Refactoring (UI Components):
{"commonality": "These are all floating action buttons with the same structure: a circular button that changes color between blue (idle) and red (recording), with an icon that switches between mic and stop square. The animation and shadow styling is duplicated across 4 views.", "worth_refactoring": true, "abstraction": "Create a reusable FloatingRecordButton component that takes isRecording as a binding and an action closure. The component should handle all the styling, animation, and icon switching internally.", "suggested_name": "FloatingRecordButton", "complexity": "low"}

Example 2 - Worth Refactoring (Data Processing):
{"commonality": "Both functions fetch audio duration from a URL using AVURLAsset, extract the duration via CMTimeGetSeconds, and handle the async loading with the same error handling pattern.", "worth_refactoring": true, "abstraction": "Extract a shared getAudioDuration(from url: URL) async throws -> Double function that handles the AVURLAsset loading and CMTime conversion. Both callers can then use this utility.", "suggested_name": "getAudioDuration", "complexity": "low"}

Example 3 - NOT Worth Refactoring (Coincidental Similarity):
{"commonality": "Both regions use a switch statement to handle enum cases, but they're switching on different enums for completely different purposes - one handles UI state, the other handles network responses.", "worth_refactoring": false, "abstraction": "These switch statements are similar in structure but serve unrelated purposes. Abstracting them would create confusing coupling between unrelated subsystems. The similarity is just a common Swift pattern, not duplicated logic.", "suggested_name": "n/a", "complexity": "n/a"}

Example 4 - NOT Worth Refactoring (Boilerplate):
{"commonality": "These are Codable conformance implementations that decode JSON fields. Each struct has its own decode implementation with similar structure.", "worth_refactoring": false, "abstraction": "This is standard Codable boilerplate that Swift requires. Each implementation handles different fields specific to its struct. Attempting to abstract this would fight the language's design and reduce clarity.", "suggested_name": "n/a", "complexity": "n/a"}

Respond with ONLY valid JSON, no markdown or explanation outside the JSON. /no_think"""

    # Build user message with code regions
    user_parts = [f"I found {cluster.size} similar code regions:\n"]

    # Show up to 3 regions from the cluster
    regions_to_show = min(3, len(cluster.chunks))
    for i, chunk in enumerate(cluster.chunks[:regions_to_show]):
        user_parts.append(f"Region {i+1}: {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})")
        user_parts.append(f"```{chunk.language}")
        # Show up to 600 chars of each region for better context
        content = chunk.content[:600]
        if len(chunk.content) > 600:
            content += "\n// ... (truncated)"
        user_parts.append(content)
        user_parts.append("```\n")

    if cluster.size > regions_to_show:
        user_parts.append(f"(Plus {cluster.size - regions_to_show} more similar regions)\n")

    user_parts.append("Analyze these regions and recommend how to refactor them.")

    user_message = "\n".join(user_parts)

    # Build full prompt with ChatML format
    # Prime with '{"' to force immediate JSON output (skip thinking mode)
    prompt = (
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_message}\n<|im_end|>\n"
        '<|im_start|>assistant\n{"'
    )
    return prompt


def _parse_analysis(output: str, primed: bool = True) -> Optional[ClusterAnalysis]:
    """Parse LLM output into ClusterAnalysis.

    Args:
        output: Raw LLM output
        primed: If True, prepend '{"' (we primed the output to skip thinking)
    """
    try:
        raw = output.strip()

        # If we primed with '{"', prepend it back
        if primed and not raw.startswith('{'):
            raw = '{"' + raw

        # Handle Qwen3 /think mode - extract content after </think>
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        # Handle markdown code fences
        if "```json" in raw:
            start = raw.index("```json") + 7
            end = raw.index("```", start) if "```" in raw[start:] else len(raw)
            raw = raw[start:end].strip()
        elif "```" in raw:
            # Generic code fence
            start = raw.index("```") + 3
            end = raw.index("```", start) if "```" in raw[start:] else len(raw)
            raw = raw[start:end].strip()

        # Find the JSON object in the output
        if "{" not in raw:
            return None

        start = raw.index("{")
        # Find matching closing brace
        depth = 0
        end = start
        for i, c in enumerate(raw[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        raw = raw[start:end]

        data = json.loads(raw)

        # Validate we got meaningful content (not just echoes of field names)
        commonality = data.get("commonality", "").strip()
        abstraction = data.get("abstraction", "").strip()
        suggested_name = data.get("suggested_name", "").strip()
        complexity = data.get("complexity", "medium").strip().lower()

        # Parse worth_refactoring - default to True for backwards compatibility
        worth_refactoring_raw = data.get("worth_refactoring", True)
        if isinstance(worth_refactoring_raw, bool):
            worth_refactoring = worth_refactoring_raw
        elif isinstance(worth_refactoring_raw, str):
            worth_refactoring = worth_refactoring_raw.lower() in ("true", "yes", "1")
        else:
            worth_refactoring = True

        # Reject trivially short responses
        if len(commonality) < 20 or len(abstraction) < 20:
            return None

        # Normalize complexity (allow "n/a" for not-worth-refactoring cases)
        if complexity in ("n/a", "na", "none", ""):
            complexity = "n/a" if not worth_refactoring else "medium"
        elif complexity not in ("low", "medium", "high"):
            complexity = "medium"

        # Handle "n/a" suggested names
        if suggested_name.lower() in ("n/a", "na", "none", ""):
            suggested_name = "n/a" if not worth_refactoring else "sharedUtility"

        return ClusterAnalysis(
            commonality=commonality,
            abstraction=abstraction,
            suggested_name=suggested_name or "sharedUtility",
            complexity=complexity,
            worth_refactoring=worth_refactoring,
        )

    except (json.JSONDecodeError, ValueError, IndexError) as e:
        # Debug: uncomment to see parsing failures
        # print(f"Failed to parse: {output[:300]}... Error: {e}")
        return None


def download_model(
    model_name: str = "Qwen/Qwen3-0.6B-GGUF",
    quantization: str = "q4_k_m",
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Download a GGUF model from HuggingFace.

    Args:
        model_name: HuggingFace model repo
        quantization: Quantization level (q4_k_m, q8_0, etc.)
        cache_dir: Where to save
        verbose: Print progress

    Returns:
        Path to downloaded model or None
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        if verbose:
            print("huggingface_hub not installed")
        return None

    cache_path = cache_dir or (Path.home() / ".cache" / "cse")
    cache_path.mkdir(parents=True, exist_ok=True)

    filename = f"qwen3-0.6b-{quantization}.gguf"

    if verbose:
        print(f"Downloading {filename}...")

    try:
        path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            local_dir=cache_path,
            local_dir_use_symlinks=False,
        )
        return Path(path)
    except Exception as e:
        if verbose:
            print(f"Failed to download: {e}")
        return None


# --- Parallel Processing Support ---

# Global variable for worker's LLM instance (initialized once per process)
_worker_llm = None
_worker_model_path = None


def _init_worker(model_path: str):
    """Initialize worker process with LLM model (called once per process)."""
    global _worker_llm, _worker_model_path

    _worker_model_path = model_path

    with _suppress_stderr():
        from llama_cpp import Llama
        _worker_llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_threads=2,  # Fewer threads per worker to share CPU
            n_gpu_layers=-1,
            verbose=False,
        )


def _analyze_cluster_worker(cluster_data: dict) -> tuple[int, Optional[ClusterAnalysis]]:
    """Worker function for parallel processing.

    Uses the pre-initialized LLM model (loaded once per process via initializer).

    Args:
        cluster_data: Serializable dict representation of the cluster

    Returns:
        (cluster_id, analysis) tuple
    """
    global _worker_llm

    cluster_id = cluster_data['id']

    try:
        if _worker_llm is None:
            return (cluster_id, None)

        # Reconstruct cluster-like object for prompt building
        from .models import CodeChunk, Cluster

        chunks = [
            CodeChunk(
                content=c['content'],
                file_path=Path(c['file_path']),
                start_line=c['start_line'],
                end_line=c['end_line'],
                language=c['language'],
                chunk_type=c['chunk_type'],
                name=c.get('name'),
            )
            for c in cluster_data['chunks']
        ]

        cluster = Cluster(
            id=cluster_data['id'],
            chunks=chunks,
            centroid_idx=cluster_data.get('centroid_idx', 0),
            similarity_score=cluster_data.get('similarity_score', 0.9),
        )

        prompt = _build_analysis_prompt(cluster)

        response = _worker_llm(
            prompt,
            max_tokens=400,
            temperature=0.5,
            stop=["<|im_end|>", "\n\n\n", "</think>"],
        )

        output = response["choices"][0]["text"].strip()
        analysis = _parse_analysis(output, primed=True)

        return (cluster_id, analysis)

    except Exception as e:
        return (cluster_id, None)


def _cluster_to_dict(cluster) -> dict:
    """Convert a Cluster object to a serializable dict for multiprocessing."""
    return {
        'id': cluster.id,
        'chunks': [
            {
                'content': c.content,
                'file_path': str(c.file_path),
                'start_line': c.start_line,
                'end_line': c.end_line,
                'language': c.language,
                'chunk_type': c.chunk_type,
                'name': c.name,
            }
            for c in cluster.chunks
        ],
        'centroid_idx': cluster.centroid_idx,
        'similarity_score': cluster.similarity_score,
    }


def analyze_clusters_parallel(
    clusters: List["Cluster"],
    existing_analyses: dict[int, ClusterAnalysis] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    model_path: Optional[Path] = None,
    max_clusters: Optional[int] = None,
    num_workers: int = 2,
    verbose: bool = False,
) -> tuple[dict[int, ClusterAnalysis], List[int]]:
    """
    Analyze clusters using parallel multiprocessing.

    Each worker loads its own model instance and processes clusters independently.
    This can significantly speed up analysis on multi-core systems.

    Args:
        clusters: List of Cluster objects
        existing_analyses: Already-completed analyses to skip
        on_progress: Optional callback(current, total, message) for progress updates
        model_path: Path to GGUF model
        max_clusters: Maximum clusters to analyze
        num_workers: Number of parallel workers (default: 2)
        verbose: Print progress

    Returns:
        (analyses_dict, failed_cluster_ids)
    """
    import multiprocessing as mp

    existing = existing_analyses or {}
    results = dict(existing)
    failed = []

    model_file = _find_model(model_path)
    if not model_file:
        if verbose:
            print("   ⚠️  No LLM model found - skipping analysis")
        return results, failed

    # Filter to clusters we haven't analyzed yet
    clusters_to_analyze = clusters if max_clusters is None else clusters[:max_clusters]
    pending = [c for c in clusters_to_analyze if c.id not in existing]
    total_to_analyze = len(clusters_to_analyze)

    # Progress callback: Starting workers
    if on_progress:
        on_progress(0, total_to_analyze, f"Starting {num_workers} workers...")

    if verbose:
        print(f"   Parallel analysis with {num_workers} workers...")
        if existing:
            print(f"   Resuming: {len(existing)} already done, {len(pending)} remaining")

    if not pending:
        # If nothing to do, still report completion
        if on_progress:
            on_progress(total_to_analyze, total_to_analyze, f"Analyzed {total_to_analyze} clusters")
        return results, failed

    # Convert clusters to serializable dicts
    work_items = [_cluster_to_dict(cluster) for cluster in pending]

    # Process in parallel
    # Use 'spawn' to avoid issues with fork and llama.cpp
    ctx = mp.get_context('spawn')

    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(str(model_file),)
    ) as pool:
        # imap_unordered for progress as results come in
        completed = len(existing)
        for i, (cluster_id, analysis) in enumerate(pool.imap_unordered(_analyze_cluster_worker, work_items)):
            completed += 1

            # Progress callback: Results coming in
            if on_progress:
                on_progress(completed, total_to_analyze, f"Analyzed {completed}/{total_to_analyze}")

            if analysis:
                results[cluster_id] = analysis
                if verbose:
                    status = "✓" if analysis.worth_refactoring else "⚠️ skip"
                    print(f"   [{i+1}/{len(pending)}] Cluster {cluster_id}: {status}")
            else:
                failed.append(cluster_id)
                if verbose:
                    print(f"   [{i+1}/{len(pending)}] Cluster {cluster_id}: ✗ failed")

    # Progress callback: Complete
    if on_progress:
        on_progress(total_to_analyze, total_to_analyze, f"Analyzed {total_to_analyze} clusters")

    return results, failed
