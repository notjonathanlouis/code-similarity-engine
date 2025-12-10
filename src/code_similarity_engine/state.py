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
State management for incremental processing.

Provides checkpointing and resume functionality for long-running analyses.
State is stored as JSON in .cse_cache/state.json for easy inspection.

Stages tracked:
1. index    - Chunking complete
2. embed    - Embeddings computed (uses embeddings.db)
3. cluster  - Clustering complete
4. rerank   - Reranking complete (optional)
5. analyze  - LLM analysis (per-cluster)
6. report   - Report generated
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

from .cache import CACHE_DIR


STATE_FILE = "state.json"


@dataclass
class RunConfig:
    """Configuration for a run - used to detect if config changed."""
    threshold: float
    min_cluster: int
    focus_patterns: List[str]
    exclude_patterns: List[str]
    analyze: bool
    rerank: bool
    max_chunks: int
    min_lines: int
    max_lines: int


@dataclass
class ClusterState:
    """Serializable cluster definition."""
    id: int
    chunk_indices: List[int]  # Indices into the chunks list
    similarity_score: float
    centroid_idx: Optional[int] = None


@dataclass
class AnalysisState:
    """Serializable analysis result."""
    cluster_id: int
    commonality: str
    abstraction: str
    suggested_name: str
    complexity: str
    worth_refactoring: bool
    analyzed_at: float = field(default_factory=time.time)


@dataclass
class RunState:
    """Complete state of an analysis run."""
    # Run metadata
    project_path: str
    started_at: float = field(default_factory=time.time)
    config: Optional[RunConfig] = None

    # Stage completion
    stage: str = "init"  # init, indexed, embedded, clustered, reranked, analyzed, complete

    # Chunk state (saved after indexing)
    total_chunks: int = 0
    chunk_files: List[str] = field(default_factory=list)  # File paths for chunks

    # Cluster state (saved after clustering)
    clusters: List[ClusterState] = field(default_factory=list)

    # Analysis state (saved incrementally)
    analyses: Dict[int, AnalysisState] = field(default_factory=dict)

    # Progress tracking
    embeddings_completed: int = 0
    analyses_completed: int = 0

    # Error tracking
    last_error: Optional[str] = None
    failed_clusters: List[int] = field(default_factory=list)


def get_state_path(project_root: Path) -> Path:
    """Get path to state file."""
    cache_dir = project_root / CACHE_DIR
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / STATE_FILE


def load_state(project_root: Path) -> Optional[RunState]:
    """
    Load existing state from disk.

    Returns None if no state exists or state is invalid.
    """
    state_path = get_state_path(project_root)

    if not state_path.exists():
        return None

    try:
        with open(state_path, 'r') as f:
            data = json.load(f)

        # Reconstruct RunConfig
        config_data = data.get('config')
        config = RunConfig(**config_data) if config_data else None

        # Reconstruct ClusterStates
        clusters = [ClusterState(**c) for c in data.get('clusters', [])]

        # Reconstruct AnalysisStates
        analyses = {}
        for cid, a in data.get('analyses', {}).items():
            analyses[int(cid)] = AnalysisState(**a)

        return RunState(
            project_path=data['project_path'],
            started_at=data.get('started_at', time.time()),
            config=config,
            stage=data.get('stage', 'init'),
            total_chunks=data.get('total_chunks', 0),
            chunk_files=data.get('chunk_files', []),
            clusters=clusters,
            analyses=analyses,
            embeddings_completed=data.get('embeddings_completed', 0),
            analyses_completed=data.get('analyses_completed', 0),
            last_error=data.get('last_error'),
            failed_clusters=data.get('failed_clusters', []),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # Invalid state file - return None to start fresh
        return None


def save_state(project_root: Path, state: RunState):
    """
    Save state to disk atomically.

    Writes to a temp file then renames for crash safety.
    """
    state_path = get_state_path(project_root)
    temp_path = state_path.with_suffix('.tmp')

    # Convert to serializable dict
    data = {
        'project_path': state.project_path,
        'started_at': state.started_at,
        'config': asdict(state.config) if state.config else None,
        'stage': state.stage,
        'total_chunks': state.total_chunks,
        'chunk_files': state.chunk_files,
        'clusters': [asdict(c) for c in state.clusters],
        'analyses': {str(k): asdict(v) for k, v in state.analyses.items()},
        'embeddings_completed': state.embeddings_completed,
        'analyses_completed': state.analyses_completed,
        'last_error': state.last_error,
        'failed_clusters': state.failed_clusters,
    }

    # Write atomically
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)

    temp_path.rename(state_path)


def clear_state(project_root: Path) -> bool:
    """Delete state file. Returns True if file existed."""
    state_path = get_state_path(project_root)
    if state_path.exists():
        state_path.unlink()
        return True
    return False


def can_resume(
    project_root: Path,
    current_config: RunConfig,
) -> tuple[bool, Optional[str], Optional[RunState]]:
    """
    Check if we can resume from existing state.

    Returns:
        (can_resume, reason_if_not, state)
    """
    state = load_state(project_root)

    if state is None:
        return (False, "No previous state found", None)

    # Check if project path matches
    if state.project_path != str(project_root):
        return (False, f"Project path changed: {state.project_path} -> {project_root}", state)

    # Check if config matches (key parameters)
    if state.config:
        if state.config.threshold != current_config.threshold:
            return (False, f"Threshold changed: {state.config.threshold} -> {current_config.threshold}", state)
        if state.config.min_cluster != current_config.min_cluster:
            return (False, f"Min cluster changed: {state.config.min_cluster} -> {current_config.min_cluster}", state)
        if state.config.focus_patterns != current_config.focus_patterns:
            return (False, "Focus patterns changed", state)
        if state.config.exclude_patterns != current_config.exclude_patterns:
            return (False, "Exclude patterns changed", state)

    # Can resume from any non-complete stage
    if state.stage == "complete":
        return (False, "Previous run completed", state)

    return (True, None, state)


def config_from_cli_args(
    threshold: float,
    min_cluster: int,
    focus: tuple,
    exclude: tuple,
    analyze: bool,
    rerank: bool,
    max_chunks: int,
    min_lines: int,
    max_lines: int,
) -> RunConfig:
    """Create RunConfig from CLI arguments."""
    return RunConfig(
        threshold=threshold,
        min_cluster=min_cluster,
        focus_patterns=list(focus),
        exclude_patterns=list(exclude),
        analyze=analyze,
        rerank=rerank,
        max_chunks=max_chunks,
        min_lines=min_lines,
        max_lines=max_lines,
    )


def create_cluster_states(clusters: list) -> List[ClusterState]:
    """Convert Cluster objects to serializable ClusterState."""
    states = []
    for cluster in clusters:
        states.append(ClusterState(
            id=cluster.id,
            chunk_indices=[i for i, c in enumerate(cluster.chunks)],  # This won't work as-is
            similarity_score=cluster.similarity_score,
            centroid_idx=cluster.centroid_idx,
        ))
    return states


def get_progress_summary(state: RunState) -> str:
    """Get human-readable progress summary."""
    lines = [
        f"Stage: {state.stage}",
        f"Chunks: {state.total_chunks}",
        f"Embeddings: {state.embeddings_completed}/{state.total_chunks}",
        f"Clusters: {len(state.clusters)}",
        f"Analyses: {state.analyses_completed}/{len(state.clusters)}",
    ]

    if state.failed_clusters:
        lines.append(f"Failed: {len(state.failed_clusters)} clusters")

    if state.last_error:
        lines.append(f"Last error: {state.last_error[:50]}...")

    return "\n".join(lines)
