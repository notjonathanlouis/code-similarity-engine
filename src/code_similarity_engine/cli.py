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
CLI entry point for code-similarity-engine.

Usage:
    cse <path> [options]
    cse --download-models
    cse --model-status
    cse --help
"""

# === STDERR SUPPRESSION ===
# Must happen BEFORE any imports that might load llama-cpp
# The Metal init messages come from C code that ignores Python's stderr
import sys
import os

# Set env vars before any llama imports
os.environ["GGML_LOG_LEVEL"] = "0"
os.environ["LLAMA_LOG_LEVEL"] = "0"
os.environ["GGML_METAL_LOG_LEVEL"] = "0"

# For truly stubborn C-level output, we need to redirect fd 2
# Save original stderr and redirect to devnull during imports
_original_stderr_fd = os.dup(2)
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 2)
# === END STDERR SUPPRESSION SETUP ===

import click
from pathlib import Path
from typing import Optional, Callable
import contextlib
import io

from . import __version__


@contextlib.contextmanager
def suppress_stderr():
    """
    Suppress stderr at the OS level (captures C library output like llama.cpp).

    This uses os.dup2 to redirect file descriptor 2 (stderr) to /dev/null,
    which catches output from C extensions that bypass Python's sys.stderr.
    """
    # Use raw fd 2 directly, not sys.stderr.fileno() which may be detached
    stderr_fd = 2
    saved_stderr = os.dup(stderr_fd)

    try:
        # Open /dev/null (or NUL on Windows)
        devnull = os.open(os.devnull, os.O_WRONLY)
        # Redirect stderr to /dev/null
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        # Also flush Python's stderr buffer
        sys.stderr.flush()
        yield
    finally:
        # Restore original stderr
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)
from .indexer import index_codebase
from .embedder import embed_chunks, find_embedding_model
from .clusterer import cluster_vectors
from .reporter import report_clusters, OutputFormat
from .model_manager import (
    ensure_models,
    download_all_models,
    print_model_status,
    get_model_path,
    download_model,
)
from .config import load_config

# === RESTORE STDERR ===
# Now that imports are done, restore stderr so errors can print
os.dup2(_original_stderr_fd, 2)
os.close(_original_stderr_fd)
os.close(_devnull_fd)
# === END RESTORE ===


# Extension to output format mapping for -o FILE.EXT
EXTENSION_FORMAT_MAP = {
    '.md': 'markdown',
    '.html': 'html',
    '.json': 'json',
    '.txt': 'text',
}


def calculate_severity_score(cluster, analysis=None) -> float:
    """
    Calculate severity score for a cluster based on drift risk.

    Higher score = more urgent to fix.
    Formula considers:
    - Total lines (more = more maintenance burden)
    - File spread (more files = higher inconsistency risk)
    - Similarity (LOWER = copies are diverging = URGENT)
    - Complexity (lower = easier quick win)
    - Worth refactoring (false = deprioritize)
    """
    # If LLM says not worth refactoring, give it very low priority
    if analysis and hasattr(analysis, 'worth_refactoring') and not analysis.worth_refactoring:
        return 0.01  # Almost zero, but still sortable

    lines = cluster.total_lines()
    files = cluster.file_count
    similarity = cluster.similarity_score  # 0.0 - 1.0

    # Drift risk: lower similarity across many files = danger
    drift_risk = (1 - similarity) * files * 10

    # Base value: lines you'd save × file spread
    base_value = lines * files

    # Complexity factor (from LLM analysis if available)
    if analysis and hasattr(analysis, 'complexity'):
        complexity_factor = {"low": 1.0, "medium": 0.6, "high": 0.3}.get(
            analysis.complexity, 0.6
        )
    else:
        complexity_factor = 0.6  # Default to medium

    return (base_value + drift_risk) * complexity_factor


def sort_clusters(clusters, analyses, sort_by: str):
    """Sort clusters by the specified criteria."""
    if sort_by == "severity":
        # Drift risk-based severity (default)
        return sorted(
            clusters,
            key=lambda c: calculate_severity_score(c, analyses.get(c.id)),
            reverse=True
        )
    elif sort_by == "lines":
        # Most duplicated lines first
        return sorted(clusters, key=lambda c: c.total_lines(), reverse=True)
    elif sort_by == "files":
        # Most file spread first
        return sorted(clusters, key=lambda c: c.file_count, reverse=True)
    elif sort_by == "similarity":
        # Highest similarity first (easiest to abstract)
        return sorted(clusters, key=lambda c: c.similarity_score, reverse=True)
    elif sort_by == "quick-wins":
        # Low complexity + high similarity + reasonable lines (skip not-worth-refactoring)
        def quick_win_score(c):
            analysis = analyses.get(c.id)
            # Skip clusters not worth refactoring
            if analysis and hasattr(analysis, 'worth_refactoring') and not analysis.worth_refactoring:
                return 0.01
            if analysis and hasattr(analysis, 'complexity'):
                complexity_mult = {"low": 3.0, "medium": 1.5, "high": 0.5, "n/a": 0.1}.get(
                    analysis.complexity, 1.0
                )
            else:
                complexity_mult = 1.0
            return c.similarity_score * complexity_mult * min(c.total_lines(), 100)
        return sorted(clusters, key=quick_win_score, reverse=True)
    else:
        return clusters  # No sorting


def merge_config_with_cli(
    config: dict,
    cli_value,
    config_key: str,
    default_value,
):
    """
    Merge config file value with CLI value.

    If CLI value differs from default, use CLI (user explicitly set it).
    Otherwise, use config value if present, else use default.

    Args:
        config: Config dict from file
        cli_value: Value from CLI argument
        config_key: Key to look up in config
        default_value: Default value for this option

    Returns:
        Final value to use
    """
    # If CLI value differs from default, user explicitly set it - use CLI
    if cli_value != default_value:
        return cli_value

    # Otherwise, use config value if present, else default
    return config.get(config_key, default_value)


def print_progress(current: int, total: int, message: str, width: int = 30):
    """Print a progress bar with message."""
    if total == 0:
        pct = 100
    else:
        pct = int(current / total * 100)
    filled = int(width * current / max(total, 1))
    bar = "=" * filled + ">" + " " * (width - filled - 1) if filled < width else "=" * width
    # Use \r to overwrite line, \033[K to clear to end of line
    click.echo(f"\r   [{bar}] {current}/{total} {message}\033[K", nl=False)
    if current >= total:
        click.echo()  # newline when complete


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True), required=False)
@click.option(
    "-t", "--threshold",
    type=float,
    default=0.80,
    help="Similarity threshold 0.0-1.0 (default: 0.80)"
)
@click.option(
    "-m", "--min-cluster",
    type=int,
    default=2,
    help="Minimum chunks per cluster (default: 2)"
)
@click.option(
    "-o", "--output",
    type=str,
    default=None,
    help="Output file path (e.g., report.md, output.html, data.json)"
)
@click.option(
    "--sort-by",
    type=click.Choice(["severity", "lines", "files", "similarity", "quick-wins"]),
    default="severity",
    help="Sort clusters by: severity (drift risk), lines, files, similarity, quick-wins"
)
@click.option(
    "-e", "--exclude",
    multiple=True,
    help="Glob patterns to exclude (repeatable)"
)
@click.option(
    "-f", "--focus",
    multiple=True,
    help="Only analyze matching paths (repeatable)"
)
@click.option(
    "-l", "--lang",
    type=str,
    default=None,
    help="Force language detection"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Show progress for all stages"
)
@click.option(
    "--max-chunks",
    type=int,
    default=10000,
    help="Maximum chunks to process (default: 10000)"
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Embedding batch size (default: 32)"
)
@click.option(
    "--min-lines",
    type=int,
    default=5,
    help="Minimum lines per chunk (default: 5)"
)
@click.option(
    "--max-lines",
    type=int,
    default=100,
    help="Maximum lines per chunk (default: 100)"
)
@click.option(
    "--analyze/--no-analyze",
    default=False,
    help="Run LLM analysis on clusters (default: off, use --analyze to enable)"
)
@click.option(
    "--llm-model",
    type=click.Path(exists=True),
    default=None,
    help="Path to LLM GGUF model (auto-detected)"
)
@click.option(
    "--max-analyze",
    type=int,
    default=None,
    help="Max clusters to analyze with LLM (default: all)"
)
@click.option(
    "--parallel",
    type=int,
    default=0,
    help="Number of parallel LLM workers (0=sequential, 2-4 recommended)"
)
@click.option(
    "--rerank/--no-rerank",
    default=True,
    help="Use reranker to improve cluster quality (default: on, use --no-rerank to skip)"
)
@click.option(
    "--rerank-model",
    type=click.Path(exists=True),
    default=None,
    help="Path to reranker GGUF model (auto-detected)"
)
@click.option(
    "--rerank-threshold",
    type=float,
    default=0.5,
    help="Minimum reranker score to keep (default: 0.5)"
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear embedding cache before running"
)
@click.option(
    "--download-models",
    is_flag=True,
    help="Download all required models and exit"
)
@click.option(
    "--model-status",
    is_flag=True,
    help="Show status of all models and exit"
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume from previous state if available (default: on)"
)
@click.option(
    "--show-state",
    is_flag=True,
    help="Show current analysis state and exit"
)
@click.option(
    "--clear-state",
    is_flag=True,
    help="Clear saved state before running"
)
@click.option(
    "-q", "--quiet",
    is_flag=True,
    help="Suppress model loading messages"
)
@click.version_option(version=__version__)
def main(
    path: Optional[str],
    threshold: float,
    min_cluster: int,
    output: str,
    sort_by: str,
    exclude: tuple,
    focus: tuple,
    lang: Optional[str],
    verbose: bool,
    max_chunks: int,
    batch_size: int,
    min_lines: int,
    max_lines: int,
    analyze: bool,
    llm_model: Optional[str],
    max_analyze: int,
    parallel: int,
    rerank: bool,
    rerank_model: Optional[str],
    rerank_threshold: float,
    clear_cache: bool,
    download_models: bool,
    model_status: bool,
    resume: bool,
    show_state: bool,
    clear_state: bool,
    quiet: bool,
):
    """
    Find semantic code similarities for abstraction opportunities.

    PATH is the root directory to analyze.

    Workflow:

      # Step 1: Prep (index, embed, cluster, rerank) - saves to .cse_cache/
      cse ./src

      # Step 2: Export report (reads from cache)
      cse ./src -o report.md

      # Step 3: Add LLM analysis (updates cache)
      cse ./src --analyze

      # Step 4: Export again (now includes analysis)
      cse ./src -o report.html

    More examples:

      # High-precision search
      cse ./src --threshold 0.90 --min-cluster 3

      # Focus on specific files
      cse ./MyApp --focus "*.swift" -o swift_report.md

      # Download models first
      cse --download-models
    """
    # Suppress llama-cpp/ggml noise if quiet mode is enabled
    if quiet:
        os.environ["GGML_METAL_LOG_LEVEL"] = "0"

    # Handle model management commands (don't require path)
    if model_status:
        print_model_status()
        sys.exit(0)

    if download_models:
        click.echo("📦 Downloading all models for code-similarity-engine...")
        results = download_all_models(verbose=True)
        failed = [k for k, v in results.items() if v is None]
        if failed:
            click.echo(f"\n❌ Failed to download: {', '.join(failed)}", err=True)
            sys.exit(1)
        click.echo("\n✅ All models downloaded successfully!")
        sys.exit(0)

    # Path is required for analysis
    if path is None:
        click.echo("❌ Error: PATH is required for analysis.", err=True)
        click.echo("   Use --help for usage information.", err=True)
        click.echo("   Use --download-models to download models.", err=True)
        click.echo("   Use --model-status to check model status.", err=True)
        sys.exit(1)

    root_path = Path(path).resolve()
    cache_root = Path.cwd()  # Cache always in CWD, not inside analyzed path

    # Load config file and merge with CLI args
    config = load_config(root_path)

    # Config values override defaults, but explicit CLI args override config
    threshold = merge_config_with_cli(config, threshold, "threshold", 0.80)
    min_cluster = merge_config_with_cli(config, min_cluster, "min_cluster", 2)
    # Note: output is now a file path, not a format - skip merge
    max_chunks = merge_config_with_cli(config, max_chunks, "max_chunks", 10000)
    batch_size = merge_config_with_cli(config, batch_size, "batch_size", 32)
    min_lines = merge_config_with_cli(config, min_lines, "min_lines", 5)
    max_lines = merge_config_with_cli(config, max_lines, "max_lines", 100)
    analyze = merge_config_with_cli(config, analyze, "analyze", False)
    rerank = merge_config_with_cli(config, rerank, "rerank", True)
    rerank_threshold = merge_config_with_cli(config, rerank_threshold, "rerank_threshold", 0.5)
    verbose = merge_config_with_cli(config, verbose, "verbose", False)

    # Handle exclude/focus patterns from config
    # These are lists in config, tuples from CLI
    if not exclude and "exclude" in config:
        exclude = tuple(config["exclude"]) if isinstance(config["exclude"], list) else ()
    if not focus and "focus" in config:
        focus = tuple(config["focus"]) if isinstance(config["focus"], list) else ()

    # Handle optional path parameters from config
    if llm_model is None and "llm_model" in config:
        llm_model = config["llm_model"]
    if rerank_model is None and "rerank_model" in config:
        rerank_model = config["rerank_model"]
    if max_analyze is None and "max_analyze" in config:
        max_analyze = config["max_analyze"]
    if lang is None and "lang" in config:
        lang = config["lang"]

    if verbose and config:
        click.echo(f"📝 Loaded config from .cserc/.cse.toml")

    # Handle --clear-cache - exit immediately after
    if clear_cache:
        from .cache import clear_cache as do_clear_cache
        if do_clear_cache(cache_root):
            click.echo("🗑️  Cleared embedding cache")
        else:
            click.echo("   No cache to clear")
        sys.exit(0)  # EXIT HERE

    # State management imports
    from .state import (
        load_state, save_state, clear_state as do_clear_state,
        get_progress_summary, RunState, AnalysisState,
        config_from_cli_args,
    )

    # Handle --show-state - exit immediately after
    if show_state:
        state = load_state(cache_root)
        if state:
            click.echo("📊 Current analysis state:")
            click.echo(get_progress_summary(state))
        else:
            click.echo("No saved state found.")
        sys.exit(0)  # EXIT HERE

    # Handle --clear-state - exit immediately after
    if clear_state:
        if do_clear_state(cache_root):
            click.echo("🗑️  Cleared analysis state")
        else:
            click.echo("   No state to clear")
        sys.exit(0)  # EXIT HERE

    # Create current config for state comparison
    current_config = config_from_cli_args(
        threshold=threshold,
        min_cluster=min_cluster,
        focus=focus,
        exclude=exclude,
        analyze=analyze,
        rerank=rerank,
        max_chunks=max_chunks,
        min_lines=min_lines,
        max_lines=max_lines,
    )

    # Try to load existing state for resume
    existing_state = None
    existing_analyses = {}
    if resume:
        existing_state = load_state(cache_root)
        if existing_state and existing_state.analyses:
            # Convert AnalysisState to ClusterAnalysis for compatibility
            from .analyzer import ClusterAnalysis
            for cid, ast in existing_state.analyses.items():
                existing_analyses[cid] = ClusterAnalysis(
                    commonality=ast.commonality,
                    abstraction=ast.abstraction,
                    suggested_name=ast.suggested_name,
                    complexity=ast.complexity,
                    worth_refactoring=ast.worth_refactoring,
                )
            if verbose:
                click.echo(f"📂 Loaded {len(existing_analyses)} previous analyses")

    # Initialize new state
    run_state = RunState(
        project_path=str(root_path),
        config=current_config,
    )

    if verbose:
        click.echo(f"🔍 Analyzing: {root_path}")
        click.echo(f"   Threshold: {threshold}")
        click.echo(f"   Min cluster size: {min_cluster}")
        click.echo(f"   Output format: {output}")
        if analyze:
            click.echo(f"   LLM analysis: enabled")
        if rerank:
            click.echo(f"   Reranking: enabled")

    # Stage 1: Index
    click.echo("📂 Stage 1: Indexing codebase...")
    chunks = index_codebase(
        root_path=root_path,
        exclude_patterns=list(exclude),
        focus_patterns=list(focus),
        forced_language=lang,
        min_lines=min_lines,
        max_lines=max_lines,
        max_chunks=max_chunks,
        verbose=False,  # Use progress callback instead
        on_progress=lambda c, t, m: print_progress(c, t, m) if not quiet else None,
    )

    if not chunks:
        click.echo("❌ No code chunks found. Check your path and filters.", err=True)
        sys.exit(1)

    click.echo(f"   Found {len(chunks)} chunks")

    # Stage 2: Embed
    click.echo("\n🧠 Stage 2: Generating embeddings...")

    # Ensure embedding model is available
    if not find_embedding_model():
        click.echo("   No embedding model found, downloading...")
        downloaded = download_model("embedding", verbose=verbose)
        if not downloaded:
            click.echo("❌ Failed to download embedding model.", err=True)
            click.echo("   Run: cse --download-models", err=True)
            sys.exit(1)

    try:
        with suppress_stderr():
            vectors = embed_chunks(
                chunks=chunks,
                project_root=cache_root,
                batch_size=batch_size,
                verbose=False,  # Use progress callback instead
                quiet=quiet,
                on_progress=lambda c, t, m: print_progress(c, t, m) if not quiet else None,
            )
    except (ImportError, FileNotFoundError) as e:
        click.echo(f"❌ Embedding failed: {e}", err=True)
        sys.exit(1)

    # Stage 3: Cluster
    click.echo("\n🔗 Stage 3: Clustering similar regions...")

    clusters = cluster_vectors(
        vectors=vectors,
        chunks=chunks,
        threshold=threshold,
        min_cluster_size=min_cluster,
        verbose=False,  # Use progress callback instead
        on_progress=lambda c, t, m: print_progress(c, t, m) if not quiet else None,
    )

    if not clusters:
        click.echo("✨ No similar regions found above threshold. Your code is unique!")
        sys.exit(0)

    click.echo(f"   Found {len(clusters)} clusters")

    # Stage 4: Rerank (optional)
    if rerank and clusters:
        click.echo("\n📊 Stage 4: Reranking clusters...")

        # Ensure reranker model is available
        if rerank_model is None and not get_model_path("reranker"):
            click.echo("   No reranker model found, downloading...")
            download_model("reranker", verbose=verbose)

        from .reranker import rerank_clusters as do_rerank

        try:
            with suppress_stderr():
                reranked = do_rerank(
                    clusters=clusters,
                    model_path=Path(rerank_model) if rerank_model else None,
                    threshold=rerank_threshold,
                    verbose=False,  # Use progress callback instead
                    quiet=quiet,
                    on_progress=lambda c, t, m: print_progress(c, t, m) if not quiet else None,
                )
            # Extract refined clusters
            clusters = [c for c, _ in reranked]
            # Filter out empty clusters
            clusters = [c for c in clusters if len(c.chunks) >= min_cluster]

            click.echo(f"   After reranking: {len(clusters)} clusters")
        except (ImportError, FileNotFoundError) as e:
            click.echo(f"   ⚠️  Reranking skipped: {e}")

    # Stage 5: Analyze (optional LLM) - with incremental checkpoints
    analyses = dict(existing_analyses)  # Start with any resumed analyses
    if analyze:
        # Calculate stage number dynamically
        stage_num = 5 if rerank else 4
        click.echo(f"\n🤖 Stage {stage_num}: LLM analysis...")

        # Ensure LLM model is available
        if llm_model is None and not get_model_path("llm"):
            click.echo("   No LLM model found, downloading...")
            download_model("llm", verbose=verbose)

        from .analyzer import analyze_clusters_incremental, analyze_clusters_parallel, ClusterAnalysis

        if parallel > 0:
            # Parallel processing (no incremental callbacks, saves at end)
            with suppress_stderr():
                analyses, failed = analyze_clusters_parallel(
                    clusters=clusters,
                    existing_analyses=existing_analyses,
                    model_path=Path(llm_model) if llm_model else None,
                    max_clusters=max_analyze,
                    num_workers=parallel,
                    verbose=False,
                    on_progress=lambda c, t, m: print_progress(c, t, m) if not quiet else None,
                )

            # Save all results at once
            for cluster_id, analysis in analyses.items():
                if cluster_id not in existing_analyses:
                    run_state.analyses[cluster_id] = AnalysisState(
                        cluster_id=cluster_id,
                        commonality=analysis.commonality,
                        abstraction=analysis.abstraction,
                        suggested_name=analysis.suggested_name,
                        complexity=analysis.complexity,
                        worth_refactoring=analysis.worth_refactoring,
                    )
            run_state.analyses_completed = len(run_state.analyses)
            save_state(cache_root, run_state)

        else:
            # Sequential processing with incremental checkpoints
            def on_analysis_complete(cluster_id: int, analysis: ClusterAnalysis):
                run_state.analyses[cluster_id] = AnalysisState(
                    cluster_id=cluster_id,
                    commonality=analysis.commonality,
                    abstraction=analysis.abstraction,
                    suggested_name=analysis.suggested_name,
                    complexity=analysis.complexity,
                    worth_refactoring=analysis.worth_refactoring,
                )
                run_state.analyses_completed = len(run_state.analyses)
                save_state(cache_root, run_state)

            with suppress_stderr():
                analyses, failed = analyze_clusters_incremental(
                    clusters=clusters,
                    existing_analyses=existing_analyses,
                    on_analysis_complete=on_analysis_complete,
                    model_path=Path(llm_model) if llm_model else None,
                    max_clusters=max_analyze,
                    verbose=False,
                    quiet=quiet,
                    on_progress=lambda c, t, m: print_progress(c, t, m) if not quiet else None,
                )

        if failed:
            run_state.failed_clusters = failed
            save_state(cache_root, run_state)
            click.echo(f"   ⚠️  {len(failed)} clusters failed to analyze")

        click.echo(f"   Analyzed {len(analyses)} clusters")

    # Sort clusters
    if sort_by != "severity" or analyses:  # Always sort if we have analyses or non-default sort
        clusters = sort_clusters(clusters, analyses, sort_by)

    # Mark state as complete (prep done)
    run_state.stage = "complete"
    save_state(cache_root, run_state)

    # Final Stage: Report (only if -o FILE.EXT provided)
    if output:
        output_path = Path(output)
        ext = output_path.suffix.lower()

        if ext not in EXTENSION_FORMAT_MAP:
            valid_exts = ', '.join(EXTENSION_FORMAT_MAP.keys())
            click.echo(f"❌ Invalid output extension '{ext}'. Valid: {valid_exts}", err=True)
            sys.exit(1)

        # Calculate final stage number dynamically
        final_stage = 4
        if rerank:
            final_stage += 1
        if analyze:
            final_stage += 1

        click.echo(f"\n📝 Stage {final_stage}: Generating report...")

        output_format = OutputFormat(EXTENSION_FORMAT_MAP[ext])
        report = report_clusters(
            clusters=clusters,
            root_path=root_path,
            threshold=threshold,
            output_format=output_format,
            analyses=analyses,
        )

        # Write to file
        output_path.write_text(report)
        click.echo(f"   ✅ Report written to: {output_path}")

        if not analyze and analyses == {}:
            click.echo(f"\n💡 To add LLM insights: cse {path} --analyze && cse {path} -o {output}")
    else:
        # No output requested - just show summary
        click.echo(f"\n✅ Prep complete! Found {len(clusters)} clusters.")
        click.echo(f"   💾 Cache saved to: .cse_cache/")
        if not analyze:
            click.echo(f"\n💡 Next steps:")
            click.echo(f"   • Export report:    cse {path} -o report.md")
            click.echo(f"   • Add LLM analysis: cse {path} --analyze")
            click.echo(f"   • Then export:      cse {path} -o report.html")


# Entry point alias for pyproject.toml
cli = main


if __name__ == "__main__":
    main()
