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
Model manager for code-similarity-engine.

Handles automatic downloading and caching of GGUF models from HuggingFace.
Models are stored in ~/.cache/cse/models/ by default.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
import os


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    name: str           # Human-readable name
    repo_id: str        # HuggingFace repo ID
    filename: str       # File to download
    size_mb: int        # Approximate size in MB
    purpose: str        # What it's used for


# Model registry - all models CSE can use
MODELS: Dict[str, ModelInfo] = {
    "embedding": ModelInfo(
        name="Qwen3-Embedding-0.6B",
        repo_id="Qwen/Qwen3-Embedding-0.6B-GGUF",
        filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
        size_mb=610,
        purpose="Code embeddings (semantic similarity)",
    ),
    "llm": ModelInfo(
        name="Qwen3-0.6B",
        repo_id="unsloth/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q4_K_M.gguf",
        size_mb=378,
        purpose="LLM analysis (explaining clusters)",
    ),
    "reranker": ModelInfo(
        name="Qwen3-Reranker-0.6B",
        repo_id="Mungert/Qwen3-Reranker-0.6B-GGUF",
        filename="Qwen3-Reranker-0.6B-q4_k_m.gguf",
        size_mb=395,
        purpose="Reranking (improving cluster quality)",
    ),
}


def get_models_dir() -> Path:
    """Get the default models directory (~/.cache/cse/models/)."""
    cache_dir = Path.home() / ".cache" / "cse" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_path(model_key: str) -> Optional[Path]:
    """
    Get path to a model file, checking multiple locations.

    Args:
        model_key: One of "embedding", "llm", "reranker"

    Returns:
        Path to model file if found, None otherwise
    """
    if model_key not in MODELS:
        return None

    model = MODELS[model_key]

    # Check default cache location
    cache_path = get_models_dir() / model.filename
    if cache_path.exists():
        return cache_path

    # Check Tether's ML Models directory (development fallback)
    tether_path = Path("/Volumes/APPLE-STORAGE/Tether/Tether/Resources/ML Models") / model.filename
    if tether_path.exists():
        return tether_path

    return None


def is_model_available(model_key: str) -> bool:
    """Check if a model is available locally."""
    return get_model_path(model_key) is not None


def get_missing_models(required: List[str]) -> List[str]:
    """
    Check which required models are missing.

    Args:
        required: List of model keys to check (e.g., ["embedding", "llm"])

    Returns:
        List of missing model keys
    """
    return [key for key in required if not is_model_available(key)]


def download_model(
    model_key: str,
    force: bool = False,
    verbose: bool = True,
) -> Optional[Path]:
    """
    Download a model from HuggingFace.

    Args:
        model_key: One of "embedding", "llm", "reranker"
        force: Re-download even if exists
        verbose: Print progress

    Returns:
        Path to downloaded model, or None if failed
    """
    if model_key not in MODELS:
        if verbose:
            print(f"   ❌ Unknown model: {model_key}")
        return None

    model = MODELS[model_key]
    dest_path = get_models_dir() / model.filename

    # Skip if exists and not forcing
    if dest_path.exists() and not force:
        if verbose:
            print(f"   ✅ {model.name} already downloaded")
        return dest_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        if verbose:
            print("   ❌ huggingface_hub not installed")
            print("   💡 Install with: pip install huggingface-hub")
        return None

    if verbose:
        print(f"   📥 Downloading {model.name} ({model.size_mb} MB)...")
        print(f"      From: {model.repo_id}")

    try:
        # Disable telemetry
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        path = hf_hub_download(
            repo_id=model.repo_id,
            filename=model.filename,
            local_dir=get_models_dir(),
            local_dir_use_symlinks=False,
        )

        if verbose:
            print(f"   ✅ Downloaded to: {path}")

        return Path(path)

    except Exception as e:
        if verbose:
            print(f"   ❌ Download failed: {e}")
        return None


def download_all_models(
    models: Optional[List[str]] = None,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Optional[Path]]:
    """
    Download multiple models.

    Args:
        models: List of model keys, or None for all
        force: Re-download even if exists
        verbose: Print progress

    Returns:
        Dict mapping model key to path (or None if failed)
    """
    if models is None:
        models = list(MODELS.keys())

    results = {}

    if verbose:
        total_mb = sum(MODELS[k].size_mb for k in models if k in MODELS)
        print(f"\n📦 Downloading {len(models)} models (~{total_mb} MB total)\n")

    for key in models:
        results[key] = download_model(key, force=force, verbose=verbose)
        if verbose:
            print()  # Blank line between models

    return results


def ensure_models(
    required: List[str],
    auto_download: bool = True,
    verbose: bool = True,
) -> bool:
    """
    Ensure required models are available, downloading if needed.

    Args:
        required: List of model keys that must be present
        auto_download: Automatically download missing models
        verbose: Print progress

    Returns:
        True if all required models are available
    """
    missing = get_missing_models(required)

    if not missing:
        return True

    if not auto_download:
        if verbose:
            print(f"\n⚠️  Missing models: {', '.join(missing)}")
            print("   Run with --download-models to download them")
            print("   Or download manually with: cse --download-models")
        return False

    if verbose:
        print(f"\n🔍 Missing models detected: {', '.join(missing)}")

    results = download_all_models(missing, verbose=verbose)

    # Check if all downloads succeeded
    failed = [k for k, v in results.items() if v is None]
    if failed:
        if verbose:
            print(f"\n❌ Failed to download: {', '.join(failed)}")
        return False

    return True


def print_model_status(verbose: bool = True):
    """Print status of all models."""
    print("\n📊 Model Status\n")
    print(f"   Cache directory: {get_models_dir()}\n")

    for key, model in MODELS.items():
        path = get_model_path(key)
        if path:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {model.name}")
            print(f"      Purpose: {model.purpose}")
            print(f"      Path: {path}")
            print(f"      Size: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {model.name} (not downloaded)")
            print(f"      Purpose: {model.purpose}")
            print(f"      Size: ~{model.size_mb} MB")
        print()


def download_models_cli():
    """CLI entry point for downloading models (cse-download-models command)."""
    import sys

    print("📦 Code Similarity Engine - Model Downloader")
    print("=" * 50)

    results = download_all_models(verbose=True)

    failed = [k for k, v in results.items() if v is None]
    if failed:
        print(f"\n❌ Failed to download: {', '.join(failed)}")
        sys.exit(1)

    print("\n✅ All models downloaded successfully!")
    print("\nYou can now run:")
    print("  cse ./your-project -v")
    sys.exit(0)
