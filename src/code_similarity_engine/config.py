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
Configuration file support for CSE.

Looks for .cserc or .cse.toml in current directory or project root.
"""

from pathlib import Path
from typing import Optional, Dict, Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


def find_config_file(start_path: Path) -> Optional[Path]:
    """
    Search for .cserc or .cse.toml in start_path and parent directories.

    Searches up to the root directory or until a config file is found.

    Args:
        start_path: Directory to start searching from

    Returns:
        Path to config file if found, None otherwise
    """
    config_names = [".cserc", ".cse.toml"]

    # Start from the given path and walk up to root
    current = start_path.resolve()

    while True:
        for name in config_names:
            config_path = current / name
            if config_path.is_file():
                return config_path

        # Move to parent directory
        parent = current.parent

        # Stop if we've reached the root
        if parent == current:
            break

        current = parent

    return None


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load CSE configuration from .cserc or .cse.toml file.

    Searches for config file starting from the given path and walking up
    parent directories. Returns empty dict if no config file is found.

    Args:
        path: Directory to start searching from

    Returns:
        Dictionary of configuration values from [cse] section,
        or empty dict if no config file found

    Example config file (.cserc or .cse.toml):
        [cse]
        threshold = 0.85
        min_cluster = 3
        exclude = ["**/tests/**", "**/node_modules/**"]
        focus = ["*.py", "*.js"]
        analyze = true
        rerank = true
        output = "markdown"
        verbose = true
        max_chunks = 15000
        batch_size = 64
        min_lines = 10
        max_lines = 150
        llm_model = "/path/to/model.gguf"
        max_analyze = 10
        rerank_model = "/path/to/reranker.gguf"
        rerank_threshold = 0.6
    """
    if tomllib is None:
        # TOML library not available, silently return empty config
        return {}

    config_path = find_config_file(path)

    if config_path is None:
        return {}

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # Extract [cse] section if it exists
        cse_config = data.get("cse", {})

        return cse_config

    except (OSError, tomllib.TOMLDecodeError):
        # File unreadable or invalid TOML - return empty config
        return {}
