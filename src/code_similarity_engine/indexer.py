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
Code indexer - extracts meaningful chunks from source files.

Uses tree-sitter for AST-aware chunking when available,
falls back to sliding window for unsupported languages.
"""

from pathlib import Path
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import os

from .models import CodeChunk
from .languages import get_chunker, detect_language, SUPPORTED_LANGUAGES


# Default patterns to always exclude
DEFAULT_EXCLUDES = [
    "*.git/*",
    "*node_modules/*",
    "*__pycache__/*",
    "*.pyc",
    "*venv/*",
    "*env/*",
    "*.egg-info/*",
    "*build/*",
    "*dist/*",
    "*.tox/*",
    "*target/*",  # Rust
    "*.build/*",  # Swift
    "*DerivedData/*",  # Xcode
    "*Pods/*",  # CocoaPods
    "*.cache/*",
]


def index_codebase(
    root_path: Path,
    exclude_patterns: Optional[List[str]] = None,
    focus_patterns: Optional[List[str]] = None,
    forced_language: Optional[str] = None,
    min_lines: int = 5,
    max_lines: int = 100,
    max_chunks: int = 10000,
    verbose: bool = False,
) -> List[CodeChunk]:
    """
    Index a codebase and extract code chunks.

    Args:
        root_path: Root directory to scan
        exclude_patterns: Glob patterns to exclude (added to defaults)
        focus_patterns: Only include files matching these patterns
        forced_language: Force all files to be parsed as this language
        min_lines: Minimum lines per chunk
        max_lines: Maximum lines per chunk
        max_chunks: Stop after this many chunks (safety limit)
        verbose: Print progress

    Returns:
        List of CodeChunk objects
    """
    # Combine default and user excludes
    all_excludes = DEFAULT_EXCLUDES + (exclude_patterns or [])

    # Find all source files
    source_files = _find_source_files(
        root_path=root_path,
        exclude_patterns=all_excludes,
        focus_patterns=focus_patterns,
        forced_language=forced_language,
    )

    if verbose:
        print(f"   Found {len(source_files)} source files")

    # Process files in parallel
    all_chunks: List[CodeChunk] = []
    processed = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(
                _process_file,
                file_path,
                root_path,
                forced_language,
                min_lines,
                max_lines,
            ): file_path
            for file_path in source_files
        }

        for future in as_completed(futures):
            if len(all_chunks) >= max_chunks:
                break

            try:
                chunks = future.result()
                all_chunks.extend(chunks)
                processed += 1

                if verbose and processed % 50 == 0:
                    print(f"   Processed {processed}/{len(source_files)} files...")

            except Exception as e:
                file_path = futures[future]
                if verbose:
                    print(f"   Warning: Failed to process {file_path}: {e}")

    # Trim to max_chunks if needed
    if len(all_chunks) > max_chunks:
        all_chunks = all_chunks[:max_chunks]

    return all_chunks


def _find_source_files(
    root_path: Path,
    exclude_patterns: List[str],
    focus_patterns: Optional[List[str]],
    forced_language: Optional[str],
) -> List[Path]:
    """Find all source files matching criteria."""
    source_files = []

    # Determine which extensions to look for
    if forced_language:
        extensions = _extensions_for_language(forced_language)
    else:
        extensions = _all_supported_extensions()

    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue

        # Check extension
        if file_path.suffix.lower() not in extensions:
            continue

        # Make relative for pattern matching
        rel_path = str(file_path.relative_to(root_path))

        # Check excludes
        if any(fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(str(file_path), pat)
               for pat in exclude_patterns):
            continue

        # Check focus patterns (if specified, file must match at least one)
        if focus_patterns:
            if not any(fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(file_path.name, pat)
                       for pat in focus_patterns):
                continue

        source_files.append(file_path)

    return source_files


def _process_file(
    file_path: Path,
    root_path: Path,
    forced_language: Optional[str],
    min_lines: int,
    max_lines: int,
) -> List[CodeChunk]:
    """Process a single file and extract chunks."""
    # Detect or use forced language
    language = forced_language or detect_language(file_path)
    if not language:
        return []

    # Read file content
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    # Get appropriate chunker
    chunker = get_chunker(language)

    # Extract chunks
    rel_path = file_path.relative_to(root_path)
    chunks = chunker.chunk(
        content=content,
        file_path=rel_path,
        language=language,
        min_lines=min_lines,
        max_lines=max_lines,
    )

    return chunks


def _extensions_for_language(language: str) -> Set[str]:
    """Get file extensions for a language."""
    ext_map = {
        "python": {".py", ".pyw"},
        "swift": {".swift"},
        "rust": {".rs"},
        "javascript": {".js", ".mjs", ".cjs"},
        "typescript": {".ts", ".tsx"},
        "go": {".go"},
        "java": {".java"},
        "c": {".c", ".h"},
        "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h"},
        "ruby": {".rb"},
        "php": {".php"},
    }
    return ext_map.get(language.lower(), set())


def _all_supported_extensions() -> Set[str]:
    """Get all supported file extensions."""
    return {
        ".py", ".pyw",      # Python
        ".swift",           # Swift
        ".rs",              # Rust
        ".js", ".mjs", ".cjs",  # JavaScript
        ".ts", ".tsx",      # TypeScript
        ".go",              # Go
        ".java",            # Java
        ".c", ".h",         # C
        ".cpp", ".cc", ".cxx", ".hpp", ".hxx",  # C++
        ".rb",              # Ruby
        ".php",             # PHP
    }
