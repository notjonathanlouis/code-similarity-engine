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
Language-specific code chunking.

Uses tree-sitter for AST-aware extraction when available,
falls back to generic sliding-window chunking.
"""

from pathlib import Path
from typing import Optional

from .base import BaseChunker
from .generic import GenericChunker


# Language registry - maps language name to chunker class
_CHUNKER_REGISTRY: dict[str, type[BaseChunker]] = {}

# Extension to language mapping
EXTENSION_MAP = {
    ".py": "python",
    ".pyw": "python",
    ".swift": "swift",
    ".rs": "rust",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".java": "java",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
}

# Languages we have tree-sitter support for
SUPPORTED_LANGUAGES = {
    "python", "swift", "rust", "javascript", "typescript", "go", "c", "cpp",
    "java", "csharp", "php", "ruby", "kotlin", "scala", "lua", "bash",
}


def register_chunker(language: str, chunker_class: type[BaseChunker]) -> None:
    """Register a chunker for a language."""
    _CHUNKER_REGISTRY[language.lower()] = chunker_class


def get_chunker(language: str) -> BaseChunker:
    """
    Get a chunker instance for the given language.

    Falls back to generic chunker if no specific one exists.
    """
    language = language.lower()

    # Try to get registered chunker
    if language in _CHUNKER_REGISTRY:
        return _CHUNKER_REGISTRY[language]()

    # Try to lazy-load tree-sitter based chunker
    chunker = _try_load_tree_sitter_chunker(language)
    if chunker:
        return chunker

    # Fall back to generic
    return GenericChunker()


def detect_language(file_path: Path) -> Optional[str]:
    """Detect language from file extension."""
    ext = file_path.suffix.lower()
    return EXTENSION_MAP.get(ext)


def _try_load_tree_sitter_chunker(language: str) -> Optional[BaseChunker]:
    """Try to load a tree-sitter chunker for the language."""
    try:
        if language == "python":
            from .python import PythonChunker
            return PythonChunker()
        elif language == "swift":
            from .swift import SwiftChunker
            return SwiftChunker()
        elif language == "rust":
            from .rust import RustChunker
            return RustChunker()
        elif language in ("javascript", "typescript"):
            from .javascript import JavaScriptChunker
            return JavaScriptChunker()
        elif language == "go":
            from .go import GoChunker
            return GoChunker()
        elif language == "c":
            from .c import CChunker
            return CChunker()
        elif language == "cpp":
            from .cpp import CppChunker
            return CppChunker()
        elif language == "java":
            from .java import JavaChunker
            return JavaChunker()
        elif language == "csharp":
            from .csharp import CSharpChunker
            return CSharpChunker()
        elif language == "php":
            from .php import PHPChunker
            return PHPChunker()
        elif language == "ruby":
            from .ruby import RubyChunker
            return RubyChunker()
        elif language == "kotlin":
            from .kotlin import KotlinChunker
            return KotlinChunker()
        elif language == "scala":
            from .scala import ScalaChunker
            return ScalaChunker()
        elif language == "lua":
            from .lua import LuaChunker
            return LuaChunker()
        elif language == "bash":
            from .bash import BashChunker
            return BashChunker()
    except ImportError:
        pass

    return None
