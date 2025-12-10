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
Lua-specific chunker using tree-sitter.

Extracts function declarations, function definitions, and local functions.
"""

from typing import List, Optional
from pathlib import Path

from .base import BaseChunker
from ..models import CodeChunk


class LuaChunker(BaseChunker):
    """AST-aware Lua chunker using tree-sitter."""

    def __init__(self):
        self._parser = None
        self._language = None

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is not None:
            return

        try:
            import tree_sitter_lua as tslua
            from tree_sitter import Language, Parser

            self._language = Language(tslua.language())
            self._parser = Parser(self._language)
        except ImportError as e:
            raise ImportError(
                "tree-sitter-lua not installed. "
                "Install with: pip install tree-sitter-lua"
            ) from e

    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """Extract Lua functions."""
        self._ensure_parser()

        tree = self._parser.parse(content.encode("utf-8"))
        chunks = []

        # Walk tree and extract relevant nodes
        self._extract_chunks(
            node=tree.root_node,
            content=content,
            file_path=file_path,
            language=language,
            min_lines=min_lines,
            max_lines=max_lines,
            chunks=chunks,
            context="",
        )

        return chunks

    def _extract_chunks(
        self,
        node,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int,
        max_lines: int,
        chunks: List[CodeChunk],
        context: str,
    ):
        """Recursively extract chunks from AST."""
        # Chunk-worthy node types
        if node.type in ("function_declaration", "function_definition", "local_function"):
            chunk = self._create_chunk(
                node, content, file_path, language, context
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return  # Don't recurse into functions

        # Recurse into other nodes
        for child in node.children:
            self._extract_chunks(
                child, content, file_path, language,
                min_lines, max_lines, chunks, context
            )

    def _create_chunk(
        self,
        node,
        content: str,
        file_path: Path,
        language: str,
        context: str,
    ) -> Optional[CodeChunk]:
        """Create a chunk from a function node."""
        name = self._get_name(node)
        start_line = node.start_point[0] + 1  # 1-indexed
        end_line = node.end_point[0] + 1

        # Extract content
        lines = content.split("\n")
        chunk_content = "\n".join(lines[start_line - 1:end_line])

        # Determine chunk type
        chunk_type = "function"
        if node.type == "local_function":
            chunk_type = "local_function"

        return CodeChunk(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=chunk_content,
            language=language,
            chunk_type=chunk_type,
            name=name,
            context=context,
        )

    def _get_name(self, node) -> Optional[str]:
        """Extract function name from node."""
        # For function_declaration: function name(...)
        # For function_definition: variable = function(...)
        # For local_function: local function name(...)

        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
            elif child.type == "dot_index_expression" or child.type == "method_index_expression":
                # Handle table.method or table:method syntax
                return child.text.decode("utf-8")

        return None
