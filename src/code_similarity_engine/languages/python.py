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
Python-specific chunker using tree-sitter.

Extracts functions, methods, and classes.
"""

from typing import List, Optional, Tuple
from pathlib import Path

from .base import BaseChunker
from ..models import CodeChunk


class PythonChunker(BaseChunker):
    """AST-aware Python chunker using tree-sitter."""

    def __init__(self):
        self._parser = None
        self._language = None

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is not None:
            return

        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser

            self._language = Language(tspython.language())
            self._parser = Parser(self._language)
        except ImportError as e:
            raise ImportError(
                "tree-sitter-python not installed. "
                "Install with: pip install tree-sitter-python"
            ) from e

    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """Extract Python functions, methods, and classes."""
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
        if node.type == "function_definition":
            chunk = self._create_function_chunk(
                node, content, file_path, language, context
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return  # Don't recurse into functions

        elif node.type == "class_definition":
            # Get class name for context
            class_name = self._get_class_name(node)
            new_context = f"class {class_name}" if class_name else context

            # Extract the class itself as a chunk (if not too big)
            chunk = self._create_class_chunk(
                node, content, file_path, language, context
            )
            if chunk and min_lines <= chunk.line_count <= max_lines:
                chunks.append(chunk)

            # Also extract methods within
            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, new_context
                )
            return

        # Recurse into other nodes
        for child in node.children:
            self._extract_chunks(
                child, content, file_path, language,
                min_lines, max_lines, chunks, context
            )

    def _create_function_chunk(
        self,
        node,
        content: str,
        file_path: Path,
        language: str,
        context: str,
    ) -> Optional[CodeChunk]:
        """Create a chunk from a function_definition node."""
        name = self._get_function_name(node)
        start_line = node.start_point[0] + 1  # 1-indexed
        end_line = node.end_point[0] + 1

        # Extract content
        lines = content.split("\n")
        chunk_content = "\n".join(lines[start_line - 1:end_line])

        # Determine if method or function
        chunk_type = "method" if context.startswith("class") else "function"

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

    def _create_class_chunk(
        self,
        node,
        content: str,
        file_path: Path,
        language: str,
        context: str,
    ) -> Optional[CodeChunk]:
        """Create a chunk from a class_definition node."""
        name = self._get_class_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        lines = content.split("\n")
        chunk_content = "\n".join(lines[start_line - 1:end_line])

        return CodeChunk(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=chunk_content,
            language=language,
            chunk_type="class",
            name=name,
            context=context,
        )

    def _get_function_name(self, node) -> Optional[str]:
        """Extract function name from node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _get_class_name(self, node) -> Optional[str]:
        """Extract class name from node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None
