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
Kotlin-specific chunker using tree-sitter.

Extracts functions, classes, objects, companion objects, interfaces, and significant lambdas.
"""

from typing import List, Optional
from pathlib import Path

from .base import BaseChunker
from ..models import CodeChunk


class KotlinChunker(BaseChunker):
    """AST-aware Kotlin chunker using tree-sitter."""

    def __init__(self):
        self._parser = None
        self._language = None

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is not None:
            return

        try:
            import tree_sitter_kotlin as tskotlin
            from tree_sitter import Language, Parser

            self._language = Language(tskotlin.language())
            self._parser = Parser(self._language)
        except ImportError as e:
            raise ImportError(
                "tree-sitter-kotlin not installed. "
                "Install with: pip install tree-sitter-kotlin"
            ) from e

    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """Extract Kotlin functions, classes, objects, and lambdas."""
        self._ensure_parser()

        tree = self._parser.parse(content.encode("utf-8"))
        chunks = []

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
        # Function types
        func_types = {
            "function_declaration",
        }

        # Container types that provide context
        container_types = {
            "class_declaration",
            "object_declaration",
            "interface_declaration",
        }

        # Special container types
        if node.type in func_types:
            chunk = self._create_function_chunk(
                node, content, file_path, language, context
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return

        elif node.type == "companion_object":
            # Companion objects provide context for their members
            new_context = f"{context}.companion" if context else "companion"

            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, new_context
                )
            return

        elif node.type in container_types:
            # Get container name for context
            name = self._get_name(node)
            type_name = node.type.replace("_declaration", "")
            new_context = f"{type_name} {name}" if name else context

            # Recurse into container
            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, new_context
                )
            return

        elif node.type == "lambda_literal":
            # Extract significant lambdas (multi-line)
            chunk = self._create_lambda_chunk(
                node, content, file_path, language, context
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
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
        """Create a chunk from a function node."""
        name = self._get_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        lines = content.split("\n")
        chunk_content = "\n".join(lines[start_line - 1:end_line])

        # Determine type based on context
        if context:
            chunk_type = "method"
        else:
            chunk_type = "function"

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

    def _create_lambda_chunk(
        self,
        node,
        content: str,
        file_path: Path,
        language: str,
        context: str,
    ) -> Optional[CodeChunk]:
        """Create a chunk from a significant lambda literal."""
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
            chunk_type="lambda",
            name=None,
            context=context,
        )

    def _get_name(self, node) -> Optional[str]:
        """Extract name from declaration node."""
        # Look for simple_identifier child (Kotlin uses this for names)
        for child in node.children:
            if child.type == "simple_identifier":
                return child.text.decode("utf-8")
            # Also check type_identifier for class/object names
            if child.type == "type_identifier":
                return child.text.decode("utf-8")
        return None
