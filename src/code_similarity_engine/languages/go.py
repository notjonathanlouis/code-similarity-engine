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
Go-specific chunker using tree-sitter.

Extracts functions and methods.
"""

from typing import List, Optional
from pathlib import Path

from .base import BaseChunker
from ..models import CodeChunk


class GoChunker(BaseChunker):
    """AST-aware Go chunker using tree-sitter."""

    def __init__(self):
        self._parser = None
        self._language = None

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is not None:
            return

        try:
            import tree_sitter_go as tsgo
            from tree_sitter import Language, Parser

            self._language = Language(tsgo.language())
            self._parser = Parser(self._language)
        except ImportError as e:
            raise ImportError(
                "tree-sitter-go not installed. "
                "Install with: pip install tree-sitter-go"
            ) from e

    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """Extract Go functions and methods."""
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
    ):
        """Recursively extract chunks from AST."""
        # Function declarations
        if node.type == "function_declaration":
            chunk = self._create_function_chunk(
                node, content, file_path, language, ""
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return

        # Method declarations (functions with receivers)
        elif node.type == "method_declaration":
            receiver = self._get_receiver_type(node)
            context = f"type {receiver}" if receiver else ""

            chunk = self._create_function_chunk(
                node, content, file_path, language, context, is_method=True
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return

        # Recurse into other nodes
        for child in node.children:
            self._extract_chunks(
                child, content, file_path, language,
                min_lines, max_lines, chunks
            )

    def _create_function_chunk(
        self,
        node,
        content: str,
        file_path: Path,
        language: str,
        context: str,
        is_method: bool = False,
    ) -> Optional[CodeChunk]:
        """Create a chunk from a function/method node."""
        name = self._get_function_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        lines = content.split("\n")
        chunk_content = "\n".join(lines[start_line - 1:end_line])

        chunk_type = "method" if is_method else "function"

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

    def _get_function_name(self, node) -> Optional[str]:
        """Extract function name from node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _get_receiver_type(self, node) -> Optional[str]:
        """Extract receiver type from method declaration."""
        for child in node.children:
            if child.type == "parameter_list":
                # First param list is receiver
                for param in child.children:
                    if param.type == "parameter_declaration":
                        # Get type from parameter
                        for subchild in param.children:
                            if subchild.type == "type_identifier":
                                return subchild.text.decode("utf-8")
                            if subchild.type == "pointer_type":
                                for ptr_child in subchild.children:
                                    if ptr_child.type == "type_identifier":
                                        return ptr_child.text.decode("utf-8")
                break  # Only check first param list
        return None
