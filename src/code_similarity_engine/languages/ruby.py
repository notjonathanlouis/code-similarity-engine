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
Ruby-specific chunker using tree-sitter.

Extracts methods, singleton methods, classes, and modules.
"""

from typing import List, Optional
from pathlib import Path

from .base import BaseChunker
from ..models import CodeChunk


class RubyChunker(BaseChunker):
    """AST-aware Ruby chunker using tree-sitter."""

    def __init__(self):
        self._parser = None
        self._language = None

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is not None:
            return

        try:
            import tree_sitter_ruby as tsruby
            from tree_sitter import Language, Parser

            self._language = Language(tsruby.language())
            self._parser = Parser(self._language)
        except ImportError as e:
            raise ImportError(
                "tree-sitter-ruby not installed. "
                "Install with: pip install tree-sitter-ruby"
            ) from e

    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """Extract Ruby methods, classes, modules, and blocks."""
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
        if node.type in ("method", "singleton_method"):
            chunk = self._create_chunk(
                node, content, file_path, language, context, node.type
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return  # Don't recurse into methods

        elif node.type == "class":
            # Get class name for context
            class_name = self._get_name(node)
            new_context = f"class {class_name}" if class_name else context

            # Extract the class itself as a chunk (if not too big)
            chunk = self._create_chunk(
                node, content, file_path, language, context, "class"
            )
            if chunk and min_lines <= chunk.line_count <= max_lines:
                chunks.append(chunk)

            # Also extract methods/nested classes within
            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, new_context
                )
            return

        elif node.type == "module":
            # Get module name for context
            module_name = self._get_name(node)
            new_context = f"module {module_name}" if module_name else context

            # Modules are primarily for context, but extract if standalone
            chunk = self._create_chunk(
                node, content, file_path, language, context, "module"
            )
            if chunk and min_lines <= chunk.line_count <= max_lines:
                chunks.append(chunk)

            # Recurse into module contents
            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, new_context
                )
            return

        elif node.type == "block":
            # Extract significant blocks (do...end)
            chunk = self._create_chunk(
                node, content, file_path, language, context, "block"
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return  # Don't recurse into blocks

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
        chunk_type: str,
    ) -> Optional[CodeChunk]:
        """Create a chunk from an AST node."""
        name = self._get_name(node)
        start_line = node.start_point[0] + 1  # 1-indexed
        end_line = node.end_point[0] + 1

        # Extract content
        lines = content.split("\n")
        chunk_content = "\n".join(lines[start_line - 1:end_line])

        # Refine chunk type for singleton methods
        if chunk_type == "singleton_method":
            chunk_type = "class_method"

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
        """Extract name from node (method, class, module, etc.)."""
        # For methods, look for identifier or method_name child
        for child in node.children:
            if child.type in ("identifier", "constant", "method_name"):
                return child.text.decode("utf-8")
            # For singleton methods like "def self.foo"
            elif child.type == "scope_resolution":
                # Get the method name after ::
                for subchild in child.children:
                    if subchild.type in ("identifier", "constant"):
                        return f"self.{subchild.text.decode('utf-8')}"
            # Sometimes the name is nested in a "name" node
            elif child.type == "name":
                for subchild in child.children:
                    if subchild.type in ("identifier", "constant"):
                        return subchild.text.decode("utf-8")

        # For blocks, try to find a descriptive name from surrounding context
        if node.type == "block":
            # Look for method call before the block
            parent = node.parent
            if parent:
                for child in parent.children:
                    if child.type in ("identifier", "constant"):
                        return f"{child.text.decode('utf-8')}_block"

        return None
