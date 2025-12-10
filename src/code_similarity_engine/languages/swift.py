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
Swift-specific chunker using tree-sitter.

Extracts functions, methods, computed properties, and closures.
"""

from typing import List, Optional
from pathlib import Path

from .base import BaseChunker
from ..models import CodeChunk


class SwiftChunker(BaseChunker):
    """AST-aware Swift chunker using tree-sitter."""

    def __init__(self):
        self._parser = None
        self._language = None

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is not None:
            return

        try:
            import tree_sitter_swift as tsswift
            from tree_sitter import Language, Parser

            self._language = Language(tsswift.language())
            self._parser = Parser(self._language)
        except ImportError as e:
            raise ImportError(
                "tree-sitter-swift not installed. "
                "Install with: pip install tree-sitter-swift"
            ) from e

    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """Extract Swift functions, methods, and properties."""
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
        # Swift function types
        func_types = {
            "function_declaration",
            "init_declaration",
            "deinit_declaration",
            "subscript_declaration",
        }

        # Property types
        prop_types = {
            "computed_property",
            "property_declaration",  # with getter/setter
        }

        # Container types
        container_types = {
            "class_declaration",
            "struct_declaration",
            "enum_declaration",
            "protocol_declaration",
            "extension_declaration",
            "actor_declaration",
        }

        if node.type in func_types:
            chunk = self._create_function_chunk(
                node, content, file_path, language, context
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return

        elif node.type in prop_types:
            chunk = self._create_property_chunk(
                node, content, file_path, language, context
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
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

        # Special handling for init/deinit
        if node.type == "init_declaration":
            name = "init"
            chunk_type = "initializer"
        elif node.type == "deinit_declaration":
            name = "deinit"
            chunk_type = "deinitializer"

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

    def _create_property_chunk(
        self,
        node,
        content: str,
        file_path: Path,
        language: str,
        context: str,
    ) -> Optional[CodeChunk]:
        """Create a chunk from a property node."""
        name = self._get_name(node)
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
            chunk_type="computed_property",
            name=name,
            context=context,
        )

    def _get_name(self, node) -> Optional[str]:
        """Extract name from declaration node."""
        # Look for simple_identifier child
        for child in node.children:
            if child.type == "simple_identifier":
                return child.text.decode("utf-8")
            # Also check pattern for variable declarations
            if child.type == "pattern" or child.type == "identifier_pattern":
                for subchild in child.children:
                    if subchild.type == "simple_identifier":
                        return subchild.text.decode("utf-8")
        return None
