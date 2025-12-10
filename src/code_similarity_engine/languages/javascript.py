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
JavaScript/TypeScript chunker using tree-sitter.

Extracts functions, arrow functions, methods, and classes.
"""

from typing import List, Optional
from pathlib import Path

from .base import BaseChunker
from ..models import CodeChunk


class JavaScriptChunker(BaseChunker):
    """AST-aware JavaScript/TypeScript chunker."""

    def __init__(self):
        self._parser = None
        self._language = None

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is not None:
            return

        try:
            # Try TypeScript first (superset of JS)
            try:
                import tree_sitter_typescript as tstypescript
                from tree_sitter import Language, Parser

                self._language = Language(tstypescript.language_typescript())
                self._parser = Parser(self._language)
            except ImportError:
                # Fall back to JavaScript
                import tree_sitter_javascript as tsjavascript
                from tree_sitter import Language, Parser

                self._language = Language(tsjavascript.language())
                self._parser = Parser(self._language)

        except ImportError as e:
            raise ImportError(
                "tree-sitter-javascript/typescript not installed. "
                "Install with: pip install tree-sitter-javascript"
            ) from e

    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """Extract JavaScript/TypeScript functions and classes."""
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
            "function",
            "arrow_function",
            "method_definition",
            "function_expression",
        }

        if node.type in func_types:
            chunk = self._create_function_chunk(
                node, content, file_path, language, context
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return

        # Class declarations
        elif node.type == "class_declaration" or node.type == "class":
            class_name = self._get_class_name(node)
            new_context = f"class {class_name}" if class_name else "class"

            # Extract class body contents
            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, new_context
                )
            return

        # Object methods (const foo = { bar() {} })
        elif node.type == "pair":
            # Check if value is a function
            for child in node.children:
                if child.type in ("function", "arrow_function"):
                    chunk = self._create_function_chunk(
                        child, content, file_path, language, context
                    )
                    if chunk and chunk.line_count >= min_lines:
                        # Try to get name from pair
                        name = self._get_pair_name(node)
                        if name:
                            chunk.name = name
                        chunks.extend(self._split_large_chunk(chunk, max_lines))
            return

        # Variable declarations might contain arrow functions
        elif node.type == "variable_declarator":
            for child in node.children:
                if child.type == "arrow_function":
                    chunk = self._create_function_chunk(
                        child, content, file_path, language, context
                    )
                    if chunk and chunk.line_count >= min_lines:
                        # Get name from variable
                        name = self._get_declarator_name(node)
                        if name:
                            chunk.name = name
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
        name = self._get_function_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        lines = content.split("\n")
        chunk_content = "\n".join(lines[start_line - 1:end_line])

        # Determine chunk type
        if node.type == "method_definition":
            chunk_type = "method"
        elif node.type == "arrow_function":
            chunk_type = "arrow_function"
        elif context:
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

    def _get_function_name(self, node) -> Optional[str]:
        """Extract function name from node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
            if child.type == "property_identifier":
                return child.text.decode("utf-8")
        return None

    def _get_class_name(self, node) -> Optional[str]:
        """Extract class name from node."""
        for child in node.children:
            if child.type == "identifier" or child.type == "type_identifier":
                return child.text.decode("utf-8")
        return None

    def _get_pair_name(self, node) -> Optional[str]:
        """Get name from object pair."""
        for child in node.children:
            if child.type in ("property_identifier", "string", "identifier"):
                text = child.text.decode("utf-8")
                # Strip quotes from strings
                return text.strip("'\"")
        return None

    def _get_declarator_name(self, node) -> Optional[str]:
        """Get name from variable declarator."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None
