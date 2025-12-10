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
Scala-specific chunker using tree-sitter.

Extracts functions, classes, objects, traits, and val/var definitions.
"""

from typing import List, Optional
from pathlib import Path

from .base import BaseChunker
from ..models import CodeChunk


class ScalaChunker(BaseChunker):
    """AST-aware Scala chunker using tree-sitter."""

    def __init__(self):
        self._parser = None
        self._language = None

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is not None:
            return

        try:
            import tree_sitter_scala as tsscala
            from tree_sitter import Language, Parser

            self._language = Language(tsscala.language())
            self._parser = Parser(self._language)
        except ImportError as e:
            raise ImportError(
                "tree-sitter-scala not installed. "
                "Install with: pip install tree-sitter-scala"
            ) from e

    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """Extract Scala functions, classes, objects, and traits."""
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
            package="",
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
        package: str,
    ):
        """Recursively extract chunks from AST."""
        # Package clause - use as context for all subsequent nodes
        if node.type == "package_clause":
            package_name = self._get_package_name(node)
            new_package = package_name if package_name else package

            # Continue processing siblings with package context
            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, context, new_package
                )
            return

        # Function definition
        elif node.type == "function_definition":
            chunk = self._create_chunk(
                node, content, file_path, language, "function", context, package
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return

        # Val/Var definitions (only extract significant ones)
        elif node.type in ("val_definition", "var_definition"):
            chunk = self._create_chunk(
                node, content, file_path, language,
                "val" if node.type == "val_definition" else "var",
                context, package
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return

        # Class definition - recurse into it with context
        elif node.type == "class_definition":
            class_name = self._get_name(node)
            new_context = f"class {class_name}" if class_name else "class"
            full_context = f"{package}.{new_context}" if package else new_context

            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, full_context, package
                )
            return

        # Object definition - recurse into it with context
        elif node.type == "object_definition":
            object_name = self._get_name(node)
            new_context = f"object {object_name}" if object_name else "object"
            full_context = f"{package}.{new_context}" if package else new_context

            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, full_context, package
                )
            return

        # Trait definition - recurse into it with context
        elif node.type == "trait_definition":
            trait_name = self._get_name(node)
            new_context = f"trait {trait_name}" if trait_name else "trait"
            full_context = f"{package}.{new_context}" if package else new_context

            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, full_context, package
                )
            return

        # Recurse into other nodes
        for child in node.children:
            self._extract_chunks(
                child, content, file_path, language,
                min_lines, max_lines, chunks, context, package
            )

    def _create_chunk(
        self,
        node,
        content: str,
        file_path: Path,
        language: str,
        chunk_type: str,
        context: str,
        package: str,
    ) -> Optional[CodeChunk]:
        """Create a chunk from an AST node."""
        name = self._get_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        lines = content.split("\n")
        chunk_content = "\n".join(lines[start_line - 1:end_line])

        # Determine if it's a method (inside class/object/trait)
        if context and chunk_type == "function":
            chunk_type = "method"

        # Build full context with package
        full_context = f"{package}.{context}" if package and context else (package or context)

        return CodeChunk(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=chunk_content,
            language=language,
            chunk_type=chunk_type,
            name=name,
            context=full_context if full_context else None,
        )

    def _get_name(self, node) -> Optional[str]:
        """Extract name from declaration node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
            # Pattern definitions may have pattern_definition nodes
            if child.type == "pattern_definition":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        return subchild.text.decode("utf-8")
        return None

    def _get_package_name(self, node) -> Optional[str]:
        """Extract package name from package_clause node."""
        for child in node.children:
            if child.type == "package_identifier":
                return child.text.decode("utf-8")
            # Handle qualified identifiers like package com.example.app
            if child.type == "stable_identifier":
                return child.text.decode("utf-8")
        return None
