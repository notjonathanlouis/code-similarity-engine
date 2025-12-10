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
C++-specific chunker using tree-sitter.

Extends C chunker with class, namespace, and lambda support.
"""

from typing import List, Optional
from pathlib import Path

from .base import BaseChunker
from ..models import CodeChunk


class CppChunker(BaseChunker):
    """AST-aware C++ chunker using tree-sitter."""

    def __init__(self):
        self._parser = None
        self._language = None

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is not None:
            return

        try:
            import tree_sitter_cpp as tscpp
            from tree_sitter import Language, Parser

            self._language = Language(tscpp.language())
            self._parser = Parser(self._language)
        except ImportError as e:
            raise ImportError(
                "tree-sitter-cpp not installed. "
                "Install with: pip install tree-sitter-cpp"
            ) from e

    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """Extract C++ functions, classes, namespaces, etc."""
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
        # Function definitions (including member functions)
        if node.type == "function_definition":
            chunk = self._create_chunk(
                node, content, file_path, language,
                "method" if context else "function",
                context
            )
            if chunk and chunk.line_count >= min_lines:
                chunks.extend(self._split_large_chunk(chunk, max_lines))
            return

        # Class definitions
        elif node.type == "class_specifier":
            if self._has_body(node):
                class_name = self._get_name(node)
                new_context = f"class {class_name}" if class_name else "class"

                # Extract the whole class as a chunk if small enough
                chunk = self._create_chunk(
                    node, content, file_path, language, "class", context
                )
                if chunk and chunk.line_count >= min_lines and chunk.line_count <= max_lines:
                    chunks.append(chunk)
                else:
                    # For large classes, recurse to extract methods
                    for child in node.children:
                        self._extract_chunks(
                            child, content, file_path, language,
                            min_lines, max_lines, chunks, new_context
                        )
                return

        # Struct definitions (C++ structs are like classes)
        elif node.type == "struct_specifier":
            if self._has_body(node):
                struct_name = self._get_name(node)
                new_context = f"struct {struct_name}" if struct_name else "struct"

                chunk = self._create_chunk(
                    node, content, file_path, language, "struct", context
                )
                if chunk and chunk.line_count >= min_lines and chunk.line_count <= max_lines:
                    chunks.append(chunk)
                else:
                    for child in node.children:
                        self._extract_chunks(
                            child, content, file_path, language,
                            min_lines, max_lines, chunks, new_context
                        )
                return

        # Namespace definitions
        elif node.type == "namespace_definition":
            ns_name = self._get_name(node)
            new_context = f"namespace {ns_name}" if ns_name else "namespace"

            for child in node.children:
                self._extract_chunks(
                    child, content, file_path, language,
                    min_lines, max_lines, chunks, new_context
                )
            return

        # Enum definitions (including enum class)
        elif node.type in ("enum_specifier", "enum_class_specifier"):
            if self._has_body(node):
                chunk = self._create_chunk(
                    node, content, file_path, language, "enum", context
                )
                if chunk and chunk.line_count >= min_lines:
                    chunks.extend(self._split_large_chunk(chunk, max_lines))
                return

        # Template declarations - recurse into the templated entity
        elif node.type == "template_declaration":
            for child in node.children:
                if child.type in ("function_definition", "class_specifier", "struct_specifier"):
                    self._extract_chunks(
                        child, content, file_path, language,
                        min_lines, max_lines, chunks, context
                    )
            return

        # Lambda expressions (capture significant ones)
        elif node.type == "lambda_expression":
            chunk = self._create_chunk(
                node, content, file_path, language, "lambda", context
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

    def _create_chunk(
        self,
        node,
        content: str,
        file_path: Path,
        language: str,
        chunk_type: str,
        context: str,
    ) -> Optional[CodeChunk]:
        """Create a chunk from an AST node."""
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
            chunk_type=chunk_type,
            name=name,
            context=context,
        )

    def _get_name(self, node) -> Optional[str]:
        """Extract name from declaration node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
            if child.type == "type_identifier":
                return child.text.decode("utf-8")
            if child.type == "name":
                return child.text.decode("utf-8")
            # For namespaces with :: qualified names
            if child.type == "namespace_identifier":
                return child.text.decode("utf-8")
            # Function declarators
            if child.type in ("function_declarator", "pointer_declarator"):
                return self._get_name(child)
            # Qualified names (e.g., ClassName::methodName)
            if child.type == "qualified_identifier":
                # Return the rightmost identifier
                for subchild in reversed(child.children):
                    if subchild.type == "identifier":
                        return subchild.text.decode("utf-8")
        return None

    def _has_body(self, node) -> bool:
        """Check if a class/struct/enum has a body."""
        for child in node.children:
            if child.type in ("field_declaration_list", "enumerator_list", "declaration_list"):
                return True
        return False
