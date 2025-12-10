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
Base chunker interface.
"""

from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

from ..models import CodeChunk


class BaseChunker(ABC):
    """Abstract base class for language-specific chunkers."""

    @abstractmethod
    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """
        Extract code chunks from file content.

        Args:
            content: Full file content
            file_path: Relative path to file
            language: Detected language
            min_lines: Minimum lines per chunk
            max_lines: Maximum lines per chunk (split larger)

        Returns:
            List of CodeChunk objects
        """
        pass

    def _split_large_chunk(
        self,
        chunk: CodeChunk,
        max_lines: int,
    ) -> List[CodeChunk]:
        """Split a chunk that exceeds max_lines."""
        if chunk.line_count <= max_lines:
            return [chunk]

        lines = chunk.content.split("\n")
        chunks = []

        # Split into roughly equal parts
        n_parts = (len(lines) + max_lines - 1) // max_lines

        for i in range(n_parts):
            start_idx = i * max_lines
            end_idx = min((i + 1) * max_lines, len(lines))

            part_content = "\n".join(lines[start_idx:end_idx])
            part_start = chunk.start_line + start_idx
            part_end = chunk.start_line + end_idx - 1

            chunks.append(CodeChunk(
                file_path=chunk.file_path,
                start_line=part_start,
                end_line=part_end,
                content=part_content,
                language=chunk.language,
                chunk_type=f"{chunk.chunk_type}_part",
                name=f"{chunk.name}_part{i+1}" if chunk.name else None,
                context=chunk.context,
            ))

        return chunks
