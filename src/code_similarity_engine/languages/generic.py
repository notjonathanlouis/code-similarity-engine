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
Generic fallback chunker using sliding window.

Used when tree-sitter grammar is not available for the language.
"""

from typing import List
from pathlib import Path

from .base import BaseChunker
from ..models import CodeChunk


class GenericChunker(BaseChunker):
    """
    Fallback chunker using sliding window with overlap.

    Tries to break at blank lines for more natural boundaries.
    """

    def __init__(self, overlap_lines: int = 5):
        self.overlap = overlap_lines

    def chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        min_lines: int = 5,
        max_lines: int = 100,
    ) -> List[CodeChunk]:
        """Extract chunks using sliding window."""
        lines = content.split("\n")

        if len(lines) < min_lines:
            return []

        chunks = []
        window_size = max_lines
        step = window_size - self.overlap

        i = 0
        while i < len(lines):
            # Find window end
            end = min(i + window_size, len(lines))

            # Try to break at blank line for cleaner boundary
            if end < len(lines):
                # Look for blank line near the end
                for j in range(end - 1, max(i + min_lines, end - 10), -1):
                    if len(lines[j].strip()) == 0:
                        end = j
                        break

            # Extract chunk
            chunk_lines = lines[i:end]
            chunk_content = "\n".join(chunk_lines)

            # Skip if too small
            if len(chunk_lines) >= min_lines:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    start_line=i + 1,  # 1-indexed
                    end_line=end,
                    content=chunk_content,
                    language=language,
                    chunk_type="block",
                    name=None,
                    context="",
                ))

            # Move window
            i += step

            # Don't create tiny final chunk
            if len(lines) - i < min_lines:
                break

        return chunks
