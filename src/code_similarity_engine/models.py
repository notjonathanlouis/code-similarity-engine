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
Data models for code-similarity-engine.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import hashlib


@dataclass
class CodeChunk:
    """A chunk of code extracted from a source file."""

    file_path: Path          # Relative path from root
    start_line: int          # Starting line number (1-indexed)
    end_line: int            # Ending line number (inclusive)
    content: str             # The actual code
    language: str            # Detected/forced language
    chunk_type: str          # "function", "method", "class", "block"
    name: Optional[str] = None  # Function/method name if available
    context: str = ""        # Surrounding context (class name, imports hint)

    @property
    def id(self) -> str:
        """Unique identifier based on content hash."""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:12]
        return f"{self.file_path}:{self.start_line}-{self.end_line}:{content_hash}"

    @property
    def line_count(self) -> int:
        """Number of lines in this chunk."""
        return self.end_line - self.start_line + 1

    @property
    def location(self) -> str:
        """Human-readable location string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def preview(self, max_chars: int = 60) -> str:
        """Short preview of the content."""
        first_line = self.content.split('\n')[0].strip()
        if len(first_line) > max_chars:
            return first_line[:max_chars-3] + "..."
        return first_line


@dataclass
class Cluster:
    """A group of semantically similar code chunks."""

    id: int                           # Cluster identifier
    chunks: List[CodeChunk]           # Member chunks
    similarity_score: float           # Average pairwise similarity
    centroid_idx: Optional[int] = None  # Index of most representative chunk

    @property
    def size(self) -> int:
        """Number of chunks in this cluster."""
        return len(self.chunks)

    @property
    def files(self) -> List[Path]:
        """Unique files in this cluster."""
        return list(set(c.file_path for c in self.chunks))

    @property
    def file_count(self) -> int:
        """Number of unique files."""
        return len(self.files)

    @property
    def representative(self) -> CodeChunk:
        """Most representative chunk (closest to centroid)."""
        if self.centroid_idx is not None:
            return self.chunks[self.centroid_idx]
        return self.chunks[0]

    def total_lines(self) -> int:
        """Total lines of duplicated code."""
        return sum(c.line_count for c in self.chunks)
