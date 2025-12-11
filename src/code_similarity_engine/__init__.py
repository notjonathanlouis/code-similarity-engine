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
Code Similarity Engine - Find semantic code patterns for abstraction opportunities.

Uses embeddings to find code regions that do similar things but look different,
helping identify refactoring opportunities that syntactic tools miss.

No telemetry. Models cached locally after first download.
"""

__version__ = "0.3.0"

from .indexer import index_codebase
from .embedder import embed_chunks
from .clusterer import cluster_vectors
from .reporter import report_clusters
from .analyzer import analyze_cluster, analyze_clusters
from .config import load_config, find_config_file

__all__ = [
    "__version__",
    "index_codebase",
    "embed_chunks",
    "cluster_vectors",
    "report_clusters",
    "analyze_cluster",
    "analyze_clusters",
    "load_config",
    "find_config_file",
]
