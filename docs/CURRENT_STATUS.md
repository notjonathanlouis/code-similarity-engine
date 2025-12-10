# Code Similarity Engine - Current Status

**Last Updated:** December 9, 2025
**Version:** 0.1.0 (MVP)

> See also: [Future Development Tasks](./ROADMAP.md)

---

## 🎯 What It Does

Code Similarity Engine (`cse`) is a CLI tool that uses semantic embeddings to find code regions that do similar things but look different. Unlike syntactic tools (regex, linters), it catches patterns that have different variable names, structure, or formatting.

**Primary User:** LLMs (Claude, etc.) - helps identify refactoring opportunities
**Secondary User:** Human developers doing code cleanup

---

## ✅ Working Features

### Core Pipeline

| Stage | Component | Status | Implementation |
|-------|-----------|--------|----------------|
| 1 | **Indexer** | ✅ Working | Threaded file scanning, tree-sitter AST parsing |
| 2 | **Embedder** | ✅ Working | Direct `transformers` (no sentence-transformers telemetry) |
| 3 | **Clusterer** | ✅ Working | Agglomerative clustering with cosine distance |
| 4 | **Analyzer** | ✅ Working | llama-cpp-python with GGUF models (optional) |
| 5 | **Reporter** | ✅ Working | text/markdown/json output formats |

### Language Support (tree-sitter)

| Language | Chunker | Extracts |
|----------|---------|----------|
| Python | ✅ `python.py` | functions, methods, classes |
| Swift | ✅ `swift.py` | functions, methods, computed properties, init/deinit |
| Rust | ✅ `rust.py` | functions, impl blocks, traits |
| JavaScript/TypeScript | ✅ `javascript.py` | functions, arrow functions, methods, classes |
| Go | ✅ `go.py` | functions, methods |
| Other | ✅ `generic.py` | sliding window fallback |

### CLI Options

```bash
cse <path> [options]

Core Options:
  -t, --threshold FLOAT      Similarity threshold 0.0-1.0 (default: 0.80)
  -m, --min-cluster INT      Minimum chunks per cluster (default: 2)
  -o, --output FORMAT        text | markdown | json (default: text)
  -v, --verbose              Show progress for all stages

Filtering:
  -f, --focus PATTERN        Only analyze matching paths (repeatable)
  -e, --exclude PATTERN      Glob patterns to exclude (repeatable)
  -l, --lang LANG            Force language detection

LLM Analysis:
  --analyze / --no-analyze   Use LLM to explain clusters
  --llm-model PATH           Path to GGUF model
  --max-analyze INT          Max clusters to analyze (default: 20)

Advanced:
  --embedding-model TEXT     HuggingFace model for embeddings
  --offline                  Run fully offline (models must be cached)
  --max-chunks INT           Safety limit (default: 10000)
  --batch-size INT           Embedding batch size (auto-detected)
```

---

## 📁 Project Structure

```
code-similarity-engine/
├── src/code_similarity_engine/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # python -m entry point
│   ├── cli.py               # Click CLI (all options)
│   ├── models.py            # CodeChunk, Cluster dataclasses
│   ├── indexer.py           # File scanning, orchestrates chunking
│   ├── embedder.py          # Direct transformers (no telemetry)
│   ├── clusterer.py         # Agglomerative clustering
│   ├── analyzer.py          # LLM analysis with llama-cpp
│   ├── reporter.py          # Output formatting
│   └── languages/
│       ├── __init__.py      # Language registry
│       ├── base.py          # BaseChunker ABC
│       ├── python.py        # tree-sitter Python
│       ├── swift.py         # tree-sitter Swift
│       ├── rust.py          # tree-sitter Rust
│       ├── javascript.py    # tree-sitter JS/TS
│       ├── go.py            # tree-sitter Go
│       └── generic.py       # Sliding window fallback
├── docs/
│   ├── CURRENT_STATUS.md    # This file
│   └── ROADMAP.md           # Future tasks
├── requirements.txt
├── download_qwen3_models.sh
└── tests/                   # (empty, needs tests)
```

---

## 🧠 Models

### Embedding Model (current)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Loaded via:** Direct `transformers` (no telemetry)
- **Device:** Auto-detects MPS/CUDA/CPU

### LLM Models (downloaded)

Located in `/Volumes/APPLE-STORAGE/Tether/Tether/Resources/ML Models/`:

| Model | Size | Purpose | Status |
|-------|------|---------|--------|
| `Qwen3-0.6B-Q4_K_M.gguf` | 378 MB | LLM analysis | ✅ Downloaded |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | 610 MB | Embeddings (future) | ✅ Downloaded |
| `Qwen3-Reranker-0.6B-Q4_K_M.gguf` | 378 MB | Reranking (future) | ✅ Downloaded |

---

## 🚀 Usage Examples

```bash
cd /Volumes/APPLE-STORAGE/GitHub/code-similarity-engine

# Basic analysis (Python codebase)
PYTHONPATH=src venv/bin/python -m code_similarity_engine ./src -v

# Swift project with high threshold
PYTHONPATH=src venv/bin/python -m code_similarity_engine \
  /Volumes/APPLE-STORAGE/Tether/Tether \
  --focus "*.swift" -t 0.85 --max-chunks 500

# With LLM analysis
PYTHONPATH=src venv/bin/python -m code_similarity_engine ./src \
  --analyze \
  --llm-model "/Volumes/APPLE-STORAGE/Tether/Tether/Resources/ML Models/Qwen3-0.6B-Q4_K_M.gguf"

# JSON output for tooling
PYTHONPATH=src venv/bin/python -m code_similarity_engine ./src \
  -o json > report.json
```

---

## 📊 Test Results

### On code-similarity-engine itself (16 files, 61 chunks)
- Found 6 clusters at 80% threshold
- Top finding: `_ensure_parser()` duplicated 5x across language chunkers (88% similar)

### On Tether Swift codebase (146 files, 500 chunks)
- Found 93 clusters at 80% threshold
- Top findings:
  - `formatDate()` duplicated 3x in same file (100%)
  - `truncateTranscript()` duplicated across LLM files (100%)
  - Various prompt builders (98% similar)

---

## ⚠️ Known Limitations

1. **LLM Analysis Quality:** Small models (0.6B) sometimes echo template placeholders
2. **No Caching:** Re-embeds all files on each run
3. **No PyPI Package:** Can't `pip install` yet (pyproject.toml pending)
4. **Embedding via HF:** Still downloads from HuggingFace (not GGUF)

---

## 📦 Dependencies

```
# Core (no telemetry!)
click>=8.0
transformers>=4.30.0
torch>=2.0
scikit-learn>=1.0
numpy>=1.21
huggingface-hub>=0.20.0

# Tree-sitter
tree-sitter>=0.21
tree-sitter-python>=0.21
tree-sitter-swift>=0.21  # optional

# LLM Analysis
llama-cpp-python>=0.2.0  # optional
```

---

> **Next Steps:** See [ROADMAP.md](./ROADMAP.md) for planned enhancements.
