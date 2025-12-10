# Sprint: CSE Phase 2-4 Implementation

**Created:** December 9, 2025
**Status:** In Progress

---

## Overview

This sprint consolidates all high-priority features into a single coordinated effort:
- GGUF-based embeddings (fully offline)
- Embedding caching (10x speedup)
- Better LLM prompts
- Reranker integration
- PyPI packaging with auto-download

---

## Tasks

### 1. GGUF Embeddings (Priority: Critical)

**Goal:** Replace HuggingFace `transformers` with `llama-cpp-python` for embeddings.

**Files to modify:**
- `src/code_similarity_engine/embedder.py`

**Implementation:**
```python
from llama_cpp import Llama

def embed_chunks_gguf(chunks, model_path, batch_size=32, verbose=False):
    """Embed using GGUF model via llama-cpp-python."""
    llm = Llama(
        model_path=str(model_path),
        embedding=True,  # Enable embedding mode
        n_ctx=512,
        verbose=False,
    )

    embeddings = []
    for chunk in chunks:
        text = _prepare_chunk_text(chunk)
        emb = llm.embed(text)  # Returns list of floats
        embeddings.append(emb)

    return np.array(embeddings)
```

**CLI addition:**
- `--embedding-gguf PATH` - Use GGUF model for embeddings instead of transformers

**Model:** `Qwen3-Embedding-0.6B-Q8_0.gguf` (already downloaded)

---

### 2. Embedding Caching (Priority: High)

**Goal:** Don't re-embed unchanged files. 10x speedup on subsequent runs.

**Files to create/modify:**
- `src/code_similarity_engine/cache.py` (new)
- `src/code_similarity_engine/embedder.py` (integrate cache)
- `src/code_similarity_engine/cli.py` (add --clear-cache)

**Cache location:** `.cse_cache/` in project root (where cse is run)

**Schema:**
```python
# SQLite schema
CREATE TABLE embeddings (
    file_path TEXT,
    chunk_hash TEXT,  # hash of chunk content
    mtime REAL,       # file modification time
    embedding BLOB,   # numpy array as bytes
    model_id TEXT,    # which model created this
    PRIMARY KEY (file_path, chunk_hash)
);
```

**Logic:**
1. Before embedding, check cache for each chunk
2. If cached AND mtime matches AND model_id matches → use cached
3. Otherwise → embed and store in cache
4. `--clear-cache` deletes `.cse_cache/`

---

### 3. Better LLM Prompts (Priority: High)

**Goal:** Improve Qwen3-0.6B analysis quality.

**Current issue:** Small models sometimes echo template placeholders.

**Files to modify:**
- `src/code_similarity_engine/analyzer.py`

**Improvements:**
1. Add few-shot example to prompt
2. Try `/think` mode for Qwen3 (reasoning before JSON)
3. Adjust temperature (try 0.1-0.2 for more deterministic output)
4. Add `--prompt-style` CLI option (default/thinking/verbose)

**Example few-shot prompt:**
```
Example input: 3 similar code regions found. Example: def format_date(d): return d.strftime('%Y-%m-%d')
Example output: {"commonality":"Date formatting to string","abstraction":"Extract to shared formatDate utility","suggested_name":"formatDateString","complexity":"low"}

Actual input: {cluster.size} similar code regions found. Example: {code_preview}
Output:
```

---

### 4. Reranker Integration (Priority: Medium)

**Goal:** Use `Qwen3-Reranker-0.6B` to improve cluster quality.

**Files to create/modify:**
- `src/code_similarity_engine/reranker.py` (new)
- `src/code_similarity_engine/cli.py` (add --rerank)
- `src/code_similarity_engine/clusterer.py` (integrate reranker)

**Implementation:**
```python
def rerank_cluster(cluster, reranker_model_path):
    """Rerank cluster members by similarity to representative."""
    # Use reranker to score each chunk against representative
    # Filter out low-scoring chunks (threshold ~0.5)
    # Return refined cluster
```

**Model:** `Qwen3-Reranker-0.6B-Q4_K_M.gguf` (already downloaded)

---

### 5. PyPI Package (Priority: High)

**Goal:** `pip install code-similarity-engine`

**Files to create:**
- `pyproject.toml` (proper build config)
- `LICENSE` (GPL-3.0)
- `README.md` (usage guide only - no description that triggers content policy)

**pyproject.toml:**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "code-similarity-engine"
version = "0.1.0"
description = ""  # Left blank intentionally
readme = "README.md"
license = {text = "GPL-3.0"}
requires-python = ">=3.9"
dependencies = [
    "click>=8.0",
    "numpy>=1.21",
    "scikit-learn>=1.0",
    "tree-sitter>=0.21",
    "llama-cpp-python>=0.2.0",
]

[project.optional-dependencies]
transformers = [
    "transformers>=4.30.0",
    "torch>=2.0",
]

[project.scripts]
cse = "code_similarity_engine.cli:cli"
```

---

### 6. Model Auto-Download (Priority: High)

**Goal:** Models download automatically on first run.

**Files to create/modify:**
- `src/code_similarity_engine/models.py` (add download functions)
- `src/code_similarity_engine/cli.py` (check and download on startup)

**Implementation:**
```python
MODELS = {
    "llm": {
        "repo": "unsloth/Qwen3-0.6B-GGUF",
        "file": "Qwen3-0.6B-Q4_K_M.gguf",
        "size_mb": 378,
    },
    "embedding": {
        "repo": "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "file": "Qwen3-Embedding-0.6B-Q8_0.gguf",
        "size_mb": 610,
    },
    "reranker": {
        "repo": "sinjab/Qwen3-Reranker-0.6B-Q4_K_M-GGUF",
        "file": "Qwen3-Reranker-0.6B-Q4_K_M.gguf",
        "size_mb": 378,
    },
}

def ensure_models_downloaded(cache_dir=None, verbose=False):
    """Download models if not present."""
    from huggingface_hub import hf_hub_download
    # Check each model, download if missing
```

**Cache location:** `~/.cache/cse/models/`

---

## Execution Plan

### Parallelizable (can use agents):
1. **Cache module** - Independent, clear spec
2. **Reranker module** - Independent, clear spec
3. **pyproject.toml + LICENSE** - Independent, simple

### Sequential (need iteration):
1. **GGUF embeddings** - Core change, needs testing
2. **LLM prompts** - Needs experimentation
3. **Model auto-download** - Depends on package structure

---

## Testing Plan

After each feature:
```bash
cd /Volumes/APPLE-STORAGE/GitHub/code-similarity-engine
PYTHONPATH=src venv/bin/python -m code_similarity_engine ./src -v

# With GGUF embeddings:
PYTHONPATH=src venv/bin/python -m code_similarity_engine ./src \
  --embedding-gguf "/Volumes/APPLE-STORAGE/Tether/Tether/Resources/ML Models/Qwen3-Embedding-0.6B-Q8_0.gguf" -v

# With reranker:
PYTHONPATH=src venv/bin/python -m code_similarity_engine ./src \
  --rerank -v
```

---

## Notes

- **Content policy:** Keep README/pyproject.toml description minimal or blank
- **Models location:** Currently in Tether's ML Models dir, will move to ~/.cache/cse/models/
- **Agents:** Use for cache.py, reranker.py, pyproject.toml - clear specs, independent work
