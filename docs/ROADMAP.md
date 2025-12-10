# Code Similarity Engine - Roadmap

**Last Updated:** December 9, 2025

> See also: [Current Status](./CURRENT_STATUS.md)

---

## 🎯 Vision

A fully offline, privacy-respecting code analysis tool that helps developers and LLMs find abstraction opportunities through semantic similarity.

---

## 📋 Phase 2: Embedding Improvements

### 2.1 GGUF Embeddings (Priority: High)
**Goal:** Replace HuggingFace transformers with llama.cpp embeddings for fully offline operation.

- [ ] Integrate `Qwen3-Embedding-0.6B-Q8_0.gguf` via llama-cpp-python
- [ ] Add `--embedding-gguf` CLI option for GGUF model path
- [ ] Benchmark speed/quality vs transformers
- [ ] Update embedder.py to support both modes

**Why:** Eliminates network dependency, faster cold start, consistent with LLM approach.

### 2.2 Embedding Caching (Priority: Medium)
**Goal:** Don't re-embed unchanged files.

- [ ] Create `.cse_cache/embeddings.db` (SQLite)
- [ ] Store: file_path, mtime, content_hash, embedding_vector
- [ ] Check cache before embedding, only embed changed files
- [ ] Add `--clear-cache` CLI option

**Why:** 10x speedup on subsequent runs.

---

## 📋 Phase 3: LLM Analysis Quality

### 3.1 Better Prompts (Priority: High)
**Goal:** Improve Qwen3-0.6B analysis quality.

- [ ] Add few-shot examples to prompt
- [ ] Try different prompt templates (Alpaca, Vicuna, raw)
- [ ] Experiment with temperature/top_p settings
- [ ] Add `--prompt-style` CLI option

**Current Issue:** Small models sometimes echo template placeholders instead of analyzing.

### 3.2 Reranker Integration (Priority: Medium)
**Goal:** Use `Qwen3-Reranker-0.6B` to improve cluster quality.

- [ ] Add reranker as optional post-processing step
- [ ] Rerank cluster members by similarity to centroid
- [ ] Filter low-quality matches
- [ ] Add `--rerank` CLI option

**Why:** Better precision, fewer false positives.

---

## 📋 Phase 4: Output & Integration

### 4.1 PyPI Package (Priority: High)
**Goal:** `pip install code-similarity-engine`

- [ ] Create proper pyproject.toml (with safe description)
- [ ] Add `cse` console script entry point
- [ ] Set up GitHub Actions for PyPI publishing
- [ ] Add LICENSE file (GPL-3.0)

### 4.2 IDE Integration (Priority: Low)
**Goal:** Show similarity hints in editors.

- [ ] VS Code extension (read JSON output)
- [ ] Highlight similar regions with decorations
- [ ] Quick action: "Show similar code"

---

## 📋 Phase 5: Advanced Features

### 5.1 Cross-Repository Analysis (Priority: Low)
**Goal:** Find patterns across multiple projects.

- [ ] Accept multiple paths as arguments
- [ ] Merge embeddings from different repos
- [ ] Report which patterns appear in which repos

### 5.2 Auto-Refactor Suggestions (Priority: Low)
**Goal:** Generate actual refactored code.

- [ ] For each cluster, generate extracted function
- [ ] Show diff of changes needed
- [ ] Optional: apply changes automatically

### 5.3 Watch Mode (Priority: Low)
**Goal:** Re-analyze on file changes.

- [ ] Use `watchdog` for file system events
- [ ] Incremental re-embedding
- [ ] Live-updating report

---

## 🐛 Known Bugs / Tech Debt

| Issue | Priority | Notes |
|-------|----------|-------|
| No tests | High | Need pytest fixtures with sample code |
| No type hints in some files | Medium | Add mypy checking |
| Verbose llama.cpp output | Low | Suppress kernel warnings in non-verbose mode |
| Progress bar overlap | Low | tqdm + click.echo conflict |

---

## 💡 Ideas (Unplanned)

- **Visualization:** 2D/3D embedding space plot with cluster highlighting
- **Learning:** Track accepted/rejected suggestions to tune threshold
- **Git Integration:** Only analyze changed files since last commit
- **Language Server Protocol:** LSP server for IDE integration
- **Web UI:** Simple Flask app for non-CLI users

---

## 📅 Timeline

| Phase | Target | Status |
|-------|--------|--------|
| Phase 1: MVP | Dec 2025 | ✅ Complete |
| Phase 2: Embeddings | Jan 2026 | 🔜 Next |
| Phase 3: LLM Quality | Jan 2026 | Planned |
| Phase 4: PyPI | Feb 2026 | Planned |
| Phase 5: Advanced | Q2 2026 | Wishlist |

---

> **Current Status:** See [CURRENT_STATUS.md](./CURRENT_STATUS.md) for what's working today.
