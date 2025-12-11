# Code Similarity Engine

**Find duplicate code patterns and abstraction opportunities across your entire codebase.**

[![PyPI version](https://badge.fury.io/py/code-similarity-engine.svg)](https://pypi.org/project/code-similarity-engine/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---
## Models

CSE uses three Qwen3 GGUF models, all running 100% locally:

| Model | Size | Purpose |
|-------|------|---------|
| Qwen3-Embedding-0.6B | 610 MB | Semantic code embeddings |
| Qwen3-0.6B | 378 MB | LLM analysis & recommendations |
| Qwen3-Reranker-0.6B | 395 MB | Cross-encoder cluster validation |

Models are stored in `~/.cache/cse/models/` and downloaded automatically on first use.

---

## Privacy

**Your code never leaves your machine.**

- All models run locally via llama.cpp
- No telemetry, no analytics, no network calls
- Embedding cache stored locally only
- Open source - verify it yourself

---

## Why CSE?

**Code that performs similar functions can be abstracted, and such code emerges naturally throughout the lifetime of a project** but traditional tools are not robust against syntactical differences, and therefore miss many abstraciton opportunities. 

**We present a simple but robust CLI tool** that performs semantic embedding, comparison, and then k-means clustering to find regions of code showing high semantic similarity. This method finds code regions with similar meaning, and using reranking and setting minimums for similarity, this system can find regions of code that can be abstracted reliably and systematically. 

However, this only shows you where you can make abstractions, but not how. Using Qwen3 0.6B locally, we can go even further to provide a **step-by-step guide to abstracting these similarities**, making it trivial for you to **reduce code duplication and provide clearer design and syntax** throughout your project!

**What you get:**
- Simple, clean CLI tool
- On-device inferencing and systematic code similarity analysis
- Support for 16 languages and counting
- Beautiful reports in Markdown, HTML, or JSON
- 100% offline - your code never leaves your machine

---

## Quick Start

```bash
# Install
pip install code-similarity-engine

# Download models (~1.4 GB, one-time)
cse --download-models

# Prep your project (index, embed, cluster) - results cached to .cse_cache/
cse ./your-project

# Export a report
cse ./your-project -o report.html

# Add LLM analysis for refactoring recommendations
cse ./your-project --analyze

# Export again (now includes analysis)
cse ./your-project -o report.html
```

That's it! CSE will find similar code regions and explain how to refactor them.

---

## CLI Reference

```
cse <path> [options]

Workflow:
  cse ./src                  Prep: index, embed, cluster, rerank (cached)
  cse ./src -o report.md     Export report from cache
  cse ./src --analyze        Add LLM analysis to cache
  cse ./src -o report.html   Export again with analysis

Output:
  -o, --output FILE          Output file path (.md, .html, .json, .txt)
  --sort-by CRITERIA         severity | lines | files | similarity | quick-wins

Filtering:
  -f, --focus PATTERN        Only analyze matching files (repeatable)
  -e, --exclude PATTERN      Exclude matching files (repeatable)
  -l, --lang LANG            Force language detection

Thresholds:
  -t, --threshold FLOAT      Similarity threshold 0.0-1.0 (default: 0.80)
  -m, --min-cluster INT      Minimum chunks per cluster (default: 2)

AI Features:
  --analyze / --no-analyze   LLM explanations and recommendations (default: off)
  --rerank / --no-rerank     Cross-encoder cluster validation (default: on)

Advanced:
  --max-chunks INT           Maximum chunks to process (default: 10000)
  --min-lines INT            Minimum lines per chunk (default: 5)
  --max-lines INT            Maximum lines per chunk (default: 100)
  -q, --quiet                Suppress model loading messages
  -v, --verbose              Show detailed progress

Model Management:
  --download-models          Download all required models
  --model-status             Show status of all models
  --clear-cache              Clear embedding cache
```

---

## Features

### Semantic Code Analysis
Using tree-sitter as a foundation, CSE performs systematic chunking of the codebase, then embeds each chunk using the **Qwen3-0.6B Embedding** model to transform the chunks into meaning space. Next, CSE analyzes the chunks using k-means clustering to group chunks with similar meaning (and thus functionality). Finally, our program prompts **Qwen3-0.6B**, a tiny on-device LLM, with the chunks and instruction to find and identify abstraction opportunities within the chunks. 

**This fundamentally trivializes code cleanup in codebases of any size:** Simply generate a report and divide the tasks amongst your workforce. No brainstorming, no hour-long search and report sessions for code improvements, just the code regions to abstract and how to abstract them. 

### 16 Languages Supported
| Language | Language | Language | Language |
|----------|----------|----------|----------|
| Python | Swift | Rust | JavaScript |
| TypeScript | Go | C | C++ |
| Java | C# | PHP | Ruby |
| Kotlin | Scala | Lua | Bash |

### Multiple Output Formats

```bash
cse ./src -o report.md       # Markdown
cse ./src -o report.html     # Interactive HTML with dark mode
cse ./src -o report.json     # Machine-readable JSON
cse ./src -o report.txt      # Plain text
```

### Smart Sorting

```bash
cse ./src --sort-by severity     # Drift risk (default) - diverging copies first
cse ./src --sort-by quick-wins   # Easy fixes first
cse ./src --sort-by lines        # Most duplicated lines first
cse ./src --sort-by files        # Most spread across files first
```

### Project Configuration

Create a `.cserc` file in your project root:

```toml
[cse]
threshold = 0.85
exclude = ["**/tests/**", "**/node_modules/**", "**/vendor/**"]
focus = ["*.py", "*.js"]
```

---

## Usage Examples

```bash
# Typical workflow
cse ./src                              # Prep (fast, cached)
cse ./src -o report.md                 # Export basic report
cse ./src --analyze                    # Add LLM analysis (slower)
cse ./src -o report.html               # Export with recommendations

# Analyze a Swift iOS project
cse ./MyApp --focus "*.swift" -o swift_report.html

# Analyze Android project (Java + Kotlin)
cse ./android-app --focus "*.java" --focus "*.kt" -o android.md

# High-precision search (fewer, stronger matches)
cse ./src --threshold 0.90 --min-cluster 3 -o precise.md

# Quick scan without reranking (fastest)
cse ./src --no-rerank -o quick.md

# Quiet mode (suppress progress output)
cse ./src -q -o report.md

# Verbose mode (show all progress)
cse ./src -v
```

---

## How It Works

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Index     │ -> │   Embed     │ -> │   Cluster   │ -> │   Analyze   │
│  (parse)    │    │ (Qwen3)     │    │ (k-means)   │    │  (LLM)      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     │                   │                  │                   │
     v                   v                  v                   v
 Tree-sitter        Semantic            Similar             Natural
 extracts           vectors             regions             language
 functions          (768-dim)           grouped             recommendations
```

1. **Index**: Tree-sitter parses your code into semantic chunks (functions, methods, classes)
2. **Embed**: Qwen3-Embedding converts each chunk to a 768-dimensional vector
3. **Cluster**: Agglomerative clustering groups similar vectors
4. **Rerank**: Cross-encoder validates cluster members (optional, on by default)
5. **Analyze**: Qwen3-LLM explains what's similar and how to refactor (optional, on by default)

---

## Configuration File

CSE looks for `.cserc` or `.cse.toml` in your project directory (and parent directories).

```toml
[cse]
# Analysis settings
threshold = 0.85
min_cluster = 3
max_chunks = 10000

# File filtering
exclude = ["**/tests/**", "**/node_modules/**", "**/vendor/**", "**/__pycache__/**"]
focus = ["*.py", "*.js", "*.ts"]

# Features
analyze = false   # Set true to always run LLM analysis
rerank = true     # Cross-encoder validation (recommended)
sort_by = "severity"

# Display
verbose = false
quiet = false
```

**Priority**: CLI arguments > Config file > Defaults

---



## Example Output

```markdown
## Cluster 1: 95% Similarity

**5 regions** across **3 files** (~120 lines)

### Analysis

**What's similar:** These functions all implement retry logic with
exponential backoff. They handle HTTP requests, catch exceptions,
and retry with increasing delays.

**Recommendation:** Extract a generic `withRetry(operation, maxAttempts, backoffMs)`
function that accepts an async operation. Each call site can then use
`withRetry(() => fetch(url), 3, 1000)` instead of duplicating the retry logic.

**Suggested name:** `withExponentialBackoff`
**Complexity:** low

| File | Lines | Type | Name |
|------|-------|------|------|
| api/client.py | 45-67 | function | fetch_with_retry |
| api/upload.py | 23-48 | function | upload_with_retry |
| api/download.py | 31-55 | function | download_with_retry |
```

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Adding a new language:**
1. Check if tree-sitter grammar exists on PyPI
2. Create `languages/{lang}.py` following existing patterns
3. Add to `EXTENSION_MAP` and `SUPPORTED_LANGUAGES` in `__init__.py`
4. Add dependency to `pyproject.toml`

---

## License

GPL-3.0-or-later

---

## Acknowledgments

- [tree-sitter](https://tree-sitter.github.io/) - Incremental parsing
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Local LLM inference
- [Qwen3](https://huggingface.co/Qwen) - Embedding and LLM models
- Built with love by [Jonathan Louis](https://github.com/notjonathanlouis) using [Claude Code CLI](https://claude.com/product/claude-code)
