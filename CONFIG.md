# Configuration File Support

CSE supports project-level configuration via `.cserc` or `.cse.toml` files.

## Quick Start

Create a `.cserc` file in your project root:

```toml
[cse]
threshold = 0.85
min_cluster = 3
exclude = ["**/tests/**", "**/node_modules/**"]
focus = ["*.py", "*.js"]
analyze = true
```

Run CSE - it will automatically find and use your config:

```bash
cse ./src
```

## Config File Discovery

CSE searches for config files in this order:

1. `.cserc` in current directory
2. `.cse.toml` in current directory
3. `.cserc` in parent directory
4. `.cse.toml` in parent directory
5. (continues up to filesystem root)

The first file found is used. This allows:
- Project-specific configs in project root
- Shared configs in parent directories
- User-wide configs in home directory

## Priority System

Settings are applied in this order (later wins):

1. **Built-in defaults** (e.g., `threshold = 0.80`)
2. **Config file values** (if found)
3. **Explicit CLI arguments** (always win)

### Examples

**Config file (.cserc):**
```toml
[cse]
threshold = 0.85
output = "markdown"
```

**Command: `cse ./src`**
- Uses `threshold = 0.85` (from config)
- Uses `output = "markdown"` (from config)

**Command: `cse ./src --threshold 0.90`**
- Uses `threshold = 0.90` (explicit CLI arg wins)
- Uses `output = "markdown"` (from config)

**Command: `cse ./src --threshold 0.90 -o json`**
- Uses `threshold = 0.90` (explicit CLI arg wins)
- Uses `output = "json"` (explicit CLI arg wins)

## Available Options

All CLI options can be configured via config file:

### Core Settings

```toml
[cse]
# Similarity threshold (0.0-1.0)
threshold = 0.85

# Minimum chunks per cluster
min_cluster = 3

# Output format: "text", "markdown", or "json"
output = "markdown"

# Show verbose progress
verbose = true
```

### Filtering

```toml
[cse]
# Patterns to exclude (list of glob patterns)
exclude = [
    "**/tests/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/venv/**",
]

# Only analyze matching paths (list of glob patterns)
focus = [
    "*.py",
    "*.js",
    "*.swift",
]

# Force language detection
lang = "python"
```

### Chunking

```toml
[cse]
# Maximum chunks to process
max_chunks = 15000

# Minimum lines per chunk
min_lines = 10

# Maximum lines per chunk
max_lines = 150
```

### LLM Analysis

```toml
[cse]
# Use LLM to explain clusters
analyze = true

# Path to LLM GGUF model (optional, auto-detected if omitted)
llm_model = "/path/to/model.gguf"

# Max clusters to analyze (optional, analyzes all if omitted)
max_analyze = 10
```

### Reranking

```toml
[cse]
# Use reranker to improve cluster quality
rerank = true

# Path to reranker GGUF model (optional, auto-detected if omitted)
rerank_model = "/path/to/reranker.gguf"

# Minimum reranker score to keep
rerank_threshold = 0.6
```

### Embedding

```toml
[cse]
# Embedding batch size
batch_size = 64
```

## Complete Example

See [.cserc.example](./.cserc.example) for a complete annotated config file.

## Best Practices

### Project-Level Config

Recommended for projects with specific coding standards:

```toml
# .cserc in project root
[cse]
threshold = 0.90        # High precision for strict matching
min_cluster = 3         # Only show patterns that repeat 3+ times
exclude = [
    "**/tests/**",      # Ignore test code
    "**/generated/**",  # Ignore auto-generated code
]
focus = ["*.swift"]     # Swift-only project
```

### Team Config

Share configs with your team by committing `.cserc.example`:

```bash
# Copy and customize for your machine
cp .cserc.example .cserc

# Add to .gitignore to avoid conflicts
echo ".cserc" >> .gitignore
```

### User-Wide Defaults

Create `~/.config/cse/.cserc` for personal defaults across all projects:

```toml
[cse]
verbose = true
analyze = true
rerank = true
output = "markdown"
```

Then override per-project as needed.

## Debugging

To see if a config file is being used:

```bash
# Enable verbose mode
cse ./src -v

# Look for this line in output:
# 📝 Loaded config from .cserc/.cse.toml
```

To check which config file would be used:

```python
from pathlib import Path
from code_similarity_engine import find_config_file

config_path = find_config_file(Path("."))
print(f"Config file: {config_path}")
```

## Migration Guide

### From CLI to Config

**Before:**
```bash
cse ./src \
  --threshold 0.85 \
  --min-cluster 3 \
  --exclude "**/tests/**" \
  --exclude "**/node_modules/**" \
  --focus "*.py" \
  --analyze \
  --rerank \
  -o markdown
```

**After:**

Create `.cserc`:
```toml
[cse]
threshold = 0.85
min_cluster = 3
exclude = ["**/tests/**", "**/node_modules/**"]
focus = ["*.py"]
analyze = true
rerank = true
output = "markdown"
```

Run:
```bash
cse ./src
```

## Troubleshooting

### Config Not Loading

**Problem:** Config file exists but settings aren't applied

**Solutions:**
1. Check file format - must be valid TOML
2. Check section name - must be `[cse]` (lowercase)
3. Run with `-v` to verify config is loaded
4. Ensure no TOML syntax errors

**Validate TOML:**
```python
import tomllib
with open(".cserc", "rb") as f:
    config = tomllib.load(f)
    print(config)
```

### Python < 3.11 Support

CSE automatically uses `tomli` package on Python < 3.11. If you get import errors:

```bash
pip install tomli
```

Or upgrade Python to 3.11+.

### Config Ignored

If explicit CLI arguments are used, they override config:

```bash
# This ALWAYS uses threshold 0.90, regardless of config
cse ./src --threshold 0.90
```

To use config value, omit the CLI argument:

```bash
# Uses threshold from config
cse ./src
```

## Technical Details

### Implementation

- Config loading: `src/code_similarity_engine/config.py`
- CLI integration: `src/code_similarity_engine/cli.py`
- Merge logic: `merge_config_with_cli()` function

### File Format

CSE uses [TOML](https://toml.io/) format:
- Simple, readable syntax
- Built-in Python support (3.11+)
- Type-safe (lists, bools, numbers, strings)

### Graceful Degradation

If TOML library is unavailable:
- Config loading silently returns empty dict
- CSE continues with CLI defaults
- No errors or warnings

### Security

- Config files are read-only
- No code execution
- No environment variable expansion
- No shell command interpolation
- Safe for untrusted repositories (but verify contents!)

## See Also

- [README.md](./README.md) - General usage
- [.cserc.example](./.cserc.example) - Complete config example
- [TOML Spec](https://toml.io/) - TOML format documentation
