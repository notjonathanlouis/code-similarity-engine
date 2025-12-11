# Config Quick Start

## 30 Second Setup

```bash
# 1. Copy example config
cp .cserc.example .cserc

# 2. Edit to your needs
nano .cserc

# 3. Run CSE - it auto-loads your config
cse ./src
```

## Common Configs

### Python Project

```toml
[cse]
threshold = 0.85
exclude = ["**/tests/**", "**/__pycache__/**", "**/venv/**"]
focus = ["*.py"]
```

### JavaScript/TypeScript Project

```toml
[cse]
threshold = 0.85
exclude = ["**/node_modules/**", "**/dist/**", "**/build/**"]
focus = ["*.js", "*.ts", "*.jsx", "*.tsx"]
```

### Swift/iOS Project

```toml
[cse]
threshold = 0.90
exclude = ["**/Tests/**", "**/DerivedData/**", "**/.build/**"]
focus = ["*.swift"]
min_cluster = 3
```

### Multi-Language Project

```toml
[cse]
threshold = 0.85
exclude = [
    "**/tests/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/venv/**",
    "**/build/**",
    "**/dist/**",
]
rerank = true
verbose = true
```

### High Precision (Strict Matching)

```toml
[cse]
threshold = 0.95
min_cluster = 4
rerank_threshold = 0.7
```

### Fast Analysis (Less Precision)

```toml
[cse]
threshold = 0.75
analyze = false
rerank = false
```

## Priority Cheat Sheet

```
Built-in Default → Config File → CLI Argument
     (lowest)         (middle)       (highest)
```

**Example:**
- Default: `threshold = 0.80`
- Config: `threshold = 0.85`
- CLI: `--threshold 0.90`
- Result: **0.90** (CLI wins)

## File Locations

CSE looks for config in this order (first found wins):

```
./            # Current directory
../           # Parent directory
../../        # Grandparent directory
...           # (up to filesystem root)
```

**Pro tip:** Put config in project root, CSE finds it from any subdirectory!

## Common Mistakes

### ❌ Wrong section name
```toml
[config]  # Wrong!
threshold = 0.85
```

### ✅ Correct section name
```toml
[cse]  # Correct!
threshold = 0.85
```

### ❌ Wrong type
```toml
[cse]
exclude = "**/tests/**"  # Wrong - should be array
```

### ✅ Correct type
```toml
[cse]
exclude = ["**/tests/**"]  # Correct - array
```

### ❌ Invalid TOML
```toml
[cse]
threshold: 0.85  # Wrong - uses colon
```

### ✅ Valid TOML
```toml
[cse]
threshold = 0.85  # Correct - uses equals
```

## Testing Your Config

```bash
# See if config is loaded (look for "📝 Loaded config" message)
cse ./src -v

# Override config to test CLI priority
cse ./src --threshold 0.90 -v
```

## Debugging

**Config not loading?**

1. Check filename: `.cserc` or `.cse.toml` (note the dot!)
2. Check TOML syntax: `python3 -c "import tomllib; tomllib.load(open('.cserc', 'rb'))"`
3. Check section: `[cse]` (lowercase)
4. Run with `-v` to see verbose output

**Need Python < 3.11?**

```bash
pip install tomli
```

## Full Documentation

- Complete guide: [CONFIG.md](./CONFIG.md)
- All options: [.cserc.example](./.cserc.example)
- General usage: [README.md](./README.md)
