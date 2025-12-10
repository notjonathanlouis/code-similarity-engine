# Migration Guide: v0.1.x → v0.2.0

## What's New

Version 0.2.0 adds **configuration file support**. You can now store your preferred CSE settings in a `.cserc` or `.cse.toml` file instead of typing them every time.

## Do I Need to Migrate?

**No!** v0.2.0 is **100% backward compatible**.

- All existing CLI commands work exactly as before
- Config files are **optional**
- No breaking changes

## Quick Start (Optional)

If you want to use config files:

```bash
# 1. Create a config file
cat > .cserc << 'EOF'
[cse]
threshold = 0.85
exclude = ["**/tests/**", "**/node_modules/**"]
analyze = true
EOF

# 2. Run CSE (auto-loads config)
cse ./src
```

That's it! See [QUICKSTART_CONFIG.md](./QUICKSTART_CONFIG.md) for more examples.

## Converting Your Workflow

### Before (v0.1.x)

If you were running CSE with the same arguments every time:

```bash
# Typed manually each time
cse ./src \
  --threshold 0.85 \
  --exclude "**/tests/**" \
  --exclude "**/node_modules/**" \
  --analyze \
  -o markdown
```

### After (v0.2.0)

Create `.cserc`:

```toml
[cse]
threshold = 0.85
exclude = ["**/tests/**", "**/node_modules/**"]
analyze = true
output = "markdown"
```

Run:

```bash
# Much shorter!
cse ./src
```

## Shell Aliases → Config Files

### Before

If you had shell aliases:

```bash
# .bashrc or .zshrc
alias cse-strict='cse --threshold 0.90 --min-cluster 3'
alias cse-python='cse --focus "*.py" --exclude "**/tests/**"'
```

### After

Replace with config files:

```bash
# Project A: .cserc
[cse]
threshold = 0.90
min_cluster = 3

# Project B: .cserc
[cse]
focus = ["*.py"]
exclude = ["**/tests/**"]
```

Remove aliases, run `cse ./src` in each project.

## CI/CD Pipelines

Config files work great in CI/CD:

```yaml
# .github/workflows/cse.yml
- name: Check code similarity
  run: |
    pip install code-similarity-engine
    cse ./src  # Uses .cserc from repo
```

Commit `.cserc` to your repo for consistent team settings.

## Monorepo Support

Different configs for different packages:

```
monorepo/
├── .cserc              # Root config (defaults)
├── backend/
│   └── .cserc          # Backend-specific config
└── frontend/
    └── .cserc          # Frontend-specific config
```

```bash
# Each uses its own config
cse monorepo/backend
cse monorepo/frontend
```

## New Dependencies

v0.2.0 adds one optional dependency:

- **Python 3.11+:** No new dependencies (uses built-in `tomllib`)
- **Python 3.9-3.10:** Adds `tomli` package

To install:

```bash
pip install --upgrade code-similarity-engine
```

If on Python < 3.11 and `tomli` doesn't auto-install:

```bash
pip install tomli
```

## Behavior Changes

**None!** All existing behavior preserved:

- CLI arguments work identically
- Default values unchanged
- Output formats unchanged
- Model loading unchanged

## FAQ

### Do I need to create a config file?

**No.** Config files are optional. CSE works exactly as before if no config file exists.

### Can I mix CLI args and config files?

**Yes!** CLI arguments always override config values. Example:

```bash
# Config has threshold = 0.85
cse ./src --threshold 0.90  # Uses 0.90 (CLI wins)
```

### What if my team uses different settings?

Option 1: Commit `.cserc.example`, let team members copy and customize:

```bash
cp .cserc.example .cserc
# Edit .cserc for personal preferences
```

Option 2: Commit a shared `.cserc`, override locally with CLI args.

### Does this work with scripts?

**Yes!** Config files work perfectly with scripts:

```bash
#!/bin/bash
# analyze.sh - Uses .cserc from current directory
cse ./src -o json > report.json
```

### Can I use environment variables?

Not yet, but planned for future release. For now, use config files or CLI args.

### What about Windows?

Config files work identically on Windows:

```powershell
# PowerShell
cse .\src
```

Use `.cserc` or `.cse.toml` (same as macOS/Linux).

## Rollback

If you need to revert to v0.1.x:

```bash
pip install code-similarity-engine==0.1.2
```

(Your `.cserc` files will be ignored, no cleanup needed)

## Get Help

- **Config documentation:** [CONFIG.md](./CONFIG.md)
- **Quick examples:** [QUICKSTART_CONFIG.md](./QUICKSTART_CONFIG.md)
- **Example config:** [.cserc.example](./.cserc.example)
- **General docs:** [README.md](./README.md)

## Feedback

Found a bug or have suggestions? Please open an issue on GitHub.

Enjoy CSE v0.2.0!
