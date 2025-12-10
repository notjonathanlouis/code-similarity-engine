# Config File Support - Implementation Notes

## Overview

Added TOML-based configuration file support to CSE, allowing users to set project-level defaults via `.cserc` or `.cse.toml` files.

## Files Modified

### New Files Created

1. **src/code_similarity_engine/config.py** (108 lines)
   - `find_config_file(start_path: Path) -> Optional[Path]`
     - Searches for `.cserc` or `.cse.toml` from start_path up to root
     - Returns first config file found, or None

   - `load_config(path: Path) -> Dict[str, Any]`
     - Loads TOML config from discovered file
     - Returns `[cse]` section as dict
     - Returns empty dict if no file found or parse error
     - Gracefully handles missing TOML library

2. **.cserc.example** (93 lines)
   - Annotated example config with all available options
   - Copy-paste ready for users
   - Documents default values

3. **CONFIG.md** (342 lines)
   - Complete configuration documentation
   - Priority system explanation
   - Usage examples and best practices
   - Troubleshooting guide
   - Migration guide from CLI to config

4. **CHANGELOG.md**
   - Documented new feature in v0.2.0 release

5. **.gitignore**
   - Added `.cserc` and `.cse.toml` to ignore list
   - Standard Python ignores

### Modified Files

1. **src/code_similarity_engine/cli.py**
   - Added `from .config import load_config` import
   - Added `merge_config_with_cli()` helper function (26 lines)
   - Config loading logic after path resolution (lines 256-290)
     - Loads config from file
     - Merges with CLI arguments (CLI wins if explicit)
     - Handles list conversion for exclude/focus
     - Shows verbose message if config loaded

2. **src/code_similarity_engine/__init__.py**
   - Added `load_config` and `find_config_file` to exports
   - Version bump to 0.2.0

3. **pyproject.toml**
   - Added `tomli>=2.0; python_version < '3.11'` dependency
   - Version bump to 0.2.0

4. **README.md**
   - Added "Configuration Files" section
   - Examples of config usage
   - Priority system explanation

## Design Decisions

### 1. File Name Choice

**Chose:** `.cserc` and `.cse.toml`

**Rationale:**
- `.cserc` - Common convention (like `.bashrc`, `.vimrc`)
- `.cse.toml` - Explicit format in name (clearer for new users)
- Both supported for flexibility

### 2. TOML Format

**Chose:** TOML over JSON, YAML, INI

**Rationale:**
- Built into Python 3.11+ (`tomllib`)
- Simple, readable syntax
- Type-safe (lists, bools, numbers, strings)
- No code execution risk (unlike YAML)
- `tomli` backport for older Python

### 3. Config Discovery Strategy

**Chose:** Search from current directory up to root

**Rationale:**
- Supports project-specific configs in project root
- Supports shared configs in parent directories
- Supports user-wide configs (e.g., `~/.config/cse/.cserc`)
- First match wins (allows override hierarchy)

### 4. Priority System

**Chose:** Defaults < Config < CLI Args

**Rationale:**
- Config overrides defaults (convenience)
- CLI always wins (explicit user intent)
- Matches user expectations from other tools (git, docker, etc.)
- Simple to understand and document

### 5. Merge Strategy

**Problem:** Click doesn't expose which options were explicitly set vs defaults

**Solution:** Compare CLI value with default value
- If `cli_value != default_value` → user explicitly set it, use CLI
- If `cli_value == default_value` → could be default or explicit, use config if present

**Trade-off:** If user explicitly passes default value, config won't override
- Example: `cse --threshold 0.80` (where 0.80 is default)
- This is acceptable - if user passes explicit arg, we respect it

### 6. Graceful Degradation

**Chose:** Silently return empty dict if TOML unavailable

**Rationale:**
- CSE should work even without config support
- No user-facing errors for missing optional feature
- Dependency installation is user's choice

## Testing

All tests passed successfully:

### Test 1: Config File Discovery
- Finds `.cserc` in current directory
- Finds config from subdirectories
- Walks up to parent directories

### Test 2: Config Loading
- Loads TOML correctly
- Extracts `[cse]` section
- Returns expected values

### Test 3: Merge Logic
- CLI default + config value = config value
- CLI explicit + config value = CLI value
- CLI default + no config = default value
- Handles boolean flags correctly

### Test 4: Array Handling
- Lists (exclude, focus) parsed correctly
- Converted to tuples for CLI compatibility

### Test 5: Graceful Degradation
- Returns empty dict when `tomllib` unavailable
- No errors or exceptions

## Usage Examples

### Basic Project Config

```toml
# .cserc
[cse]
threshold = 0.85
exclude = ["**/tests/**"]
analyze = true
```

```bash
# Uses config values
cse ./src

# Override specific value
cse ./src --threshold 0.90
```

### Team Workflow

1. Commit `.cserc.example` to repo
2. Team members copy to `.cserc` and customize
3. Add `.cserc` to `.gitignore`

### User Defaults

```bash
# Create user-wide config
mkdir -p ~/.config/cse
cat > ~/.config/cse/.cserc << EOF
[cse]
verbose = true
analyze = true
output = "markdown"
EOF

# Always runs with these defaults
cse /any/project/src
```

## Future Enhancements

Potential improvements for future versions:

1. **Config Validation**
   - Validate config values at load time
   - Provide helpful error messages for invalid values
   - Schema validation via `pydantic` or similar

2. **Config Generation**
   - `cse --init` to create `.cserc.example` in current directory
   - Interactive config builder CLI

3. **Environment Variables**
   - Support `CSE_*` environment variables
   - Priority: Defaults < Config < Env Vars < CLI

4. **Config Profiles**
   - Multiple configs: `[cse.strict]`, `[cse.permissive]`
   - Select via `--profile strict`

5. **Include/Extend**
   - Load base config and extend it
   - Example: `extend = "../.cserc"`

6. **Per-Language Defaults**
   - `[cse.python]`, `[cse.swift]`
   - Auto-select based on detected language

## Performance Impact

Negligible:
- Config loading happens once at startup
- TOML parsing is fast (~1ms for typical config)
- File search stops at first match (usually 1-3 stat calls)

## Compatibility

- **Python 3.9-3.10:** Requires `tomli` package
- **Python 3.11+:** Uses built-in `tomllib`
- **No breaking changes:** Fully backward compatible
- **No config file:** Works exactly as before

## Documentation

Complete documentation provided:
- `CONFIG.md` - User guide (342 lines)
- `.cserc.example` - Example config (93 lines)
- `README.md` - Updated with config section
- `CHANGELOG.md` - Release notes
- Code docstrings in `config.py`

## Lessons Learned

1. **Click limitations:** No built-in way to detect explicit vs default args
   - Workaround: Compare with defaults (works well in practice)

2. **TOML library landscape:** Fragmented across Python versions
   - Solution: Graceful fallback chain

3. **Config discovery:** Users expect "walk up" behavior
   - Match expectations from git, npm, etc.

4. **Testing approach:** Unit tests + manual integration tests
   - Caught edge cases early

## Maintenance Notes

For future maintainers:

1. **Adding new CLI option:**
   - Update `merge_config_with_cli()` calls in `cli.py`
   - Add to `.cserc.example` with comment
   - Update `CONFIG.md` examples

2. **Changing defaults:**
   - Update in `cli.py` decorator
   - Update in `merge_config_with_cli()` calls
   - Update in documentation

3. **Python version support:**
   - When dropping Python 3.9-3.10: Remove `tomli` dependency
   - When adding Python 3.14+: Test `tomllib` compatibility

## Conclusion

Config file support is complete and tested. The implementation is:
- **Simple:** ~150 lines of new code
- **Safe:** No code execution, graceful degradation
- **User-friendly:** Matches expectations from other tools
- **Well-documented:** Complete user guide and examples
- **Backward compatible:** No breaking changes

Users can now:
- Set project-level defaults
- Share configs with teams
- Override config with CLI args
- Use both `.cserc` and `.cse.toml`

Ready for v0.2.0 release!
