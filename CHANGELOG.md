# Changelog

All notable changes to code-similarity-engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-10

### Changed
- **BREAKING:** `-o/--output` now takes a file path instead of format name
  - Old: `cse ./src -o markdown > report.md`
  - New: `cse ./src -o report.md`
  - Format auto-detected from extension: `.md`, `.html`, `.json`, `.txt`
- **BREAKING:** Default behavior no longer outputs report
  - `cse ./src` now only does prep (index, embed, cluster, rerank) and saves to cache
  - Use `-o report.md` to export a report
- **BREAKING:** `--analyze` now defaults to OFF (was ON)
  - Run `cse ./src --analyze` to add LLM analysis
  - Analysis is saved to cache and included in subsequent exports
- Cache location now always in current working directory (`.cse_cache/`)
  - Previously created inside the analyzed path

### Added
- Incremental workflow: prep once, export multiple times, add analysis when ready
- Clean progress bars with no llama.cpp/ggml noise (Metal init messages suppressed)
- Helpful "Next steps" hints after prep completes
- Extension validation with helpful error messages

### Fixed
- Windows compatibility: `Path.replace()` instead of `Path.rename()` for atomic writes
- stderr suppression now properly catches all Metal initialization messages

## [0.2.0] - 2025-12-09

### Added
- Configuration file support via `.cserc` or `.cse.toml`
  - TOML-based project-level configuration
  - Automatic config discovery from current directory up to filesystem root
  - Priority system: built-in defaults < config file < explicit CLI args
  - All CLI options configurable via config file
  - Example config file included (`.cserc.example`)
  - Comprehensive documentation in `CONFIG.md`
  - Graceful degradation when TOML library unavailable
  - Support for Python 3.9+ via `tomli` backport

### Changed
- Updated `pyproject.toml` to include `tomli>=2.0` for Python < 3.11
- Version bump to 0.2.0

### Documentation
- Added `CONFIG.md` with complete configuration guide
- Updated `README.md` with configuration examples
- Added `.cserc.example` with annotated example config
- Added `.gitignore` to exclude local `.cserc` files

## [0.1.2] - Previous Release

(Version 0.1.2 changes not documented - add retroactively if needed)

## [0.1.0] - Initial Release

- Initial release with core functionality
- Semantic code similarity detection
- Tree-sitter based parsing
- Embedding-based clustering
- LLM analysis support
- Reranking support
- Multiple output formats (text, markdown, JSON)
