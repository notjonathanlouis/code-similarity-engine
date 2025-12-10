# Changelog

All notable changes to code-similarity-engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
