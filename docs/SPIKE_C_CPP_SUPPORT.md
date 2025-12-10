# SPIKE: C/C++ Language Support

**Status:** Ready to implement
**Effort:** ~30 min
**Priority:** Medium

## Summary

Add tree-sitter based chunking for C and C++ files. Infrastructure already exists (extension mappings, language detection). Just need chunker classes.

## Changes Required

### 1. pyproject.toml
```toml
"tree-sitter-c>=0.21",
"tree-sitter-cpp>=0.22",
```

### 2. New Files

| File | Based On | Lines |
|------|----------|-------|
| `languages/c.py` | `rust.py` | ~100 |
| `languages/cpp.py` | `c.py` | ~120 |

### 3. Update languages/__init__.py

Add to `SUPPORTED_LANGUAGES` and `_try_load_tree_sitter_chunker()`.

## C/C++ AST Nodes to Extract

**C:**
- `function_definition`
- `struct_specifier`
- `enum_specifier`

**C++ (additional):**
- `class_specifier`
- `namespace_definition`
- `lambda_expression`

## Test Command

```bash
cse ./some-c-project --focus "*.c" "*.cpp" "*.h" -v
```

## Notes

- C++ chunker can inherit from C chunker
- Header files (`.h`) default to C; could add heuristic for C++ detection
- Generic chunker already works as fallback
