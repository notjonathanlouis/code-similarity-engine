# ROADMAP: Universal Language Support

**Goal:** Make CSE work with EVERY programming language, hardware description language, configuration format, and markup language that has a tree-sitter grammar.

**Status:** Planning
**Source:** [Tree-sitter Wiki - List of Parsers](https://github.com/tree-sitter/tree-sitter/wiki/List-of-parsers)

---

## Implementation Strategy

### How We Add Languages

Each language needs:
1. **PyPI package** - `tree-sitter-{lang}` (most exist!)
2. **Chunker class** - `languages/{lang}.py` (~80-150 lines each)
3. **Extension mapping** - Add to `EXTENSION_MAP` in `__init__.py`
4. **AST node types** - What to extract (functions, classes, etc.)

### Chunker Inheritance Tree

```
BaseChunker
├── GenericChunker (fallback - line-based sliding window)
├── CStyleChunker (shared logic for C-family)
│   ├── CChunker
│   ├── CppChunker
│   ├── JavaChunker
│   ├── CSharpChunker
│   ├── GoChunker (already done)
│   └── RustChunker (already done)
├── FunctionalChunker (shared logic for FP langs)
│   ├── HaskellChunker
│   ├── OCamlChunker
│   ├── ElixirChunker
│   └── ClojureChunker
├── ScriptingChunker (shared logic for dynamic langs)
│   ├── PythonChunker (already done)
│   ├── RubyChunker
│   ├── PerlChunker
│   └── LuaChunker
├── ShellChunker (shared logic for shells)
│   ├── BashChunker
│   ├── FishChunker
│   └── PowerShellChunker
├── HDLChunker (hardware description languages)
│   ├── VerilogChunker
│   ├── VHDLChunker
│   └── SystemVerilogChunker
└── ConfigChunker (config/data formats)
    ├── YAMLChunker
    ├── TOMLChunker
    └── TerraformChunker
```

---

## Priority Tiers

### Tier 1: Maximum Impact (Week 1)
*Most-used languages, biggest user bases*

| Language | PyPI Package | Extensions | Est. Effort |
|----------|--------------|------------|-------------|
| **Java** | `tree-sitter-java` | `.java` | 20 min |
| **C#** | `tree-sitter-c-sharp` | `.cs` | 20 min |
| **PHP** | `tree-sitter-php` | `.php` | 20 min |
| **Ruby** | `tree-sitter-ruby` | `.rb` | 15 min |
| **Kotlin** | `tree-sitter-kotlin` | `.kt`, `.kts` | 20 min |
| **Scala** | `tree-sitter-scala` | `.scala`, `.sc` | 20 min |
| **Lua** | `tree-sitter-lua` | `.lua` | 15 min |
| **Bash** | `tree-sitter-bash` | `.sh`, `.bash` | 15 min |

**Total: ~2.5 hours**

### Tier 2: DevOps & Infrastructure (Week 2)
*Every platform engineer needs this*

| Language | PyPI Package | Extensions | Est. Effort |
|----------|--------------|------------|-------------|
| **Terraform/HCL** | `tree-sitter-hcl` | `.tf`, `.hcl` | 20 min |
| **YAML** | `tree-sitter-yaml` | `.yml`, `.yaml` | 15 min |
| **Dockerfile** | `tree-sitter-dockerfile` | `Dockerfile` | 10 min |
| **SQL** | `tree-sitter-sql` | `.sql` | 20 min |
| **Makefile** | `tree-sitter-make` | `Makefile` | 15 min |
| **CMake** | `tree-sitter-cmake` | `CMakeLists.txt` | 15 min |
| **Nginx** | `tree-sitter-nginx` | `nginx.conf` | 10 min |
| **TOML** | `tree-sitter-toml` | `.toml` | 10 min |
| **JSON** | `tree-sitter-json` | `.json` | 10 min |

**Total: ~2 hours**

### Tier 3: Hardware & Embedded (Week 3)
*EE folks deserve love too*

| Language | PyPI Package | Extensions | Est. Effort |
|----------|--------------|------------|-------------|
| **Verilog** | `tree-sitter-verilog` | `.v`, `.vh` | 25 min |
| **VHDL** | `tree-sitter-vhdl` | `.vhd`, `.vhdl` | 25 min |
| **SystemVerilog** | `tree-sitter-systemverilog` | `.sv`, `.svh` | 25 min |
| **ARM Assembly** | `tree-sitter-asm` | `.s`, `.S`, `.asm` | 30 min |
| **CUDA** | `tree-sitter-cuda` | `.cu`, `.cuh` | 20 min |
| **GLSL** | `tree-sitter-glsl` | `.glsl`, `.vert`, `.frag` | 20 min |
| **WGSL** | `tree-sitter-wgsl` | `.wgsl` | 15 min |

**Total: ~2.5 hours**

### Tier 4: Functional & Academic (Week 4)
*PLT enthusiasts and academics*

| Language | PyPI Package | Extensions | Est. Effort |
|----------|--------------|------------|-------------|
| **Haskell** | `tree-sitter-haskell` | `.hs` | 25 min |
| **OCaml** | `tree-sitter-ocaml` | `.ml`, `.mli` | 25 min |
| **Elixir** | `tree-sitter-elixir` | `.ex`, `.exs` | 20 min |
| **Erlang** | `tree-sitter-erlang` | `.erl`, `.hrl` | 20 min |
| **Clojure** | `tree-sitter-clojure` | `.clj`, `.cljs`, `.cljc` | 20 min |
| **F#** | `tree-sitter-fsharp` | `.fs`, `.fsi`, `.fsx` | 20 min |
| **Elm** | `tree-sitter-elm` | `.elm` | 15 min |
| **Julia** | `tree-sitter-julia` | `.jl` | 20 min |
| **R** | `tree-sitter-r` | `.r`, `.R` | 15 min |

**Total: ~3 hours**

### Tier 5: Web3 & Emerging (Week 5)
*Blockchain devs have SO much copy-paste*

| Language | PyPI Package | Extensions | Est. Effort |
|----------|--------------|------------|-------------|
| **Solidity** | `tree-sitter-solidity` | `.sol` | 20 min |
| **Move** | `tree-sitter-move` | `.move` | 20 min |
| **Cairo** | `tree-sitter-cairo` | `.cairo` | 20 min |
| **Vyper** | (via Python) | `.vy` | 10 min |
| **Noir** | `tree-sitter-noir` | `.nr` | 20 min |

**Total: ~1.5 hours**

### Tier 6: Game Development (Week 6)
*Game devs LOVE to copy-paste*

| Language | PyPI Package | Extensions | Est. Effort |
|----------|--------------|------------|-------------|
| **GDScript** | `tree-sitter-gdscript` | `.gd` | 20 min |
| **Luau** | `tree-sitter-luau` | `.luau` | 15 min |
| **HLSL** | `tree-sitter-hlsl` | `.hlsl` | 20 min |
| **ShaderLab** | (custom) | `.shader` | 25 min |

**Total: ~1.5 hours**

### Tier 7: Documentation & Markup (Week 7)
*Even docs have patterns*

| Language | PyPI Package | Extensions | Est. Effort |
|----------|--------------|------------|-------------|
| **Markdown** | `tree-sitter-markdown` | `.md` | 15 min |
| **LaTeX** | `tree-sitter-latex` | `.tex` | 20 min |
| **reStructuredText** | `tree-sitter-rst` | `.rst` | 15 min |
| **AsciiDoc** | `tree-sitter-asciidoc` | `.adoc` | 15 min |

**Total: ~1 hour**

---

## AST Node Extraction Guide

### C-Family (C, C++, Java, C#, Go, Rust, Kotlin, Scala)
```
function_definition / method_declaration
class_declaration / struct_specifier
interface_declaration / trait_item
enum_specifier / enum_declaration
namespace_definition / module_declaration
```

### Functional (Haskell, OCaml, Elixir, Clojure)
```
function_clause / function_definition
module_definition
type_declaration / data_declaration
pattern_matching blocks
```

### Scripting (Python, Ruby, Perl, Lua, Bash)
```
function_definition
class_definition
method_definition
block / do_block
```

### Hardware (Verilog, VHDL, SystemVerilog)
```
module_declaration
always_block / process_statement
function_declaration / task_declaration
generate_block
```

### Config (Terraform, YAML, TOML)
```
block / resource_block
mapping / sequence
table / array_of_tables
```

---

## Implementation Order

```
Phase 1 (v0.2.0): Tier 1 - Java, C#, PHP, Ruby, Kotlin, Scala, Lua, Bash
Phase 2 (v0.3.0): Tier 2 - Terraform, YAML, SQL, Dockerfile, Make, CMake
Phase 3 (v0.4.0): Tier 3 - Verilog, VHDL, SystemVerilog, ARM, CUDA, GLSL
Phase 4 (v0.5.0): Tier 4 - Haskell, OCaml, Elixir, Erlang, Clojure, F#
Phase 5 (v0.6.0): Tier 5 + 6 - Solidity, Move, GDScript, shaders
Phase 6 (v0.7.0): Tier 7 + remaining - Docs, markup, edge cases
Phase 7 (v1.0.0): Polish, test all languages, comprehensive docs
```

---

## Batch Implementation Script

We can generate chunkers programmatically! See `scripts/generate_chunker.py`:

```python
# Template-based chunker generation
# Input: language name, AST nodes, extensions
# Output: languages/{lang}.py with proper chunker class
```

---

## Testing Strategy

1. **Unit tests per language** - Sample files with known chunks
2. **Integration test** - Run on real repos (Linux kernel, React, etc.)
3. **Benchmark** - Measure chunking speed per language
4. **CI matrix** - Test all languages on Python 3.9-3.12

---

## Community Contributions

Make it EASY for people to add languages:
1. `CONTRIBUTING.md` with step-by-step guide
2. Issue template for "Add language: X"
3. Chunker generator script
4. Test file templates

---

## Notes

- Some languages share PyPI packages (e.g., `tree-sitter-javascript` handles TypeScript)
- Some grammars may not have PyPI packages yet - we can build from source
- Assembly variants may need multiple extensions (ARM vs x86 vs RISC-V)
- Config files often don't have "functions" - chunk by top-level blocks instead

---

## Sources

- [Tree-sitter Wiki - List of Parsers](https://github.com/tree-sitter/tree-sitter/wiki/List-of-parsers)
- [Tree-sitter Grammars Org](https://github.com/tree-sitter-grammars)
- [py-tree-sitter-languages](https://github.com/grantjenks/py-tree-sitter-languages)

---

*"Every developer deserves to know where their duplication lives."*
