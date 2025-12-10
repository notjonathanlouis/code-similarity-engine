# Code Similarity Engine - Find and analyze duplicate code patterns
# Copyright (C) 2025  Jonathan Louis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Report generator - formats cluster results for output.

Supports text, markdown, and json output formats.
Includes optional LLM analysis when available.
"""

from typing import List, Optional, Dict
from pathlib import Path
from enum import Enum
import json
from datetime import datetime

from .models import Cluster, CodeChunk


class OutputFormat(Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


def report_clusters(
    clusters: List[Cluster],
    root_path: Path,
    threshold: float,
    output_format: OutputFormat = OutputFormat.TEXT,
    analyses: Optional[Dict[int, "ClusterAnalysis"]] = None,
) -> str:
    """
    Generate a report of similar code clusters.

    Args:
        clusters: List of Cluster objects
        root_path: Root path (for display)
        threshold: Similarity threshold used
        output_format: Desired output format
        analyses: Optional dict of cluster_id -> ClusterAnalysis

    Returns:
        Formatted report string
    """
    analyses = analyses or {}

    if output_format == OutputFormat.TEXT:
        return _format_text(clusters, root_path, threshold, analyses)
    elif output_format == OutputFormat.MARKDOWN:
        return _format_markdown(clusters, root_path, threshold, analyses)
    elif output_format == OutputFormat.JSON:
        return _format_json(clusters, root_path, threshold, analyses)
    elif output_format == OutputFormat.HTML:
        return _format_html(clusters, root_path, threshold, analyses)
    else:
        raise ValueError(f"Unknown format: {output_format}")


def _format_text(
    clusters: List[Cluster],
    root_path: Path,
    threshold: float,
    analyses: Dict[int, "ClusterAnalysis"],
) -> str:
    """Plain text format with unicode decorations."""
    lines = []

    # Header
    total_lines = sum(c.total_lines() for c in clusters)
    lines.append(f"🔍 Found {len(clusters)} similarity clusters in {root_path}")
    lines.append(f"   Threshold: {threshold:.0%} | Total duplicated lines: ~{total_lines}")
    if analyses:
        lines.append(f"   LLM analysis: {len(analyses)} clusters analyzed")
    lines.append("")

    for cluster in clusters:
        lines.append("━" * 70)
        lines.append(f"Cluster #{cluster.id}: Similarity {cluster.similarity_score:.0%}")
        lines.append(f"Files: {cluster.file_count} | Regions: {cluster.size} | Lines: ~{cluster.total_lines()}")
        lines.append("━" * 70)
        lines.append("")

        # List regions
        lines.append("📍 Similar Regions:")
        for chunk in cluster.chunks:
            preview = chunk.preview(50)
            lines.append(f"   • {chunk.location}")
            if chunk.name:
                lines.append(f"     └─ {chunk.chunk_type}: {chunk.name}")
            lines.append(f"     └─ {preview}")
        lines.append("")

        # LLM Analysis (if available)
        if cluster.id in analyses:
            analysis = analyses[cluster.id]
            if analysis.worth_refactoring:
                lines.append("🤖 LLM Analysis:")
                lines.append(f"   📝 Commonality: {analysis.commonality}")
                lines.append(f"   💡 Suggestion: {analysis.abstraction}")
                lines.append(f"   🏷️  Suggested name: {analysis.suggested_name}")
                lines.append(f"   ⚡ Complexity: {analysis.complexity}")
            else:
                lines.append("⚠️ Not Recommended for Refactoring:")
                lines.append(f"   📝 What's Similar: {analysis.commonality}")
                lines.append(f"   ❌ Why Not: {analysis.abstraction}")
            lines.append("")

        # Show representative code
        rep = cluster.representative
        lines.append("📝 Representative Code:")
        lines.append(f"   {rep.location}")
        lines.append("")

        # Indent and truncate code
        code_lines = rep.content.split("\n")[:15]  # Max 15 lines
        for code_line in code_lines:
            lines.append(f"   │ {code_line}")
        if len(rep.content.split("\n")) > 15:
            lines.append("   │ ...")
        lines.append("")

    return "\n".join(lines)


def _format_markdown(
    clusters: List[Cluster],
    root_path: Path,
    threshold: float,
    analyses: Dict[int, "ClusterAnalysis"],
) -> str:
    """Markdown format for documentation."""
    lines = []

    # Header
    total_lines = sum(c.total_lines() for c in clusters)
    lines.append("# Code Similarity Report")
    lines.append("")
    lines.append(f"**Path:** `{root_path}`  ")
    lines.append(f"**Threshold:** {threshold:.0%}  ")
    lines.append(f"**Clusters Found:** {len(clusters)}  ")
    lines.append(f"**Total Duplicated Lines:** ~{total_lines}")
    if analyses:
        lines.append(f"**LLM Analyzed:** {len(analyses)} clusters")
    lines.append("")

    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("")
    for cluster in clusters:
        analysis = analyses.get(cluster.id)

        # Get a brief description for the TOC
        if analysis:
            label = analysis.suggested_name if analysis.suggested_name != "n/a" else "coincidental"
        elif cluster.representative.name:
            label = cluster.representative.name
        else:
            label = str(cluster.representative.file_path)

        # Show complexity badge for refactor-worthy, or "skip" for others
        badge = ""
        if analysis:
            if analysis.worth_refactoring:
                badge = f" `{analysis.complexity}`" if analysis.complexity != "n/a" else ""
            else:
                badge = " `⚠️ skip`"

        lines.append(f"- [{label}](#cluster-{cluster.id}) — {cluster.size} regions, ~{cluster.total_lines()} lines{badge}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for cluster in clusters:
        # Add anchor for TOC links
        lines.append(f"<a id=\"cluster-{cluster.id}\"></a>")
        lines.append("")
        lines.append(f"## Cluster {cluster.id}: {cluster.similarity_score:.0%} Similarity")
        lines.append("")
        lines.append(f"**{cluster.size} regions** across **{cluster.file_count} files** (~{cluster.total_lines()} lines)")
        lines.append("")

        # LLM Analysis (if available)
        if cluster.id in analyses:
            analysis = analyses[cluster.id]
            if analysis.worth_refactoring:
                lines.append("### 🤖 Analysis")
                lines.append("")
                lines.append(f"**Commonality:** {analysis.commonality}")
                lines.append("")
                lines.append(f"**Suggested Refactor:** {analysis.abstraction}")
                lines.append("")
                lines.append(f"- **Suggested Name:** `{analysis.suggested_name}`")
                lines.append(f"- **Complexity:** {analysis.complexity}")
            else:
                lines.append("### ⚠️ Not Recommended for Refactoring")
                lines.append("")
                lines.append(f"**What's Similar:** {analysis.commonality}")
                lines.append("")
                lines.append(f"**Why Not Refactor:** {analysis.abstraction}")
            lines.append("")

        # Table of regions
        lines.append("### Regions")
        lines.append("")
        lines.append("| File | Lines | Type | Name |")
        lines.append("|------|-------|------|------|")
        for chunk in cluster.chunks:
            name = chunk.name or "-"
            lines.append(f"| `{chunk.file_path}` | {chunk.start_line}-{chunk.end_line} | {chunk.chunk_type} | {name} |")
        lines.append("")

        # Code sample
        rep = cluster.representative
        lang = rep.language
        lines.append("### Representative Code")
        lines.append("")
        lines.append(f"From `{rep.location}`:")
        lines.append("")
        lines.append(f"```{lang}")

        # Truncate code for markdown
        code_lines = rep.content.split("\n")[:20]
        lines.append("\n".join(code_lines))
        if len(rep.content.split("\n")) > 20:
            lines.append("// ... (truncated)")

        lines.append("```")
        lines.append("")
        lines.append("[↑ Back to Table of Contents](#table-of-contents)")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _format_json(
    clusters: List[Cluster],
    root_path: Path,
    threshold: float,
    analyses: Dict[int, "ClusterAnalysis"],
) -> str:
    """JSON format for programmatic use."""
    data = {
        "meta": {
            "path": str(root_path),
            "threshold": threshold,
            "cluster_count": len(clusters),
            "total_duplicated_lines": sum(c.total_lines() for c in clusters),
            "llm_analyzed_count": len(analyses),
            "timestamp": datetime.now().isoformat(),
        },
        "clusters": [],
    }

    for cluster in clusters:
        cluster_data = {
            "id": cluster.id,
            "similarity": round(cluster.similarity_score, 4),
            "file_count": cluster.file_count,
            "region_count": cluster.size,
            "total_lines": cluster.total_lines(),
            "regions": [
                {
                    "file": str(chunk.file_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "name": chunk.name,
                    "preview": chunk.preview(80),
                }
                for chunk in cluster.chunks
            ],
            "representative": {
                "file": str(cluster.representative.file_path),
                "start_line": cluster.representative.start_line,
                "end_line": cluster.representative.end_line,
                "content": cluster.representative.content,
            },
        }

        # Add LLM analysis if available
        if cluster.id in analyses:
            analysis = analyses[cluster.id]
            cluster_data["analysis"] = {
                "commonality": analysis.commonality,
                "abstraction": analysis.abstraction,
                "suggested_name": analysis.suggested_name,
                "complexity": analysis.complexity,
            }

        data["clusters"].append(cluster_data)

    return json.dumps(data, indent=2)


def _format_html(
    clusters: List[Cluster],
    root_path: Path,
    threshold: float,
    analyses: Dict[int, "ClusterAnalysis"],
) -> str:
    """HTML format with embedded CSS/JS - fully offline, no external deps."""
    import html

    total_lines = sum(c.total_lines() for c in clusters)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Embedded CSS - dark/light mode, syntax highlighting, responsive
    css = """
:root {
    --bg: #ffffff; --bg-alt: #f5f5f5; --text: #1a1a1a; --text-muted: #666;
    --border: #e0e0e0; --accent: #0066cc; --accent-light: #e6f0ff;
    --code-bg: #f8f8f8; --code-border: #ddd;
    --severity-high: #dc3545; --severity-med: #fd7e14; --severity-low: #28a745;
}
[data-theme="dark"] {
    --bg: #1a1a2e; --bg-alt: #16213e; --text: #eaeaea; --text-muted: #aaa;
    --border: #333; --accent: #4da6ff; --accent-light: #1a3a5c;
    --code-bg: #0f0f1a; --code-border: #333;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); margin: 0; padding: 20px; line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; }
header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
h1 { margin: 0; font-size: 1.8em; }
.controls { display: flex; gap: 10px; align-items: center; }
button { background: var(--accent); color: white; border: none; padding: 8px 16px;
    border-radius: 4px; cursor: pointer; font-size: 14px; }
button:hover { opacity: 0.9; }
.theme-toggle { background: var(--bg-alt); color: var(--text); border: 1px solid var(--border); }
input[type="text"] { padding: 8px 12px; border: 1px solid var(--border); border-radius: 4px;
    background: var(--bg); color: var(--text); width: 200px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px; margin-bottom: 30px; }
.stat-card { background: var(--bg-alt); padding: 15px; border-radius: 8px; text-align: center; }
.stat-value { font-size: 2em; font-weight: bold; color: var(--accent); }
.stat-label { color: var(--text-muted); font-size: 0.9em; }
.toc { background: var(--bg-alt); padding: 20px; border-radius: 8px; margin-bottom: 30px; }
.toc h2 { margin-top: 0; }
.toc-list { list-style: none; padding: 0; margin: 0; max-height: 300px; overflow-y: auto; }
.toc-item { padding: 8px 0; border-bottom: 1px solid var(--border); display: flex;
    justify-content: space-between; align-items: center; }
.toc-item:last-child { border-bottom: none; }
.toc-link { color: var(--accent); text-decoration: none; }
.toc-link:hover { text-decoration: underline; }
.toc-meta { font-size: 0.85em; color: var(--text-muted); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75em;
    font-weight: bold; text-transform: uppercase; }
.badge-low { background: var(--severity-low); color: white; }
.badge-medium { background: var(--severity-med); color: white; }
.badge-high { background: var(--severity-high); color: white; }
details { background: var(--bg-alt); border-radius: 8px; margin-bottom: 20px;
    border: 1px solid var(--border); }
summary { padding: 15px 20px; cursor: pointer; font-weight: bold; font-size: 1.1em;
    display: flex; justify-content: space-between; align-items: center; }
summary::-webkit-details-marker { display: none; }
summary::before { content: '▶'; margin-right: 10px; transition: transform 0.2s; }
details[open] summary::before { transform: rotate(90deg); }
.cluster-content { padding: 0 20px 20px; }
.cluster-header { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 15px; }
.cluster-stat { background: var(--bg); padding: 8px 12px; border-radius: 4px; font-size: 0.9em; }
.analysis-box { background: var(--accent-light); padding: 15px; border-radius: 8px;
    margin-bottom: 15px; border-left: 4px solid var(--accent); }
.analysis-box h4 { margin: 0 0 10px; color: var(--accent); }
table { width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 0.9em; }
th, td { padding: 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--bg); font-weight: 600; }
.code-block { position: relative; margin-top: 15px; }
.code-header { display: flex; justify-content: space-between; align-items: center;
    background: var(--code-border); padding: 8px 12px; border-radius: 8px 8px 0 0; font-size: 0.85em; }
.copy-btn { background: transparent; color: var(--text); border: 1px solid var(--border);
    padding: 4px 8px; font-size: 0.8em; }
pre { margin: 0; padding: 15px; background: var(--code-bg); border-radius: 0 0 8px 8px;
    overflow-x: auto; font-family: 'SF Mono', Monaco, 'Courier New', monospace; font-size: 0.85em;
    border: 1px solid var(--code-border); border-top: none; }
code { font-family: inherit; }
/* Minimal syntax highlighting */
.kw { color: #d73a49; } /* keywords */
.str { color: #032f62; } /* strings */
.cm { color: #6a737d; font-style: italic; } /* comments */
.fn { color: #6f42c1; } /* functions */
.num { color: #005cc5; } /* numbers */
[data-theme="dark"] .kw { color: #ff7b72; }
[data-theme="dark"] .str { color: #a5d6ff; }
[data-theme="dark"] .cm { color: #8b949e; }
[data-theme="dark"] .fn { color: #d2a8ff; }
[data-theme="dark"] .num { color: #79c0ff; }
.back-to-top { text-align: center; padding: 10px; }
.back-to-top a { color: var(--accent); text-decoration: none; }
footer { text-align: center; padding: 30px; color: var(--text-muted); font-size: 0.9em; }
.hidden { display: none !important; }
@media (max-width: 600px) {
    .stats { grid-template-columns: 1fr 1fr; }
    .cluster-header { flex-direction: column; }
}
"""

    # Embedded JavaScript - theme toggle, copy, search, minimal syntax highlighting
    js = """
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme') || 'light';
    const next = current === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', next);
    localStorage.setItem('cse-theme', next);
    document.getElementById('theme-btn').textContent = next === 'dark' ? '☀️ Light' : '🌙 Dark';
}
function copyCode(btn) {
    const pre = btn.closest('.code-block').querySelector('pre');
    navigator.clipboard.writeText(pre.textContent).then(() => {
        const orig = btn.textContent;
        btn.textContent = '✓ Copied!';
        setTimeout(() => btn.textContent = orig, 1500);
    });
}
function filterClusters() {
    const query = document.getElementById('search').value.toLowerCase();
    document.querySelectorAll('details.cluster').forEach(el => {
        const text = el.textContent.toLowerCase();
        el.classList.toggle('hidden', query && !text.includes(query));
    });
    document.querySelectorAll('.toc-item').forEach(el => {
        const text = el.textContent.toLowerCase();
        el.classList.toggle('hidden', query && !text.includes(query));
    });
}
// Apply saved theme on load
document.addEventListener('DOMContentLoaded', () => {
    const saved = localStorage.getItem('cse-theme');
    if (saved) {
        document.documentElement.setAttribute('data-theme', saved);
        document.getElementById('theme-btn').textContent = saved === 'dark' ? '☀️ Light' : '🌙 Dark';
    }
});
// Minimal syntax highlighting
function highlight(code, lang) {
    // Keywords by language family
    const keywords = {
        'swift': /\\b(func|var|let|class|struct|enum|if|else|for|while|return|import|guard|self|nil|true|false|private|public|static|override|async|await|throws|try|catch)\\b/g,
        'python': /\\b(def|class|if|elif|else|for|while|return|import|from|as|try|except|finally|with|async|await|True|False|None|self|lambda|yield)\\b/g,
        'javascript': /\\b(function|const|let|var|class|if|else|for|while|return|import|export|from|async|await|try|catch|finally|this|new|true|false|null|undefined)\\b/g,
        'rust': /\\b(fn|let|mut|const|struct|enum|impl|if|else|for|while|loop|return|use|mod|pub|self|Self|true|false|async|await|match|where)\\b/g,
        'go': /\\b(func|var|const|type|struct|interface|if|else|for|range|return|import|package|go|defer|chan|select|true|false|nil)\\b/g,
        'c': /\\b(void|int|char|float|double|if|else|for|while|return|struct|enum|typedef|const|static|extern|sizeof|NULL|true|false)\\b/g,
        'cpp': /\\b(void|int|char|float|double|class|struct|enum|if|else|for|while|return|namespace|using|public|private|protected|virtual|override|const|static|new|delete|nullptr|true|false|auto|template)\\b/g,
    };
    const kw = keywords[lang] || keywords['c'];
    let h = code
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/(["'`])(?:(?!\\1)[^\\\\]|\\\\.)*\\1/g, '<span class="str">$&</span>')
        .replace(/\\/\\/.*$/gm, '<span class="cm">$&</span>')
        .replace(/\\/\\*[\\s\\S]*?\\*\\//g, '<span class="cm">$&</span>')
        .replace(/#.*$/gm, '<span class="cm">$&</span>')
        .replace(/\\b\\d+\\.?\\d*\\b/g, '<span class="num">$&</span>')
        .replace(kw, '<span class="kw">$&</span>');
    return h;
}
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('pre code').forEach(el => {
        const lang = el.className.replace('language-', '');
        el.innerHTML = highlight(el.textContent, lang);
    });
});
"""

    # Build HTML
    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Similarity Report - {html.escape(str(root_path))}</title>
    <style>{css}</style>
</head>
<body>
<div class="container">
    <header>
        <h1>🔍 Code Similarity Report</h1>
        <div class="controls">
            <input type="text" id="search" placeholder="Filter clusters..." oninput="filterClusters()">
            <button id="theme-btn" class="theme-toggle" onclick="toggleTheme()">🌙 Dark</button>
        </div>
    </header>

    <div class="stats">
        <div class="stat-card"><div class="stat-value">{len(clusters)}</div><div class="stat-label">Clusters</div></div>
        <div class="stat-card"><div class="stat-value">~{total_lines:,}</div><div class="stat-label">Duplicated Lines</div></div>
        <div class="stat-card"><div class="stat-value">{threshold:.0%}</div><div class="stat-label">Threshold</div></div>
        <div class="stat-card"><div class="stat-value">{len(analyses)}</div><div class="stat-label">LLM Analyzed</div></div>
    </div>

    <div class="toc">
        <h2>📑 Table of Contents</h2>
        <ul class="toc-list">
""")

    # TOC items
    for cluster in clusters:
        if cluster.id in analyses:
            label = html.escape(analyses[cluster.id].suggested_name)
            complexity = analyses[cluster.id].complexity or "medium"
        elif cluster.representative.name:
            label = html.escape(cluster.representative.name)
            complexity = "medium"
        else:
            label = html.escape(str(cluster.representative.file_path))
            complexity = "medium"

        badge_class = f"badge-{complexity}" if complexity in ("low", "medium", "high") else "badge-medium"

        parts.append(f"""            <li class="toc-item">
                <a class="toc-link" href="#cluster-{cluster.id}">{label}</a>
                <span class="toc-meta">{cluster.size} regions · ~{cluster.total_lines()} lines
                    <span class="badge {badge_class}">{complexity}</span>
                </span>
            </li>
""")

    parts.append("""        </ul>
    </div>
""")

    # Cluster details
    for cluster in clusters:
        rep = cluster.representative
        lang = rep.language or "text"

        if cluster.id in analyses:
            label = html.escape(analyses[cluster.id].suggested_name)
            complexity = analyses[cluster.id].complexity or "medium"
        elif rep.name:
            label = html.escape(rep.name)
            complexity = "medium"
        else:
            label = html.escape(str(rep.file_path))
            complexity = "medium"

        badge_class = f"badge-{complexity}" if complexity in ("low", "medium", "high") else "badge-medium"

        parts.append(f"""    <details class="cluster" id="cluster-{cluster.id}" open>
        <summary>
            <span>Cluster {cluster.id}: {label}</span>
            <span><span class="badge {badge_class}">{complexity}</span> {cluster.similarity_score:.0%} similar</span>
        </summary>
        <div class="cluster-content">
            <div class="cluster-header">
                <span class="cluster-stat">📁 {cluster.file_count} files</span>
                <span class="cluster-stat">📍 {cluster.size} regions</span>
                <span class="cluster-stat">📏 ~{cluster.total_lines()} lines</span>
            </div>
""")

        # Analysis box if available
        if cluster.id in analyses:
            a = analyses[cluster.id]
            parts.append(f"""            <div class="analysis-box">
                <h4>🤖 LLM Analysis</h4>
                <p><strong>What's similar:</strong> {html.escape(a.commonality)}</p>
                <p><strong>Recommendation:</strong> {html.escape(a.abstraction)}</p>
                <p><strong>Suggested name:</strong> <code>{html.escape(a.suggested_name)}</code></p>
            </div>
""")

        # Regions table
        parts.append("""            <table>
                <thead><tr><th>File</th><th>Lines</th><th>Type</th><th>Name</th></tr></thead>
                <tbody>
""")
        for chunk in cluster.chunks:
            name = html.escape(chunk.name) if chunk.name else "-"
            parts.append(f"""                    <tr>
                        <td><code>{html.escape(str(chunk.file_path))}</code></td>
                        <td>{chunk.start_line}-{chunk.end_line}</td>
                        <td>{html.escape(chunk.chunk_type)}</td>
                        <td>{name}</td>
                    </tr>
""")
        parts.append("""                </tbody>
            </table>
""")

        # Code block
        code_lines = rep.content.split("\n")[:25]
        code_content = "\n".join(code_lines)
        if len(rep.content.split("\n")) > 25:
            code_content += "\n// ... (truncated)"

        parts.append(f"""            <div class="code-block">
                <div class="code-header">
                    <span>📄 {html.escape(rep.location)}</span>
                    <button class="copy-btn" onclick="copyCode(this)">📋 Copy</button>
                </div>
                <pre><code class="language-{lang}">{html.escape(code_content)}</code></pre>
            </div>

            <div class="back-to-top"><a href="#top">↑ Back to top</a></div>
        </div>
    </details>
""")

    # Footer
    parts.append(f"""    <footer>
        Generated by <strong>code-similarity-engine</strong> on {timestamp}<br>
        <a href="https://github.com/jonathanlouis/code-similarity-engine">GitHub</a> ·
        No data sent anywhere · 100% offline
    </footer>
</div>
<script>{js}</script>
</body>
</html>""")

    return "".join(parts)
