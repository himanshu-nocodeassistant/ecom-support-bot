"""5i: Eval dashboard — served at GET /eval on the FastAPI app.

Reads backend/eval/results/*.json and renders a static HTML comparison table.
No frontend framework required — pure HTML/CSS generated server-side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent.parent / "eval" / "results"

MODES_ORDER = ["keyword", "fulltext", "hybrid", "hybrid+rerank", "hybrid+rerank+filter"]

MODE_LABELS = {
    "keyword": "Keyword (in-memory)",
    "fulltext": "Full-text (Postgres FTS)",
    "hybrid": "Hybrid (FTS + embeddings)",
    "hybrid+rerank": "Hybrid + Rerank",
    "hybrid+rerank+filter": "Hybrid + Rerank + Filter",
}


def _load_results() -> list[dict[str, Any]]:
    results = []
    for mode in MODES_ORDER:
        path = RESULTS_DIR / f"{mode.replace('+', '_')}.json"
        if path.exists():
            try:
                results.append(json.loads(path.read_text()))
            except Exception:
                pass
    return results


def _colour(val: float, all_vals: list[float], higher_is_better: bool = True) -> str:
    if not all_vals or len(set(all_vals)) == 1:
        return ""
    best = max(all_vals) if higher_is_better else min(all_vals)
    worst = min(all_vals) if higher_is_better else max(all_vals)
    if val == best:
        return "best"
    if val == worst:
        return "worst"
    return ""


def _bar(val: float, max_val: float = 1.0) -> str:
    pct = min(100, int((val / max_val) * 100)) if max_val else 0
    return f'<div class="bar-wrap"><div class="bar" style="width:{pct}%"></div></div>'


def render_dashboard() -> str:
    results = _load_results()

    if not results:
        return _wrap_html(
            "<div class='empty'>No eval results found. Run <code>python -m backend.eval.run --all-modes</code> first.</div>"
        )

    metrics = [
        ("avg_precision_at_3", "Precision@3", True),
        ("avg_recall_at_3", "Recall@3", True),
        ("avg_context_relevance", "Context Relevance", True),
        ("p50_latency_s", "P50 Latency (s)", False),
        ("p95_latency_s", "P95 Latency (s)", False),
    ]

    # Summary table
    all_vals: dict[str, list[float]] = {m[0]: [r.get(m[0], 0.0) for r in results] for m in metrics}

    header_cells = "".join(f"<th>{label}</th>" for _, label, _ in metrics)
    rows_html = ""
    for r in results:
        mode = r["mode"]
        cells = f"<td class='mode-name'>{MODE_LABELS.get(mode, mode)}</td>"
        for key, _, higher in metrics:
            val = r.get(key, 0.0)
            cls = _colour(val, all_vals[key], higher)
            display = f"{val:.4f}" if "latency" in key else f"{val:.3f}"
            cells += (
                f"<td class='metric {cls}'>{display}{_bar(val, max(all_vals[key]) or 1.0)}</td>"
            )
        cost = r.get("estimated_cost", {}).get("total_cost_usd", 0.0)
        cells += f"<td class='metric'>${cost:.6f}</td>"
        rows_html += f"<tr>{cells}</tr>"

    summary_table = f"""
    <section>
      <h2>Retrieval Mode Comparison</h2>
      <p class="meta">{results[0].get("n_queries", "?")} queries &nbsp;·&nbsp; {results[0].get("n_answerable", "?")} answerable</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Mode</th>{header_cells}<th>Est. Cost (USD)</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </section>
    """

    # Per-query drill-down for hybrid mode (best mode typically)
    drill_html = ""
    best_result = max(results, key=lambda r: r.get("avg_precision_at_3", 0.0))
    pq = best_result.get("per_query", [])
    if pq:
        pq_rows = ""
        for q in pq:
            p3 = q.get("precision_at_3", 0.0)
            r3 = q.get("recall_at_3", 0.0)
            cat = q.get("category", "")
            titles = ", ".join(t or "—" for t in q.get("retrieved_titles", []))
            expected = q.get("expected_title") or "—"
            hit_class = "hit" if r3 > 0 else ("na" if expected == "—" else "miss")
            pq_rows += f"""
              <tr class="{hit_class}">
                <td class="query-text">{_esc(q["query"])}</td>
                <td><span class="cat-badge cat-{cat}">{cat}</span></td>
                <td>{p3:.2f}</td>
                <td>{r3:.2f}</td>
                <td class="small">{_esc(expected)}</td>
                <td class="small">{_esc(titles)}</td>
              </tr>
            """
        drill_html = f"""
        <section>
          <h2>Per-query Drill-down — {MODE_LABELS.get(best_result["mode"], best_result["mode"])}</h2>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Query</th><th>Category</th><th>P@3</th><th>R@3</th>
                  <th>Expected Source</th><th>Retrieved Titles</th>
                </tr>
              </thead>
              <tbody>{pq_rows}</tbody>
            </table>
          </div>
        </section>
        """

    # Latency chart (simple CSS bar chart)
    lat_bars = ""
    max_lat = max((r.get("p95_latency_s", 0.0) for r in results), default=1.0) or 1.0
    for r in results:
        p50 = r.get("p50_latency_s", 0.0)
        p95 = r.get("p95_latency_s", 0.0)
        label = MODE_LABELS.get(r["mode"], r["mode"])
        w50 = int((p50 / max_lat) * 100)
        w95 = int((p95 / max_lat) * 100)
        lat_bars += f"""
          <div class="lat-row">
            <div class="lat-label">{_esc(label)}</div>
            <div class="lat-bars">
              <div class="lat-bar p50" style="width:{w50}%" title="P50: {p50:.4f}s">
                <span class="lat-val">P50 {p50:.3f}s</span>
              </div>
              <div class="lat-bar p95" style="width:{w95}%" title="P95: {p95:.4f}s">
                <span class="lat-val">P95 {p95:.3f}s</span>
              </div>
            </div>
          </div>
        """
    latency_section = f"""
    <section>
      <h2>Latency Distribution</h2>
      <div class="lat-chart">{lat_bars}</div>
    </section>
    """

    memory_section = _render_memory_section()
    body = summary_table + latency_section + drill_html + memory_section
    return _wrap_html(body)


def _render_memory_section() -> str:
    """Render memory recall panel from memory_eval results if available."""

    mem_path = RESULTS_DIR / "memory_eval.json"
    if not mem_path.exists():
        return """
    <section>
      <h2>Memory Recall (Phase 8)</h2>
      <div class="empty" style="padding:20px">
        No memory eval results found.
        Run: <code>python -m backend.eval.memory_eval</code>
      </div>
    </section>"""

    try:
        data = json.loads(mem_path.read_text())
    except Exception:
        return ""

    rate = data.get("memory_recall_rate", 0.0)
    recalled = data.get("recalled", 0)
    total = data.get("total", 0)
    threshold = 0.75

    rate_cls = "best" if rate >= threshold else "worst"
    bar_pct = int(rate * 100)

    return f"""
    <section>
      <h2>Memory Recall (Phase 8)</h2>
      <p class="meta">Measures fraction of memory fixtures where prior context appears in system prompt</p>
      <div style="display:flex;align-items:center;gap:24px;padding:16px 0">
        <div style="font-size:36px;font-weight:700" class="{rate_cls}">{rate:.0%}</div>
        <div>
          <div style="font-size:13px;color:var(--muted)">recall rate &nbsp;·&nbsp; {recalled}/{total} fixtures</div>
          <div style="margin-top:8px;background:var(--surface2);border-radius:6px;height:8px;width:240px">
            <div style="background:{"var(--green)" if rate >= threshold else "var(--red)"};height:8px;border-radius:6px;width:{bar_pct}%"></div>
          </div>
          <div style="font-size:11px;color:var(--muted);margin-top:4px">threshold: {threshold:.0%}</div>
        </div>
      </div>
    </section>"""


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _wrap_html(body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SupportBot — Eval Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
    --border: #2e3248; --accent: #6c63ff; --accent2: #4ecdc4;
    --text: #e8eaf0; --muted: #7b8099;
    --green: #27ae60; --red: #e74c3c; --orange: #e67e22;
  }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); padding: 32px 24px; min-height: 100vh; }}
  h1 {{ font-size: 22px; font-weight: 700; color: var(--accent); margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-bottom: 32px; }}
  section {{ margin-bottom: 40px; }}
  h2 {{ font-size: 16px; font-weight: 600; margin-bottom: 12px; color: var(--text); }}
  .meta {{ font-size: 12px; color: var(--muted); margin-bottom: 10px; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: var(--surface); color: var(--muted); font-weight: 600;
        padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border);
        white-space: nowrap; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  tr:hover td {{ background: var(--surface); }}
  .mode-name {{ font-weight: 500; white-space: nowrap; }}
  .metric {{ font-variant-numeric: tabular-nums; }}
  .best {{ color: var(--green); font-weight: 700; }}
  .worst {{ color: var(--red); }}
  .bar-wrap {{ width: 60px; display: inline-block; background: var(--surface2);
               border-radius: 3px; height: 4px; vertical-align: middle; margin-left: 6px; }}
  .bar {{ height: 4px; background: var(--accent); border-radius: 3px; }}
  /* Per-query */
  .hit td {{ border-left: 3px solid var(--green); }}
  .miss td {{ border-left: 3px solid var(--red); }}
  .na td {{ border-left: 3px solid var(--muted); }}
  .query-text {{ max-width: 280px; }}
  .small {{ font-size: 11px; color: var(--muted); max-width: 200px; }}
  .cat-badge {{ border-radius: 12px; font-size: 10px; font-weight: 600;
                padding: 2px 8px; letter-spacing: 0.3px; }}
  .cat-refund {{ background: rgba(231,76,60,.15); color: var(--red); }}
  .cat-shipping {{ background: rgba(230,126,34,.15); color: var(--orange); }}
  .cat-product {{ background: rgba(78,205,196,.15); color: var(--accent2); }}
  .cat-order {{ background: rgba(108,99,255,.15); color: var(--accent); }}
  .cat-multi-intent {{ background: rgba(39,174,96,.15); color: var(--green); }}
  .cat-off-topic, .cat-unanswerable {{ background: rgba(123,128,153,.15); color: var(--muted); }}
  /* Latency chart */
  .lat-chart {{ display: flex; flex-direction: column; gap: 12px; }}
  .lat-row {{ display: flex; align-items: center; gap: 12px; }}
  .lat-label {{ width: 220px; font-size: 12px; color: var(--muted); flex-shrink: 0; }}
  .lat-bars {{ flex: 1; display: flex; flex-direction: column; gap: 4px; }}
  .lat-bar {{ height: 18px; border-radius: 4px; display: flex; align-items: center;
              min-width: 4px; transition: width 0.3s; }}
  .lat-bar.p50 {{ background: var(--accent); opacity: 0.8; }}
  .lat-bar.p95 {{ background: var(--accent2); opacity: 0.6; }}
  .lat-val {{ font-size: 10px; color: #fff; padding: 0 6px; white-space: nowrap; }}
  code {{ background: var(--surface2); padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
  .empty {{ color: var(--muted); font-size: 14px; padding: 40px; text-align: center; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>⚡ SupportBot — Eval Dashboard</h1>
<p class="subtitle">Retrieval quality metrics across all modes &nbsp;·&nbsp;
  <a href="/">← Back to chat</a> &nbsp;·&nbsp;
  Run: <code>python -m backend.eval.run --all-modes</code></p>
{body}
</body>
</html>"""
