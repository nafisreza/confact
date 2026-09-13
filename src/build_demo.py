"""
build_demo.py
-------------
Builds results/demo.html: a single self-contained, offline HTML page for
live demos / presentations. It embeds the real data from this run --
claims, retrieved passages with MBFC credibility, and every strategy's
prediction + full reasoning -- so there's no server, no internet dependency,
and no risk of a live API call failing mid-presentation.

Run this AFTER run_experiment.py (needs results/results.csv and
results/raw_responses.jsonl to exist).

Usage:
    PYTHONPATH=src python src/build_demo.py
    # then open results/demo.html in a browser
"""
import json
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIMS_PATH = os.path.join(BASE, "data", "confact_subset.json")
RESULTS_CSV = os.path.join(BASE, "results", "results.csv")
RAW_JSONL = os.path.join(BASE, "results", "raw_responses.jsonl")
SUMMARY_CSV = os.path.join(BASE, "results", "results_summary.csv")
OUT_HTML = os.path.join(BASE, "results", "demo.html")

STRATEGY_ORDER = ["DirA", "CoT", "SF", "SBA_dir", "SBA_CoT"]
STRATEGY_LABELS = {
    "DirA": "Direct Answer (baseline)",
    "CoT": "Chain-of-Thought (baseline)",
    "SF": "Source Filtering",
    "SBA_dir": "Source Background + Direct",
    "SBA_CoT": "Source Background + CoT (paper's best config)",
}
IS_SOURCE_AWARE = {"DirA": False, "CoT": False, "SF": True, "SBA_dir": True, "SBA_CoT": True}


def load_raw_by_claim():
    raw_by_claim = {}
    passages_by_claim = {}
    with open(RAW_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            cid = rec["claim_id"]
            raw_by_claim.setdefault(cid, {})[rec["strategy"]] = rec["raw_response"]
            if cid not in passages_by_claim:
                passages_by_claim[cid] = rec["passages_used"]
    return raw_by_claim, passages_by_claim


def build_data():
    with open(CLAIMS_PATH) as f:
        claims = json.load(f)
    df = pd.read_csv(RESULTS_CSV)
    raw_by_claim, passages_by_claim = load_raw_by_claim()

    is_mock = any("[MOCK MODE" in txt for d in raw_by_claim.values() for txt in d.values())

    out_claims = []
    for c in claims:
        cid = c["id"]
        sub = df[df["claim_id"] == cid]
        strat_results = {}
        for s in STRATEGY_ORDER:
            row = sub[sub["strategy"] == s]
            if row.empty:
                continue
            row = row.iloc[0]
            strat_results[s] = {
                "prediction": None if pd.isna(row["prediction"]) else row["prediction"],
                "correct": bool(row["correct"]),
                "raw": raw_by_claim.get(cid, {}).get(s, ""),
            }
        out_claims.append({
            "id": cid,
            "claim": c["claim"],
            "question": c["question"],
            "gold": c["answer"],
            "passages": passages_by_claim.get(cid, []),
            "strategies": strat_results,
        })

    summary = []
    if os.path.exists(SUMMARY_CSV):
        sdf = pd.read_csv(SUMMARY_CSV)
        summary = sdf.to_dict("records")

    return {
        "is_mock": is_mock,
        "claims": out_claims,
        "summary": summary,
        "strategy_labels": STRATEGY_LABELS,
        "strategy_order": STRATEGY_ORDER,
        "is_source_aware": IS_SOURCE_AWARE,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Source-Aware Fact-Checking — Demo</title>
<style>
  :root {
    --bg: #0f1117; --panel: #171a24; --panel2: #1e212c; --border: #2a2e3a;
    --text: #e8e9ed; --muted: #9098a8; --accent: #6ea8fe; --good: #3fb950;
    --bad: #f85149; --warn: #d29922;
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); }
  header { padding: 18px 24px; border-bottom: 1px solid var(--border); background: var(--panel); }
  header h1 { margin: 0 0 4px 0; font-size: 20px; }
  header p { margin: 0; color: var(--muted); font-size: 13px; }
  .mock-banner { background: #4a2e0a; color: #ffd479; padding: 8px 24px; font-size: 13px; }
  .layout { display: grid; grid-template-columns: 300px 1fr; height: calc(100vh - 70px); }
  .mock-banner ~ .layout { height: calc(100vh - 104px); }
  #sidebar { border-right: 1px solid var(--border); overflow-y: auto; padding: 12px; }
  #sidebar h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
                margin: 4px 8px 8px; }
  .claim-item { padding: 10px 12px; border-radius: 8px; cursor: pointer; margin-bottom: 6px;
                border: 1px solid transparent; font-size: 13px; line-height: 1.4; }
  .claim-item:hover { background: var(--panel2); }
  .claim-item.active { background: var(--panel2); border-color: var(--accent); }
  .claim-item .flip-tag { display:inline-block; font-size: 10px; padding: 1px 6px; border-radius: 10px;
                           margin-bottom: 4px; }
  .flip-fixed { background: #123822; color: var(--good); }
  .flip-broke { background: #3a1414; color: var(--bad); }
  #main { overflow-y: auto; padding: 24px 32px; }
  .section { margin-bottom: 28px; }
  .section h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
                margin: 0 0 10px 0; }
  .claim-title { font-size: 19px; margin: 0 0 6px 0; }
  .gold-badge { display:inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px;
                font-weight: 600; }
  .gold-yes { background: #123822; color: var(--good); }
  .gold-no { background: #3a1414; color: var(--bad); }
  .passage { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
             padding: 12px 14px; margin-bottom: 10px; font-size: 13px; line-height: 1.5; }
  .passage-head { display:flex; justify-content: space-between; align-items:center; margin-bottom: 6px; }
  .domain { color: var(--muted); font-size: 12px; }
  .cred-badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
  .cred-high, .cred-mostly-factual { background: #123822; color: var(--good); }
  .cred-mixed { background: #3a2e0a; color: var(--warn); }
  .cred-low, .cred-very-low { background: #3a1414; color: var(--bad); }
  .cred-unknown { background: #23263080; color: var(--muted); }
  .strategies-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 900px) { .strategies-grid { grid-template-columns: 1fr; } }
  .strategy-card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
                   padding: 14px 16px; }
  .strategy-card.source-aware { border-color: #35507a; }
  .strat-head { display:flex; justify-content: space-between; align-items:flex-start; margin-bottom: 8px; }
  .strat-name { font-weight: 600; font-size: 13px; }
  .strat-tag { font-size: 10px; color: var(--accent); border: 1px solid #35507a; padding: 1px 6px;
               border-radius: 8px; margin-top: 3px; display:inline-block; }
  .verdict { font-size: 12px; padding: 3px 10px; border-radius: 10px; font-weight: 700; white-space: nowrap; }
  .verdict-correct { background: #123822; color: var(--good); }
  .verdict-wrong { background: #3a1414; color: var(--bad); }
  .reasoning-toggle { font-size: 12px; color: var(--accent); cursor: pointer; margin-top: 6px;
                       display:inline-block; user-select: none; }
  .reasoning-text { white-space: pre-wrap; font-size: 12px; color: var(--muted); margin-top: 8px;
                     max-height: 260px; overflow-y: auto; display: none; line-height: 1.5;
                     border-top: 1px solid var(--border); padding-top: 8px; }
  .reasoning-text.open { display: block; }
  .summary-bars { display: flex; flex-direction: column; gap: 10px; }
  .bar-row { display: grid; grid-template-columns: 190px 1fr 50px; align-items: center; gap: 10px; }
  .bar-label { font-size: 12px; }
  .bar-track { background: var(--panel2); border-radius: 6px; height: 16px; overflow: hidden; }
  .bar-fill { background: var(--accent); height: 100%; }
  .bar-val { font-size: 12px; text-align: right; color: var(--muted); }
  .filter-row { display:flex; gap:6px; margin-bottom: 10px; flex-wrap: wrap; }
  .filter-btn { font-size: 11px; padding: 4px 8px; border-radius: 8px; border: 1px solid var(--border);
                background: var(--panel2); color: var(--muted); cursor: pointer; }
  .filter-btn.active { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Source-Aware Fact-Checking with Conflicting Evidence</h1>
  <p>Based on CONFACT (Ge et al., IJCAI-25) — click a claim on the left to see retrieved evidence, source credibility, and how each strategy answered.</p>
</header>
<div id="mockBannerSlot"></div>
<div class="layout">
  <div id="sidebar">
    <h3>Overall results</h3>
    <div id="summaryPanel" class="section"></div>
    <h3>Claims (<span id="claimCount"></span>)</h3>
    <div class="filter-row">
      <button class="filter-btn active" data-filter="all">All</button>
      <button class="filter-btn" data-filter="flip-fixed">Source-aware fixed it</button>
      <button class="filter-btn" data-filter="flip-broke">Source-aware broke it</button>
    </div>
    <div id="claimList"></div>
  </div>
  <div id="main"></div>
</div>

<script>
const DATA = __DATA_JSON__;

function credClass(label) {
  return "cred-" + (label || "unknown").toLowerCase().replace(/ /g, "-");
}

function claimFlipType(claim) {
  const base = claim.strategies["DirA"];
  const best = claim.strategies["SBA_CoT"];
  if (!base || !best) return null;
  if (!base.correct && best.correct) return "flip-fixed";
  if (base.correct && !best.correct) return "flip-broke";
  return null;
}

function renderSummary() {
  const el = document.getElementById("summaryPanel");
  if (!DATA.summary || DATA.summary.length === 0) {
    el.innerHTML = '<p style="color:var(--muted);font-size:12px;">Run src/evaluate.py to populate this.</p>';
    return;
  }
  const order = DATA.strategy_order;
  const rows = order.map(s => DATA.summary.find(r => r.strategy === s)).filter(Boolean);
  let html = '<div class="summary-bars">';
  rows.forEach(r => {
    const pct = Math.round(r.accuracy * 100);
    html += `<div class="bar-row">
      <span class="bar-label">${DATA.strategy_labels[r.strategy] || r.strategy}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <span class="bar-val">${pct}%</span>
    </div>`;
  });
  html += '</div><p style="font-size:11px;color:var(--muted);margin-top:8px;">Accuracy shown; see results_summary.csv for Macro-F1.</p>';
  el.innerHTML = html;
}

let currentFilter = "all";
let currentClaimId = null;

function claimMatchesFilter(claim, filter) {
  if (filter === "all") return true;
  return claimFlipType(claim) === filter;
}

function renderClaimList() {
  const el = document.getElementById("claimList");
  const visible = DATA.claims.filter(c => claimMatchesFilter(c, currentFilter));
  document.getElementById("claimCount").textContent = visible.length + "/" + DATA.claims.length;
  el.innerHTML = visible.map(c => {
    const flip = claimFlipType(c);
    const tag = flip === "flip-fixed" ? '<span class="flip-tag flip-fixed">fixed by source-aware</span><br/>'
              : flip === "flip-broke" ? '<span class="flip-tag flip-broke">broken by source-aware</span><br/>'
              : "";
    return `<div class="claim-item ${c.id === currentClaimId ? 'active' : ''}" data-id="${c.id}">
      ${tag}${escapeHtml(c.claim)}
    </div>`;
  }).join("");
  el.querySelectorAll(".claim-item").forEach(node => {
    node.addEventListener("click", () => selectClaim(parseInt(node.dataset.id)));
  });
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function selectClaim(id) {
  currentClaimId = id;
  renderClaimList();
  const claim = DATA.claims.find(c => c.id === id);
  renderMain(claim);
}

function renderMain(claim) {
  const el = document.getElementById("main");
  if (!claim) {
    el.innerHTML = '<p style="color:var(--muted)">Select a claim from the left.</p>';
    return;
  }
  const goldClass = claim.gold === "Yes" ? "gold-yes" : "gold-no";

  let passagesHtml = claim.passages.map(p => `
    <div class="passage">
      <div class="passage-head">
        <span class="domain">${escapeHtml(p.domain || "unknown")}</span>
        <span class="cred-badge ${credClass(p.credibility)}">${escapeHtml(p.credibility || "unknown")}</span>
      </div>
    </div>`).join("");

  let strategiesHtml = DATA.strategy_order.map(s => {
    const r = claim.strategies[s];
    if (!r) return "";
    const verdictClass = r.correct ? "verdict-correct" : "verdict-wrong";
    const verdictText = r.correct ? "✓ correct" : "✗ wrong";
    const cardClass = DATA.is_source_aware[s] ? "strategy-card source-aware" : "strategy-card";
    const tag = DATA.is_source_aware[s] ? '<span class="strat-tag">source-aware</span>' : '';
    const rid = "reason-" + s + "-" + claim.id;
    return `<div class="${cardClass}">
      <div class="strat-head">
        <div>
          <div class="strat-name">${DATA.strategy_labels[s] || s}</div>
          ${tag}
        </div>
        <span class="verdict ${verdictClass}">${verdictText}</span>
      </div>
      <div style="font-size:12px;color:var(--muted);">Predicted: <b style="color:var(--text)">${r.prediction || "unparsed"}</b> · Gold: <b style="color:var(--text)">${claim.gold}</b></div>
      <span class="reasoning-toggle" onclick="toggleReasoning('${rid}')">show reasoning ▾</span>
      <div class="reasoning-text" id="${rid}">${escapeHtml(r.raw)}</div>
    </div>`;
  }).join("");

  el.innerHTML = `
    <div class="section">
      <h2>Claim</h2>
      <p class="claim-title">${escapeHtml(claim.claim)}</p>
      <p style="color:var(--muted);font-size:13px;">${escapeHtml(claim.question)}</p>
      <span class="gold-badge ${goldClass}">Gold answer: ${claim.gold}</span>
    </div>
    <div class="section">
      <h2>Retrieved &amp; ranked evidence (top-${claim.passages.length})</h2>
      ${passagesHtml || '<p style="color:var(--muted)">No passages recorded.</p>'}
    </div>
    <div class="section">
      <h2>Strategy comparison</h2>
      <div class="strategies-grid">${strategiesHtml}</div>
    </div>
  `;
}

function toggleReasoning(id) {
  document.getElementById(id).classList.toggle("open");
}

document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    renderClaimList();
  });
});

if (DATA.is_mock) {
  document.getElementById("mockBannerSlot").innerHTML =
    '<div class="mock-banner">⚠ This data was generated in MOCK MODE (no ANTHROPIC_API_KEY was set during run_experiment.py) — predictions and reasoning below are placeholders, not real model output. Re-run with a real API key before presenting.</div>';
}

renderSummary();
renderClaimList();
if (DATA.claims.length) selectClaim(DATA.claims[0].id);
</script>
</body>
</html>
"""


def main():
    data = build_data()
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"Wrote {OUT_HTML}  ({len(html)/1024:.0f} KB)")
    if data["is_mock"]:
        print("NOTE: current results are MOCK MODE placeholders -- "
              "re-run run_experiment.py with a real ANTHROPIC_API_KEY, "
              "then re-run this script, before presenting.")


if __name__ == "__main__":
    main()
