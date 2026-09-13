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
    "SBA_CoT": "Source Background + CoT",
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

    # Index (claim_id, strategy) -> first matching row in one pass, instead of
    # re-scanning the frame for every claim and again for every strategy.
    rows_by_key = {}
    for row in df.itertuples(index=False):
        rows_by_key.setdefault((int(row.claim_id), row.strategy), row)

    out_claims = []
    for c in claims:
        cid = c["id"]
        strat_results = {}
        for s in STRATEGY_ORDER:
            row = rows_by_key.get((cid, s))
            if row is None:
                continue
            strat_results[s] = {
                "prediction": None if pd.isna(row.prediction) else row.prediction,
                "correct": bool(row.correct),
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Source-Aware Fact-Checking — Demo</title>
<style>
  :root {
    --bg: #0f1117; --panel: #161923; --panel2: #1d212d; --panel3: #242938;
    --border: #2a2f3d; --border-strong: #39405280;
    --text: #e9ebf0; --text-2: #b9c0cf; --muted: #8b93a7;
    --accent: #6ea8fe;
    --cat-aware: #3987e5;   /* categorical: source-aware strategies */
    --cat-base: #c98500;    /* categorical: baseline strategies */
    --good: #3fb950; --bad: #f85149; --warn: #d29922;
    --good-bg: #12382255; --bad-bg: #3a141455; --warn-bg: #3a2e0a66;
    --radius: 10px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex; flex-direction: column;
    font-size: 14px; -webkit-font-smoothing: antialiased;
  }

  /* ---------- header ---------- */
  header {
    padding: 16px 28px; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #191d29 0%, var(--panel) 100%);
    display: flex; align-items: center; gap: 28px; flex-wrap: wrap;
  }
  .title-block { flex: 1 1 340px; min-width: 280px; }
  header h1 { margin: 0 0 3px; font-size: 18px; font-weight: 650; letter-spacing: -.01em; }
  header p { margin: 0; color: var(--muted); font-size: 12.5px; line-height: 1.45; }
  .stat-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .stat-tile {
    background: var(--panel2); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 9px 16px 8px; min-width: 108px;
  }
  .stat-tile .v { font-size: 21px; font-weight: 700; letter-spacing: -.02em; line-height: 1.15;
                  font-variant-numeric: tabular-nums; }
  .stat-tile .v .unit { font-size: 13px; font-weight: 600; color: var(--text-2); }
  .stat-tile .k { font-size: 10.5px; color: var(--muted); text-transform: uppercase;
                  letter-spacing: .06em; margin-top: 2px; white-space: nowrap; }
  .stat-tile.delta .v { color: var(--good); }

  .mock-banner { background: #4a2e0a; color: #ffd479; padding: 8px 28px; font-size: 13px; }

  /* ---------- layout ---------- */
  .layout { flex: 1; min-height: 0; display: grid; grid-template-columns: 330px 1fr; }
  #sidebar { border-right: 1px solid var(--border); overflow-y: auto; padding: 16px 14px 24px;
             background: #12141d; }
  #main { overflow-y: auto; padding: 26px 34px 48px; scroll-behavior: smooth; }
  @media (max-width: 860px) {
    .layout { grid-template-columns: 1fr; overflow-y: auto; }
    #sidebar, #main { overflow: visible; }
  }

  h3.side-label { font-size: 10.5px; text-transform: uppercase; letter-spacing: .08em;
                  color: var(--muted); margin: 20px 6px 9px; font-weight: 650; }
  h3.side-label:first-child { margin-top: 0; }

  /* ---------- summary bars ---------- */
  .legend { display: flex; gap: 14px; margin: 0 6px 9px; font-size: 11px; color: var(--text-2); }
  .legend .sw { display: inline-block; width: 9px; height: 9px; border-radius: 3px;
                margin-right: 5px; vertical-align: -1px; }
  .summary-bars { display: flex; flex-direction: column; gap: 9px; padding: 0 6px; }
  .bar-row { display: grid; grid-template-columns: 118px 1fr 42px; align-items: center; gap: 9px; }
  .bar-label { font-size: 11.5px; color: var(--text-2); white-space: nowrap; overflow: hidden;
               text-overflow: ellipsis; }
  .bar-label small { color: var(--muted); }
  .bar-track { background: var(--panel2); border-radius: 4px; height: 12px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 0 4px 4px 0; transition: width .5s cubic-bezier(.2,.7,.3,1); }
  .bar-fill.aware { background: var(--cat-aware); }
  .bar-fill.base { background: var(--cat-base); }
  .bar-val { font-size: 11.5px; text-align: right; color: var(--text);
             font-variant-numeric: tabular-nums; }
  .summary-note { font-size: 10.5px; color: var(--muted); margin: 9px 6px 0; line-height: 1.5; }

  /* ---------- claim list ---------- */
  .search-box { margin: 0 2px 8px; }
  .search-box input {
    width: 100%; background: var(--panel2); border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); font-size: 12.5px; padding: 7px 10px; outline: none;
  }
  .search-box input:focus { border-color: var(--accent); }
  .search-box input::placeholder { color: var(--muted); }
  .filter-row { display: flex; gap: 6px; margin: 0 2px 10px; flex-wrap: wrap; }
  .filter-btn { font-size: 11px; padding: 4px 9px; border-radius: 20px; border: 1px solid var(--border);
                background: transparent; color: var(--muted); cursor: pointer; transition: all .12s; }
  .filter-btn:hover { color: var(--text-2); border-color: var(--border-strong); }
  .filter-btn.active { border-color: var(--accent); color: var(--accent); background: #1a2740; }

  .claim-item {
    position: relative; padding: 9px 11px 9px 14px; border-radius: 8px; cursor: pointer;
    margin-bottom: 4px; border: 1px solid transparent; transition: background .1s;
  }
  .claim-item:hover { background: var(--panel2); }
  .claim-item.active { background: var(--panel2); border-color: var(--border-strong); }
  .claim-item.active::before {
    content: ""; position: absolute; left: 0; top: 8px; bottom: 8px; width: 3px;
    border-radius: 3px; background: var(--accent);
  }
  .ci-top { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
  .ci-num { font-size: 10px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .ci-dots { display: flex; gap: 3px; margin-left: auto; }
  .dot { width: 7px; height: 7px; border-radius: 50%; }
  .dot.ok { background: var(--good); }
  .dot.no { background: var(--bad); }
  .dot.na { background: transparent; border: 1px solid var(--muted); }
  .flip-tag { font-size: 9.5px; padding: 1px 7px; border-radius: 10px; font-weight: 600; }
  .flip-fixed { background: var(--good-bg); color: var(--good); }
  .flip-broke { background: var(--bad-bg); color: var(--bad); }
  .ci-text { font-size: 12.5px; line-height: 1.42; color: var(--text-2);
             display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .claim-item.active .ci-text { color: var(--text); }
  .empty-list { color: var(--muted); font-size: 12px; padding: 10px 6px; }

  /* ---------- main: claim ---------- */
  .section { margin-bottom: 30px; }
  .section > h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
                  color: var(--muted); margin: 0 0 12px; font-weight: 650; }
  .eyebrow { font-size: 11px; color: var(--accent); text-transform: uppercase;
             letter-spacing: .08em; font-weight: 650; margin-bottom: 6px; }
  .claim-title { font-size: 20px; font-weight: 650; letter-spacing: -.012em;
                 margin: 0 0 8px; line-height: 1.35; max-width: 820px; }
  .claim-q { color: var(--muted); font-size: 13px; margin: 0 0 12px; max-width: 820px; }
  .gold-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px;
                border-radius: 20px; font-size: 12px; font-weight: 650; }
  .gold-yes { background: var(--good-bg); color: var(--good); border: 1px solid #1d5c3455; }
  .gold-no { background: var(--bad-bg); color: var(--bad); border: 1px solid #6b232355; }

  /* ---------- evidence ---------- */
  .evidence-table { border: 1px solid var(--border); border-radius: var(--radius);
                    overflow: hidden; max-width: 820px; }
  .ev-row { display: grid; grid-template-columns: 40px 1fr auto; align-items: center;
            gap: 10px; padding: 9px 14px; background: var(--panel); font-size: 13px; }
  .ev-row + .ev-row { border-top: 1px solid var(--border); }
  .ev-row:hover { background: var(--panel2); }
  .ev-idx { color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
  .ev-domain { color: var(--text-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cred-badge { font-size: 10.5px; padding: 2px 9px; border-radius: 10px; font-weight: 650;
                white-space: nowrap; }
  .cred-high, .cred-mostly-factual { background: var(--good-bg); color: var(--good); }
  .cred-medium, .cred-mixed { background: var(--warn-bg); color: var(--warn); }
  .cred-low, .cred-very-low { background: var(--bad-bg); color: var(--bad); }
  .cred-unknown { background: #23263080; color: var(--muted); }

  /* ---------- strategy cards ---------- */
  .strategies-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
                     gap: 14px; }
  .strategy-card { background: var(--panel); border: 1px solid var(--border);
                   border-radius: var(--radius); padding: 14px 16px 12px; position: relative;
                   border-top: 3px solid var(--cat-base); }
  .strategy-card.source-aware { border-top-color: var(--cat-aware); }
  .strat-head { display: flex; justify-content: space-between; align-items: flex-start;
                gap: 10px; margin-bottom: 9px; }
  .strat-name { font-weight: 650; font-size: 13.5px; letter-spacing: -.01em; }
  .strat-kind { font-size: 10px; margin-top: 3px; color: var(--muted);
                text-transform: uppercase; letter-spacing: .06em; }
  .strategy-card.source-aware .strat-kind { color: var(--cat-aware); }
  .verdict { font-size: 11.5px; padding: 3px 11px; border-radius: 12px; font-weight: 700;
             white-space: nowrap; }
  .verdict-correct { background: var(--good-bg); color: var(--good); }
  .verdict-wrong { background: var(--bad-bg); color: var(--bad); }
  .verdict-pending { background: #23263080; color: var(--muted); }
  .pred-line { font-size: 12.5px; color: var(--muted); display: flex; gap: 14px; }
  .pred-line b { color: var(--text); font-weight: 650; }
  .reasoning-toggle { font-size: 12px; color: var(--accent); cursor: pointer; margin-top: 9px;
                      display: inline-flex; align-items: center; gap: 5px; user-select: none; }
  .reasoning-toggle .caret { transition: transform .15s; font-size: 10px; }
  .reasoning-toggle.open .caret { transform: rotate(180deg); }
  .reasoning-text { white-space: pre-wrap; font-size: 12px; color: var(--text-2); margin-top: 9px;
                    max-height: 280px; overflow-y: auto; display: none; line-height: 1.55;
                    border-top: 1px solid var(--border); padding-top: 9px; }
  .reasoning-text.open { display: block; }
  .card-pending { border-style: dashed; opacity: .65; }

  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-thumb { background: #2c3242; border-radius: 6px; border: 2px solid var(--bg); }
  ::-webkit-scrollbar-track { background: transparent; }
</style>
</head>
<body>
<header>
  <div class="title-block">
    <h1>Source-Aware Fact-Checking with Conflicting Evidence</h1>
    <p>Reproduction of CONFACT (Ge et&nbsp;al., IJCAI-25). Select a claim to inspect retrieved
       evidence, MBFC source credibility, and each strategy's answer &amp; reasoning.</p>
  </div>
  <div class="stat-row" id="statRow"></div>
</header>
<div id="mockBannerSlot"></div>
<div class="layout">
  <div id="sidebar">
    <h3 class="side-label">Accuracy by strategy</h3>
    <div id="summaryPanel"></div>
    <h3 class="side-label">Claims <span id="claimCount" style="color:var(--muted);font-weight:400"></span></h3>
    <div class="search-box"><input id="searchInput" type="search" placeholder="Search claims…" autocomplete="off"></div>
    <div class="filter-row">
      <button class="filter-btn active" data-filter="all">All</button>
      <button class="filter-btn" data-filter="flip-fixed">Fixed by source-aware</button>
      <button class="filter-btn" data-filter="flip-broke">Broken by source-aware</button>
    </div>
    <div id="claimList"></div>
  </div>
  <div id="main"></div>
</div>

<script>
const DATA = __DATA_JSON__;
const SHORT = { DirA: "Direct", CoT: "CoT", SF: "Src filter", SBA_dir: "SBA direct", SBA_CoT: "SBA + CoT" };

function credClass(label) {
  return "cred-" + (label || "unknown").toLowerCase().replace(/ /g, "-");
}
function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : s;
  return d.innerHTML;
}
function claimFlipType(claim) {
  const base = claim.strategies["DirA"];
  const aware = ["SBA_dir", "SBA_CoT", "SF"].map(s => claim.strategies[s]).filter(Boolean);
  if (!base || !aware.length) return null;
  const anyFixed = aware.some(r => r.correct) && !base.correct;
  const allBroke = base.correct && aware.every(r => !r.correct);
  if (anyFixed) return "flip-fixed";
  if (allBroke) return "flip-broke";
  return null;
}

/* ---------- header stat tiles ---------- */
function renderStats() {
  const el = document.getElementById("statRow");
  if (!DATA.summary || !DATA.summary.length) { el.innerHTML = ""; return; }
  const rows = DATA.summary;
  const best = rows.reduce((a, b) => (b.accuracy > a.accuracy ? b : a));
  const dira = rows.find(r => r.strategy === "DirA");
  const deltaPts = dira ? Math.round((best.accuracy - dira.accuracy) * 1000) / 10 : null;
  let html = `
    <div class="stat-tile">
      <div class="v">${Math.round(best.accuracy * 100)}<span class="unit">%</span></div>
      <div class="k">best · ${escapeHtml(SHORT[best.strategy] || best.strategy)}</div>
    </div>`;
  if (dira) html += `
    <div class="stat-tile">
      <div class="v">${Math.round(dira.accuracy * 100)}<span class="unit">%</span></div>
      <div class="k">baseline · Direct</div>
    </div>
    <div class="stat-tile delta">
      <div class="v">${deltaPts >= 0 ? "+" : ""}${deltaPts}<span class="unit">pt</span></div>
      <div class="k">source-aware gain</div>
    </div>`;
  html += `
    <div class="stat-tile">
      <div class="v">${DATA.claims.length}</div>
      <div class="k">claims · 5 strategies</div>
    </div>`;
  el.innerHTML = html;
}

/* ---------- summary bars ---------- */
function renderSummary() {
  const el = document.getElementById("summaryPanel");
  if (!DATA.summary || DATA.summary.length === 0) {
    el.innerHTML = '<p class="summary-note">Run src/evaluate.py to populate this.</p>';
    return;
  }
  const rows = DATA.strategy_order
    .map(s => DATA.summary.find(r => r.strategy === s)).filter(Boolean);
  const maxN = Math.max(...rows.map(r => r.n));
  let html = `
    <div class="legend">
      <span><span class="sw" style="background:var(--cat-aware)"></span>Source-aware</span>
      <span><span class="sw" style="background:var(--cat-base)"></span>Baseline</span>
    </div>
    <div class="summary-bars">`;
  rows.forEach(r => {
    const pct = Math.round(r.accuracy * 100);
    const cls = DATA.is_source_aware[r.strategy] ? "aware" : "base";
    const nNote = r.n < maxN ? ` <small>n=${r.n}</small>` : "";
    html += `<div class="bar-row" title="${escapeHtml(DATA.strategy_labels[r.strategy] || r.strategy)} — accuracy ${pct}% · macro-F1 ${r.macro_f1} · n=${r.n}">
      <span class="bar-label">${escapeHtml(SHORT[r.strategy] || r.strategy)}${nNote}</span>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
      <span class="bar-val">${pct}%</span>
    </div>`;
  });
  html += '</div><p class="summary-note">Accuracy; hover a row for macro-F1. Full metrics in results_summary.csv.</p>';
  el.innerHTML = html;
}

/* ---------- claim list ---------- */
let currentFilter = "all";
let currentClaimId = null;
let searchTerm = "";

function claimMatchesFilter(claim) {
  if (searchTerm && !claim.claim.toLowerCase().includes(searchTerm)) return false;
  if (currentFilter === "all") return true;
  return claimFlipType(claim) === currentFilter;
}
function visibleClaims() { return DATA.claims.filter(claimMatchesFilter); }

function renderClaimList() {
  const el = document.getElementById("claimList");
  const visible = visibleClaims();
  document.getElementById("claimCount").textContent = "(" + visible.length + "/" + DATA.claims.length + ")";
  if (!visible.length) {
    el.innerHTML = '<div class="empty-list">No claims match.</div>';
    return;
  }
  el.innerHTML = visible.map(c => {
    const flip = claimFlipType(c);
    const tag = flip === "flip-fixed" ? '<span class="flip-tag flip-fixed">fixed</span>'
              : flip === "flip-broke" ? '<span class="flip-tag flip-broke">broken</span>' : "";
    const dots = DATA.strategy_order.map(s => {
      const r = c.strategies[s];
      const cls = !r ? "na" : (r.correct ? "ok" : "no");
      const t = !r ? "pending" : (r.correct ? "correct" : "wrong");
      return `<span class="dot ${cls}" title="${escapeHtml(SHORT[s] || s)}: ${t}"></span>`;
    }).join("");
    return `<div class="claim-item ${c.id === currentClaimId ? "active" : ""}" data-id="${c.id}" tabindex="0">
      <div class="ci-top"><span class="ci-num">#${c.id}</span>${tag}<span class="ci-dots">${dots}</span></div>
      <div class="ci-text">${escapeHtml(c.claim)}</div>
    </div>`;
  }).join("");
  el.querySelectorAll(".claim-item").forEach(node => {
    node.addEventListener("click", () => selectClaim(parseInt(node.dataset.id)));
    node.addEventListener("keydown", e => { if (e.key === "Enter") selectClaim(parseInt(node.dataset.id)); });
  });
}

function selectClaim(id) {
  currentClaimId = id;
  renderClaimList();
  const claim = DATA.claims.find(c => c.id === id);
  renderMain(claim);
  document.getElementById("main").scrollTop = 0;
  const active = document.querySelector(".claim-item.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

/* ---------- main panel ---------- */
function renderMain(claim) {
  const el = document.getElementById("main");
  if (!claim) {
    el.innerHTML = '<p style="color:var(--muted)">Select a claim from the left.</p>';
    return;
  }
  const goldClass = claim.gold === "Yes" ? "gold-yes" : "gold-no";

  const evidenceHtml = claim.passages.length ? `
    <div class="evidence-table">${claim.passages.map((p, i) => `
      <div class="ev-row">
        <span class="ev-idx">${i + 1}</span>
        <span class="ev-domain">${escapeHtml(p.domain || "unknown")}</span>
        <span class="cred-badge ${credClass(p.credibility)}">${escapeHtml(p.credibility || "unknown")}</span>
      </div>`).join("")}
    </div>` : '<p style="color:var(--muted)">No passages recorded.</p>';

  const strategiesHtml = DATA.strategy_order.map(s => {
    const r = claim.strategies[s];
    const aware = DATA.is_source_aware[s];
    const cardClass = "strategy-card" + (aware ? " source-aware" : "");
    const kind = aware ? "source-aware" : "baseline";
    if (!r) {
      return `<div class="${cardClass} card-pending">
        <div class="strat-head">
          <div><div class="strat-name">${escapeHtml(DATA.strategy_labels[s] || s)}</div>
               <div class="strat-kind">${kind}</div></div>
          <span class="verdict verdict-pending">pending</span>
        </div>
        <div class="pred-line">Response not yet collected (rate-limit quota).</div>
      </div>`;
    }
    const verdictClass = r.correct ? "verdict-correct" : "verdict-wrong";
    const verdictText = r.correct ? "✓ correct" : "✗ wrong";
    const rid = "reason-" + s + "-" + claim.id;
    return `<div class="${cardClass}">
      <div class="strat-head">
        <div><div class="strat-name">${escapeHtml(DATA.strategy_labels[s] || s)}</div>
             <div class="strat-kind">${kind}</div></div>
        <span class="verdict ${verdictClass}">${verdictText}</span>
      </div>
      <div class="pred-line"><span>Predicted <b>${escapeHtml(r.prediction || "unparsed")}</b></span>
                             <span>Gold <b>${escapeHtml(claim.gold)}</b></span></div>
      <span class="reasoning-toggle" id="tgl-${rid}" onclick="toggleReasoning('${rid}')">
        <span class="caret">▼</span> show reasoning</span>
      <div class="reasoning-text" id="${rid}">${escapeHtml(r.raw)}</div>
    </div>`;
  }).join("");

  el.innerHTML = `
    <div class="section">
      <div class="eyebrow">Claim #${claim.id}</div>
      <p class="claim-title">${escapeHtml(claim.claim)}</p>
      <p class="claim-q">${escapeHtml(claim.question)}</p>
      <span class="gold-badge ${goldClass}">Gold answer · ${escapeHtml(claim.gold)}</span>
    </div>
    <div class="section">
      <h2>Retrieved &amp; ranked evidence — top ${claim.passages.length} passages, with MBFC credibility</h2>
      ${evidenceHtml}
    </div>
    <div class="section">
      <h2>Strategy comparison</h2>
      <div class="strategies-grid">${strategiesHtml}</div>
    </div>`;
}

function toggleReasoning(id) {
  const panel = document.getElementById(id);
  const tgl = document.getElementById("tgl-" + id);
  const open = panel.classList.toggle("open");
  tgl.classList.toggle("open", open);
  tgl.innerHTML = `<span class="caret">▼</span> ${open ? "hide" : "show"} reasoning`;
}

/* ---------- wiring ---------- */
document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    renderClaimList();
  });
});
document.getElementById("searchInput").addEventListener("input", e => {
  searchTerm = e.target.value.trim().toLowerCase();
  renderClaimList();
});
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || (e.key !== "ArrowDown" && e.key !== "ArrowUp")) return;
  const visible = visibleClaims();
  if (!visible.length) return;
  const idx = visible.findIndex(c => c.id === currentClaimId);
  const next = e.key === "ArrowDown" ? Math.min(idx + 1, visible.length - 1) : Math.max(idx - 1, 0);
  if (next !== idx) { selectClaim(visible[next].id); e.preventDefault(); }
});

if (DATA.is_mock) {
  document.getElementById("mockBannerSlot").innerHTML =
    '<div class="mock-banner">⚠ This data was generated in MOCK MODE (no GROQ_API_KEY was set during run_experiment.py) — predictions and reasoning below are placeholders, not real model output. Re-run with a real API key before presenting.</div>';
}

renderStats();
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
              "re-run run_experiment.py with a real GROQ_API_KEY, "
              "then re-run this script, before presenting.")


if __name__ == "__main__":
    main()
