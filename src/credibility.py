"""
credibility.py
--------------
Media Source Background Provider (mirrors the paper's "GT-MB": ground-truth
media background from MBFC, Section 4.3).

For every passage's source domain, look up its MBFC credibility rating and a
short bias/factuality summary. Convert the categorical rating into a
credibility score in (0, 1), analogous to the paper's s_cred (Eq. 1) -- but
here it's a direct mapping from MBFC's own label rather than a trained
predictor, since we use MBFC as ground truth (GT-MB setting).
"""
import json
import os

MBFC_LOOKUP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "mbfc_lookup.json",
)

# MBFC's own credibility vocabulary -> a numeric score, low = less trustworthy
CREDIBILITY_SCORE_MAP = {
    "high": 1.0,
    "mostly factual": 0.8,
    "mixed": 0.5,
    "low": 0.2,
    "very low": 0.0,
    "unknown": 0.5,  # neutral prior when MBFC has no rating
}

with open(MBFC_LOOKUP_PATH) as f:
    _MBFC = json.load(f)


def get_background(domain):
    """Return {domain, credibility_label, credibility_score, summary} for a domain."""
    entry = _MBFC.get(domain, {"media": domain, "credibility": "unknown", "summary": ""})
    label = (entry.get("credibility") or "unknown").lower()
    score = CREDIBILITY_SCORE_MAP.get(label, 0.5)
    return {
        "domain": domain,
        "media_name": entry.get("media", domain),
        "credibility_label": label,
        "credibility_score": score,
        "summary": entry.get("summary", ""),
    }


def annotate_passages(passages):
    """Attach a `background` dict to each passage."""
    for p in passages:
        p["background"] = get_background(p["domain"])
    return passages


def filter_low_credibility(passages, threshold=0.35):
    """Source Filtering (SF): drop passages from sources below the
    credibility threshold before ranking/generation (Section 4.3, Fig 2b)."""
    return [p for p in passages if p["background"]["credibility_score"] >= threshold]
