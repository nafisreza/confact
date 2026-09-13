"""
build_dataset.py
-----------------
Extracts a small, curated subset of the real CONFACT dataset (Ge et al., IJCAI-25)
for use in this project, along with the MBFC source-credibility ratings needed
for the source-aware strategies.

Source data: https://github.com/zoeyyes/CONFACT (HumC.pkl.gz, ModC.pkl.gz,
mbfc_media_data.pkl). Run once; outputs land in data/.
"""
import pickle
import gzip
import json
import os
import random
from urllib.parse import urlparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Where the original CONFACT release (HumC.pkl.gz, ModC.pkl.gz,
# mbfc_media_data.pkl) was unpacked -- not in this repo; override as needed.
RAW_DIR = os.environ.get(
    "CONFACT_RAW_DIR", os.path.join(_ROOT, "CONFACT-main", "data", "dataset")
)
OUT_DIR = os.path.join(_ROOT, "data")
N_CLAIMS = 30
MIN_CONTENT_LEN = 300
MIN_DOCS_WITH_CONTENT = 2
SEED = 42


def get_domain(url):
    try:
        d = urlparse(url).netloc.lower()
        return d.replace("www.", "")
    except Exception:
        return None


def load_raw():
    with gzip.open(f"{RAW_DIR}/HumC.pkl.gz", "rb") as f:
        humc = pickle.load(f)
    with gzip.open(f"{RAW_DIR}/ModC.pkl.gz", "rb") as f:
        modc = pickle.load(f)
    with open(f"{RAW_DIR}/mbfc_media_data.pkl", "rb") as f:
        mbfc = pickle.load(f)
    return humc, modc, mbfc


def usable(claim, mbfc):
    if claim.get("label") not in ("Supported", "Refuted"):
        return False
    evs = claim.get("evidence_url", [])
    good = [e for e in evs if e.get("content") and len(e["content"]) > MIN_CONTENT_LEN]
    if len(good) < MIN_DOCS_WITH_CONTENT:
        return False
    domains = [get_domain(e["original_link"]) for e in good]
    known = [d for d in domains if d in mbfc]
    return len(known) >= 1


def credibility_spread(claim, mbfc):
    evs = [e for e in claim["evidence_url"] if e.get("content") and len(e["content"]) > MIN_CONTENT_LEN]
    labs = set()
    for e in evs:
        d = get_domain(e["original_link"])
        if d in mbfc:
            labs.add(mbfc[d].get("credibility", "unknown"))
    return labs


def trim_claim(claim, mbfc):
    """Keep only fields we need, trim evidence to docs with real content,
    cap content length so the project stays lightweight."""
    evs = []
    for e in claim["evidence_url"]:
        if not e.get("content") or len(e["content"]) <= MIN_CONTENT_LEN:
            continue
        domain = get_domain(e["original_link"])
        evs.append({
            "evidence_id": e["evidence_id"],
            "url": e["original_link"],
            "domain": domain,
            "content": e["content"][:4000],  # cap length
            "mbfc_known": domain in mbfc,
        })
    return {
        "id": claim["id"],
        "claim": claim["claim"],
        "question": claim["question"],
        "label": claim["label"],  # Supported -> Yes, Refuted -> No
        "answer": "Yes" if claim["label"] == "Supported" else "No",
        "evidence": evs,
    }


def main():
    random.seed(SEED)
    humc, modc, mbfc = load_raw()

    pool = [c for c in (humc + modc) if usable(c, mbfc)]
    print(f"Usable claims in HumC+ModC: {len(pool)}")

    # Rank by credibility diversity (favor claims that actually mix high/low/mixed
    # credibility sources -- these are the interesting "conflicting evidence" cases)
    scored = [(len(credibility_spread(c, mbfc)), c) for c in pool]
    scored.sort(key=lambda x: -x[0])

    # Take a diverse, de-duplicated sample: mix of high-spread and some baseline cases,
    # balanced roughly across Supported/Refuted labels.
    top_candidates = [c for _, c in scored[:400]]
    random.shuffle(top_candidates)

    supported = [c for c in top_candidates if c["label"] == "Supported"]
    refuted = [c for c in top_candidates if c["label"] == "Refuted"]

    n_each = N_CLAIMS // 2
    selected = supported[:n_each] + refuted[:n_each]
    random.shuffle(selected)

    subset = [trim_claim(c, mbfc) for c in selected]
    for i, c in enumerate(subset):
        c["id"] = i + 1  # re-assign unique sequential IDs (source IDs collide across HumC/ModC)
    print(f"Selected {len(subset)} claims "
          f"({sum(1 for c in subset if c['label']=='Supported')} Supported / "
          f"{sum(1 for c in subset if c['label']=='Refuted')} Refuted)")

    # Build a small MBFC lookup containing only domains we actually reference
    needed_domains = {e["domain"] for c in subset for e in c["evidence"] if e["domain"]}
    mbfc_subset = {}
    for d in needed_domains:
        if d in mbfc:
            m = mbfc[d]
            mbfc_subset[d] = {
                "media": m.get("media"),
                "credibility": m.get("credibility", "unknown"),
                "summary": (m.get("details") or [""])[-1][:600],  # last "details" entry is usually the overall verdict
            }
        else:
            mbfc_subset[d] = {"media": d, "credibility": "unknown", "summary": ""}

    with open(f"{OUT_DIR}/confact_subset.json", "w") as f:
        json.dump(subset, f, indent=2)
    with open(f"{OUT_DIR}/mbfc_lookup.json", "w") as f:
        json.dump(mbfc_subset, f, indent=2)

    print(f"Wrote {OUT_DIR}/confact_subset.json and {OUT_DIR}/mbfc_lookup.json")


if __name__ == "__main__":
    main()
