"""
strategies.py
-------------
Implements the answering strategies evaluated in the project, mirroring
Section 4.2 (baselines) and 4.3 (source-aware) of Ge et al. (2025):

Baselines (no credibility info):
  - DirA : Direct Answer -- passages + question -> answer
  - CoT  : Chain-of-Thought -- reason first, then answer

Source-aware (use MBFC credibility as background, Section 4.3):
  - SF      : Source Filtering -- drop low-credibility passages before DirA
  - SBA_dir : Source Background Augmentation -- each passage annotated with
              its source's credibility background, then direct answer
  - SBA_CoT : same source-aware passages, with Chain-of-Thought reasoning

Every strategy ends with the model asked to output a final line
"Answer: Yes" or "Answer: No", which parse_answer() extracts.
"""
import re

from llm_client import call_llm
from credibility import filter_low_credibility


def _format_passages(passages, with_background=False):
    lines = []
    for i, p in enumerate(passages, 1):
        if with_background:
            bg = p["background"]
            lines.append(
                f"[Passage {i} | Source: {bg['media_name']} ({p['domain']}) | "
                f"MBFC credibility rating: {bg['credibility_label']}]\n{p['text']}"
            )
        else:
            lines.append(f"[Passage {i} | Source: {p['domain']}]\n{p['text']}")
    return "\n\n".join(lines)


def parse_answer(response_text):
    """Extract Yes/No from the model's response. Returns None if unparseable."""
    m = re.search(r"answer\s*:\s*(yes|no)", response_text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    # fallback: look for a standalone Yes/No near the end of the response
    tail = response_text.strip().splitlines()[-3:]
    for line in reversed(tail):
        m2 = re.search(r"\b(yes|no)\b", line, re.IGNORECASE)
        if m2:
            return m2.group(1).capitalize()
    return None


BASE_INSTRUCTIONS = (
    "You are a fact-checking assistant. You will be given a question and a set "
    "of retrieved passages from different web sources, which may CONFLICT with "
    "each other (some may be inaccurate or from unreliable sources). "
)

ANSWER_FORMAT = (
    "\n\nRespond with your reasoning, then end with exactly one line in the "
    "form:\nAnswer: Yes\nor\nAnswer: No"
)


def strategy_dira(question, passages):
    """Direct Answer baseline -- no credibility info, no explicit reasoning step."""
    ctx = _format_passages(passages, with_background=False)
    prompt = (
        f"{BASE_INSTRUCTIONS}\n\nQuestion: {question}\n\nRetrieved passages:\n{ctx}"
        f"\n\nBased on the passages above, answer the question directly."
        f"{ANSWER_FORMAT}"
    )
    return call_llm(prompt)


def strategy_cot(question, passages):
    """Chain-of-Thought baseline -- no credibility info, explicit reasoning step."""
    ctx = _format_passages(passages, with_background=False)
    prompt = (
        f"{BASE_INSTRUCTIONS}\n\nQuestion: {question}\n\nRetrieved passages:\n{ctx}"
        f"\n\nThink step by step: identify which passages agree or disagree, "
        f"note any conflicts, and reason about which evidence is more reliable "
        f"before answering.{ANSWER_FORMAT}"
    )
    return call_llm(prompt)


def strategy_sf(question, passages, threshold=0.35):
    """Source Filtering -- drop passages from low-credibility sources (by MBFC
    rating) before generation, then answer directly (no explicit background shown)."""
    filtered = filter_low_credibility(passages, threshold=threshold)
    if not filtered:  # don't discard all evidence if everything was filtered
        filtered = passages
    ctx = _format_passages(filtered, with_background=False)
    prompt = (
        f"{BASE_INSTRUCTIONS}\n\nQuestion: {question}\n\n"
        f"Retrieved passages (pre-filtered to only credible sources):\n{ctx}"
        f"\n\nBased on the passages above, answer the question directly."
        f"{ANSWER_FORMAT}"
    )
    return call_llm(prompt)


def strategy_sba_dir(question, passages):
    """Source Background Augmentation (direct) -- each passage is tagged with
    its source's MBFC credibility background; model must weigh this itself."""
    ctx = _format_passages(passages, with_background=True)
    prompt = (
        f"{BASE_INSTRUCTIONS}\n\nQuestion: {question}\n\n"
        f"Retrieved passages, each labeled with its source's MBFC credibility "
        f"rating:\n{ctx}"
        f"\n\nWeigh the passages according to their source credibility -- prefer "
        f"evidence from more credible sources when passages conflict -- then "
        f"answer the question directly.{ANSWER_FORMAT}"
    )
    return call_llm(prompt)


def strategy_sba_cot(question, passages):
    """Source Background Augmentation + Chain-of-Thought -- the paper's best
    performing configuration (Section 5.1, RQ3)."""
    ctx = _format_passages(passages, with_background=True)
    prompt = (
        f"{BASE_INSTRUCTIONS}\n\nQuestion: {question}\n\n"
        f"Retrieved passages, each labeled with its source's MBFC credibility "
        f"rating:\n{ctx}"
        f"\n\nThink step by step: (1) identify where passages conflict, "
        f"(2) explicitly discuss each source's credibility rating and how much "
        f"weight it deserves, (3) resolve the conflict by favoring credible "
        f"sources, then answer.{ANSWER_FORMAT}"
    )
    return call_llm(prompt)


STRATEGIES = {
    "DirA": strategy_dira,
    "CoT": strategy_cot,
    "SF": strategy_sf,
    "SBA_dir": strategy_sba_dir,
    "SBA_CoT": strategy_sba_cot,
}
