"""
retrieval.py
------------
Implements Stage 1 (Retrieval, already given by CONFACT) and Stage 2 (Ranking)
of the RAG pipeline in Ge et al. (2025), Figure 2(a).

Documents are chunked into paragraphs, then ranked against the claim/question
using TF-IDF cosine similarity (a lightweight stand-in for the paper's
learned reranker). The top-K paragraphs are returned for answer generation.
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_paragraphs(text, max_words=120):
    """Paragraph-level chunking (paper finds this outperforms sentence-level,
    Section 5.2 'Impact of Chunking Strategies')."""
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk) > 40:  # skip near-empty tail chunks
            chunks.append(chunk)
    return chunks


def build_passages(claim):
    """Chunk every evidence doc for a claim into passages, tagging each
    passage with its source domain and evidence_id."""
    passages = []
    for ev in claim["evidence"]:
        for chunk in chunk_paragraphs(ev["content"]):
            passages.append({
                "text": chunk,
                "domain": ev["domain"],
                "url": ev["url"],
                "evidence_id": ev["evidence_id"],
            })
    return passages


def rank_passages(question, passages, top_k=6):
    """Rank passages by TF-IDF cosine similarity to the question and return
    the top-K (Stage 2: Ranking)."""
    if not passages:
        return []
    texts = [p["text"] for p in passages]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        matrix = vectorizer.fit_transform([question] + texts)
    except ValueError:
        # degenerate corpus (e.g. all stopwords) -> fall back to original order
        return passages[:top_k]
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    ranked = sorted(zip(sims, passages), key=lambda x: -x[0])
    return [p for _, p in ranked[:top_k]]


def get_relevant_passages(claim, top_k=6):
    passages = build_passages(claim)
    return rank_passages(claim["question"], passages, top_k=top_k)
