"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from .task4_chunking_indexing import chunk_documents, load_documents
except ImportError:
    from task4_chunking_indexing import chunk_documents, load_documents


VECTORSTORE_PATH = Path(__file__).parent.parent / "data" / "vectorstore.pkl"
_bm25 = None
_corpus: list[dict] = []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _load_corpus() -> list[dict]:
    if VECTORSTORE_PATH.exists():
        try:
            with open(VECTORSTORE_PATH, "rb") as f:
                corpus = pickle.load(f)
            if isinstance(corpus, list) and corpus:
                return corpus
        except Exception as exc:
            print(f"[WARNING] Could not load vectorstore.pkl: {exc}")

    # Fallback for tests or first run: chunk markdown directly.
    docs = load_documents()
    return chunk_documents(docs) if docs else []


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def get_bm25():
    global _bm25, _corpus
    if _bm25 is None:
        _corpus = _load_corpus()
        if not _corpus:
            return None, []
        _bm25 = build_bm25_index(_corpus)
    return _bm25, _corpus


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    bm25, corpus = get_bm25()
    if bm25 is None or not corpus:
        return []

    scores = bm25.get_scores(_tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = corpus[int(idx)]
        results.append(
            {
                "content": chunk.get("content", ""),
                "score": float(scores[int(idx)]),
                "metadata": chunk.get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    results = lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
