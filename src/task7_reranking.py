"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import math
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY") or os.getenv("JINA_RERANKER_API_KEY")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def fallback_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Offline reranker combining normalized token overlap and original score."""
    if top_k <= 0 or not candidates:
        return []

    query_tokens = _tokens(query)
    query_len = max(len(query_tokens), 1)
    results = []

    for rank, item in enumerate(candidates):
        content_tokens = _tokens(item.get("content", ""))
        overlap_score = len(query_tokens & content_tokens) / query_len
        original_score = float(item.get("score", 0.0))
        rank_bonus = 1.0 / (rank + 1)
        score = (0.65 * overlap_score) + (0.30 * original_score) + (0.05 * rank_bonus)
        results.append({**item, "score": float(score)})

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if top_k <= 0 or not candidates:
        return []

    if not JINA_API_KEY:
        return fallback_rerank(query, candidates, top_k)

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {JINA_API_KEY}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c.get("content", "") for c in candidates],
                "top_n": top_k,
            },
            timeout=15,
        )
        response.raise_for_status()
        reranked = response.json().get("results", [])
        return [
            {**candidates[item["index"]], "score": float(item["relevance_score"])}
            for item in reranked
        ]
    except Exception as exc:
        print(f"[INFO] Jina rerank failed, using offline reranker: {exc}")
        return fallback_rerank(query, candidates, top_k)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if top_k <= 0 or not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx = remaining[0]
        best_score = float("-inf")

        for idx in remaining:
            candidate = candidates[idx]
            embedding = candidate.get("embedding")
            relevance = (
                _cosine(query_embedding, embedding)
                if embedding
                else float(candidate.get("score", 0.0))
            )

            diversity_penalty = 0.0
            for selected_idx in selected:
                selected_embedding = candidates[selected_idx].get("embedding")
                if embedding and selected_embedding:
                    diversity_penalty = max(
                        diversity_penalty,
                        _cosine(embedding, selected_embedding),
                    )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [{**candidates[idx], "score": float(candidates[idx].get("score", 0.0))} for idx in selected]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))
            if key not in items or item.get("score", 0.0) > items[key].get("score", 0.0):
                items[key] = item

    fused = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        fused.append({**items[key], "score": float(score)})
    return fused


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf(candidates if candidates and isinstance(candidates[0], list) else [candidates], top_k)
    if method == "mmr":
        # Use existing candidate scores if query embeddings are not supplied by the caller.
        return fallback_rerank(query, candidates, top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    demo = [
        {"content": "Dieu 248: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Python programming", "score": 0.6, "metadata": {}},
    ]
    for row in rerank("hinh phat tang tru ma tuy", demo, top_k=2):
        print(f"[{row['score']:.3f}] {row['content']}")
