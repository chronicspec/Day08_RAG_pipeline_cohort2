"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import math
import pickle
import re
from pathlib import Path

try:
    from .task4_chunking_indexing import (
        EMBEDDING_MODEL,
        WEAVIATE_API_KEY,
        WEAVIATE_CLASS,
        WEAVIATE_URL,
    )
except ImportError:
    from task4_chunking_indexing import (
        EMBEDDING_MODEL,
        WEAVIATE_API_KEY,
        WEAVIATE_CLASS,
        WEAVIATE_URL,
    )


PROJECT_DIR = Path(__file__).parent.parent
LOCAL_VECTORSTORE = PROJECT_DIR / "data" / "vectorstore.pkl"
_embedding_model = None
_embedding_load_failed = False


def _get_embedding_model():
    """Load BAAI/bge-m3 lazily so importing this module stays lightweight."""
    global _embedding_load_failed, _embedding_model
    if _embedding_load_failed:
        return None
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except TypeError:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _embed_query(query: str) -> list[float] | None:
    try:
        model = _get_embedding_model()
        if model is None:
            return None
        return model.encode(query).tolist()
    except Exception as exc:
        global _embedding_load_failed
        _embedding_load_failed = True
        print(f"[WARNING] Could not embed query with {EMBEDDING_MODEL}: {exc}")
        return None


def _connect_weaviate():
    try:
        import weaviate
        from weaviate.classes.init import Auth

        if WEAVIATE_URL and "xxx" not in WEAVIATE_URL:
            auth = (
                Auth.api_key(WEAVIATE_API_KEY)
                if WEAVIATE_API_KEY and "xxx" not in WEAVIATE_API_KEY
                else None
            )
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=auth,
            )
        return weaviate.connect_to_local()
    except Exception as exc:
        print(f"[INFO] Weaviate is not available, using local vectorstore: {exc}")
        return None


def _search_weaviate(query_embedding: list[float], top_k: int) -> list[dict]:
    client = _connect_weaviate()
    if client is None:
        return []

    try:
        from weaviate.classes.query import MetadataQuery

        if not client.is_ready() or not client.collections.exists(WEAVIATE_CLASS):
            return []

        collection = client.collections.get(WEAVIATE_CLASS)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            distance = obj.metadata.distance
            score = 1.0 - float(distance) if distance is not None else 0.0
            props = obj.properties
            results.append(
                {
                    "content": props.get("content", ""),
                    "score": score,
                    "metadata": {
                        "source": props.get("source"),
                        "type": props.get("type"),
                        "chunk_index": props.get("chunk_index"),
                    },
                }
            )
        return sorted(results, key=lambda item: item["score"], reverse=True)
    except Exception as exc:
        print(f"[INFO] Weaviate search failed, using local vectorstore: {exc}")
        return []
    finally:
        client.close()


def _load_local_chunks() -> list[dict]:
    if not LOCAL_VECTORSTORE.exists():
        return []
    try:
        with open(LOCAL_VECTORSTORE, "rb") as f:
            chunks = pickle.load(f)
        return chunks if isinstance(chunks, list) else []
    except Exception as exc:
        print(f"[WARNING] Could not load local vectorstore: {exc}")
        return []


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _keyword_fallback_score(query: str, content: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    content_tokens = _tokenize(content)
    return len(query_tokens & content_tokens) / len(query_tokens)


def _search_local(query: str, query_embedding: list[float] | None, top_k: int) -> list[dict]:
    chunks = _load_local_chunks()
    if not chunks:
        return []

    results = []
    for chunk in chunks:
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        embedding = chunk.get("embedding")

        if query_embedding is not None and embedding:
            score = _cosine_similarity(query_embedding, embedding)
        else:
            score = _keyword_fallback_score(query, content)

        results.append(
            {
                "content": content,
                "score": float(score),
                "metadata": metadata,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    query_embedding = _embed_query(query)

    if query_embedding is not None:
        weaviate_results = _search_weaviate(query_embedding, top_k)
        if weaviate_results:
            return weaviate_results[:top_k]

    return _search_local(query, query_embedding, top_k)


if __name__ == "__main__":
    results = semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
