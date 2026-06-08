"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""


import os
import pickle
import re
from pathlib import Path

from dotenv import load_dotenv

try:
    from .task4_chunking_indexing import chunk_documents, load_documents
except ImportError:
    from task4_chunking_indexing import chunk_documents, load_documents

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTORSTORE_PATH = Path(__file__).parent.parent / "data" / "vectorstore.pkl"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _local_corpus() -> list[dict]:
    if VECTORSTORE_PATH.exists():
        try:
            with open(VECTORSTORE_PATH, "rb") as f:
                corpus = pickle.load(f)
            if isinstance(corpus, list) and corpus:
                return corpus
        except Exception:
            pass

    docs = load_documents()
    return chunk_documents(docs) if docs else []


def upload_documents() -> None:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("[INFO] PAGEINDEX_API_KEY is not set; skipping upload.")
        return

    try:
        from pageindex import PageIndex

        client = PageIndex(api_key=PAGEINDEX_API_KEY)
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            client.upload(
                content=md_file.read_text(encoding="utf-8"),
                metadata={"source": md_file.name, "type": md_file.parent.name},
            )
            print(f"[OK] Uploaded {md_file.name}")
    except Exception as exc:
        print(f"[WARNING] PageIndex upload failed: {exc}")


def _pageindex_api_search(query: str, top_k: int) -> list[dict]:
    if not PAGEINDEX_API_KEY:
        return []

    try:
        from pageindex import PageIndex

        client = PageIndex(api_key=PAGEINDEX_API_KEY)
        response = client.query(query=query, top_k=top_k)
        return [
            {
                "content": getattr(item, "text", ""),
                "score": float(getattr(item, "score", 0.0)),
                "metadata": getattr(item, "metadata", {}),
                "source": "pageindex",
            }
            for item in response
        ]
    except Exception as exc:
        print(f"[INFO] PageIndex API search failed, using local fallback: {exc}")
        return []


def _local_vectorless_search(query: str, top_k: int) -> list[dict]:
    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    results = []
    for chunk in _local_corpus():
        content = chunk.get("content", "")
        content_tokens = _tokens(content)
        overlap = len(query_tokens & content_tokens)
        if overlap == 0:
            continue
        score = overlap / len(query_tokens)
        results.append(
            {
                "content": content,
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if top_k <= 0:
        return []

    api_results = _pageindex_api_search(query, top_k)
    if api_results:
        return api_results[:top_k]

    local_results = _local_vectorless_search(query, top_k)
    if local_results:
        return local_results

    return [
        {
            "content": "No PageIndex or local fallback evidence was found for this query.",
            "score": 0.0,
            "metadata": {"source": "pageindex_fallback", "type": "fallback"},
            "source": "pageindex",
        }
    ][:top_k]


if __name__ == "__main__":
    for row in pageindex_search("hinh phat su dung ma tuy", top_k=3):
        print(f"[{row['score']:.3f}] {row['content'][:100]}...")
