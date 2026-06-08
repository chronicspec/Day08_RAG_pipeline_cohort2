"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "").strip()
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "").strip()
WEAVIATE_CLASS = "RAGChunk"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Chọn 500 ký tự để giữ cho các đoạn văn bản đủ ngắn, tập trung vào một nội dung pháp lý hoặc một sự kiện báo chí cụ thể, tối ưu hóa cho embedding và LLM context window.
CHUNK_OVERLAP = 50      # Chọn 50 ký tự (10% chunk size) để bảo tồn ngữ cảnh liền mạch giữa các đoạn cắt, tránh mất thông tin ở các ranh giới chunk.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "BAAI/bge-m3"  # Model bge-m3 hỗ trợ đa ngôn ngữ xuất sắc, đặc biệt là tiếng Việt, và hỗ trợ tốt cho cả dense retrieval lẫn hybrid search.
EMBEDDING_DIM = 1024

# TODO: Chọn vector store
VECTOR_STORE = "weaviate"  # Chọn Weaviate để index vào vector store thực tế thay vì local pickle.


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"[WARNING] Thư mục {STANDARDIZED_DIR} không tồn tại!")
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        doc_type = "legal" if relative_path.startswith("legal/") else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": relative_path,
                "type": doc_type,
            }
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            cleaned_text = chunk_text.strip()
            if cleaned_text:
                chunks.append({
                    "content": cleaned_text,
                    "metadata": {**doc["metadata"], "chunk_index": i}
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType, VectorDistances
    from weaviate.classes.init import Auth

    # Always write to local pickle vectorstore as a fallback and for test compatibility.
    vectorstore_path = Path(__file__).parent.parent / "data" / "vectorstore.pkl"
    vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vectorstore_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[OK] Da luu {len(chunks)} chunks vao Vector Store cuc bo tai: {vectorstore_path}")

    # Check and connect to Weaviate
    client = None
    try:
        if WEAVIATE_URL and "xxx" not in WEAVIATE_URL:
            print(f"Connecting to Weaviate Cloud at {WEAVIATE_URL}...")
            auth = Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY and "xxx" not in WEAVIATE_API_KEY else None
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=auth
            )
        else:
            print("WEAVIATE_URL is not set or contains default placeholder. Trying local connection...")
            client = weaviate.connect_to_local()

        if client and client.is_ready():
            print("Successfully connected to Weaviate!")
            
            # Delete collection if it exists
            if client.collections.exists(WEAVIATE_CLASS):
                print(f"Collection {WEAVIATE_CLASS} exists. Deleting to index fresh data...")
                client.collections.delete(WEAVIATE_CLASS)

            # Create collection with properties & manual vector config (distance cosine)
            print(f"Creating collection {WEAVIATE_CLASS}...")
            collection = client.collections.create(
                name=WEAVIATE_CLASS,
                vectorizer_config=None,  # No internal vectorizer since we supply vectors manually
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                ),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="path", data_type=DataType.TEXT),
                    Property(name="type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ]
            )

            # Insert chunks into Weaviate using dynamic batching
            print(f"Indexing {len(chunks)} chunks into Weaviate...")
            with collection.batch.dynamic() as batch:
                for chunk in chunks:
                    properties = {
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "path": chunk["metadata"]["path"],
                        "type": chunk["metadata"]["type"],
                        "chunk_index": int(chunk["metadata"]["chunk_index"]),
                    }
                    batch.add_object(
                        properties=properties,
                        vector=chunk["embedding"]
                    )
            
            failed_objects = collection.batch.failed_objects
            if failed_objects:
                print(f"[WARNING] {len(failed_objects)} objects failed to index in Weaviate.")
                for fail in failed_objects[:3]:
                    print(f"  Error details: {fail.message}")
            else:
                print(f"[OK] Successfully indexed {len(chunks)} chunks into Weaviate collection '{WEAVIATE_CLASS}'.")
        else:
            print("[WARNING] Weaviate connection is not ready.")
    except Exception as e:
        print(f"[WARNING] Could not index to Weaviate: {e}")
        print("Falling back to local pickle vector store.")
    finally:
        if client:
            client.close()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
