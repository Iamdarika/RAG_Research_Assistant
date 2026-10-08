import os
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest

class HybridRerankRetriever:
    """
    Advanced Academic RAG Retriever:
    1. Dense semantic search via FastEmbed & ChromaDB.
    2. Sparse keyword search via BM25 (exact terminology, acronyms, math symbols).
    3. Reciprocal Rank Fusion (RRF) ensemble.
    4. Local Cross-Encoder Reranking via FlashRank for high-precision filtering.
    """
    def __init__(
        self,
        db_dir: str = "chroma_db",
        chunks_file: str = "chroma_db/processed_chunks.json",
        dense_k: int = 8,
        bm25_k: int = 8,
        final_k: int = 4,
        use_reranker: bool = True
    ):
        self.db_dir = db_dir
        self.chunks_file = chunks_file
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k
        self.use_reranker = use_reranker

        # 1. Initialize Dense Retriever
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.vectorstore = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.dense_k})

        # 2. Initialize Sparse BM25 Retriever
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._init_bm25()

        # 3. Initialize FlashRank Reranker
        if self.use_reranker:
            self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
        else:
            self.ranker = None

    def _init_bm25(self):
        """Loads processed chunks and builds the BM25 index."""
        if os.path.exists(self.chunks_file):
            try:
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                docs = [
                    Document(page_content=item["page_content"], metadata=item["metadata"])
                    for item in data
                ]
                if docs:
                    self.bm25_retriever = BM25Retriever.from_documents(docs)
                    self.bm25_retriever.k = self.bm25_k
            except Exception as e:
                print(f"Warning: Could not load BM25 corpus from '{self.chunks_file}': {e}")
                self.bm25_retriever = None

    def retrieve(self, query: str, final_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves, fuses, and reranks candidate passages for the given query.
        Returns a list of dicts with 'document', 'score', 'source', 'page'.
        """
        k = final_k or self.final_k

        # 1. Candidate Generation: Dense Retrieval
        dense_docs = self.dense_retriever.invoke(query)

        # 2. Candidate Generation: BM25 Sparse Retrieval
        bm25_docs = []
        if self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
            except Exception as e:
                print(f"BM25 retrieval error: {e}")

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF score = sum(1.0 / (60 + rank))
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            key = (doc.metadata.get("source", ""), doc.metadata.get("page_number", 0), doc.page_content[:100])
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60 + rank + 1))

        for rank, doc in enumerate(bm25_docs):
            key = (doc.metadata.get("source", ""), doc.metadata.get("page_number", 0), doc.page_content[:100])
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60 + rank + 1))

        # Sort candidates by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        candidate_docs = [doc_map[k] for k in sorted_keys[: max(k * 2, 8)]]

        if not candidate_docs:
            return []

        # 4. Cross-Encoder Reranking with FlashRank
        if self.use_reranker and self.ranker and candidate_docs:
            passages = [
                {
                    "id": idx,
                    "text": doc.page_content,
                    "meta": doc.metadata
                }
                for idx, doc in enumerate(candidate_docs)
            ]
            try:
                rerank_req = RerankRequest(query=query, passages=passages)
                reranked = self.ranker.rerank(rerank_req)
                
                results = []
                for item in reranked[:k]:
                    orig_doc = candidate_docs[item["id"]]
                    results.append({
                        "document": orig_doc,
                        "score": float(item.get("score", 0.0)),
                        "source": orig_doc.metadata.get("source", "Unknown"),
                        "page": orig_doc.metadata.get("page_number", orig_doc.metadata.get("page", 0) + 1),
                        "content": orig_doc.page_content
                    })
                return results
            except Exception as e:
                print(f"Reranking fallback to RRF order: {e}")

        # Fallback if reranker disabled or failed
        results = []
        for doc in candidate_docs[:k]:
            results.append({
                "document": doc,
                "score": 1.0,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1),
                "content": doc.page_content
            })
        return results
