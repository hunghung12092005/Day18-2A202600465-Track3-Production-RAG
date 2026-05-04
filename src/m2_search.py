"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass
import math
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        normalized = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
        return re.sub(r"\s+", " ", normalized).strip()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(chunk["text"]).split() for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.documents:
            return []
        tokenized_query = segment_vietnamese(query).split()
        if self.bm25 is not None:
            scores = list(self.bm25.get_scores(tokenized_query))
        else:
            query_set = set(tokenized_query)
            scores = []
            for tokens in self.corpus_tokens:
                token_set = set(tokens)
                overlap = len(query_set & token_set)
                scores.append(overlap / max(len(query_set), 1))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i]["metadata"],
                method="bm25",
            )
            for i in top_indices
            if scores[i] > 0 or len(self.documents) <= top_k
        ]


class DenseSearch:
    def __init__(self):
        self.client = None
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None
        self._encoder = None
        self._chunks: dict[str, list[dict]] = {}

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                self._encoder = False
        return self._encoder

    def _fallback_vector(self, text: str) -> dict[str, float]:
        tokens = segment_vietnamese(text).split()
        total = max(len(tokens), 1)
        counts: dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0 / total
        return counts

    def _fallback_similarity(self, query: str, text: str) -> float:
        qv = self._fallback_vector(query)
        dv = self._fallback_vector(text)
        dot = sum(qv.get(k, 0.0) * dv.get(k, 0.0) for k in set(qv) | set(dv))
        qn = math.sqrt(sum(v * v for v in qv.values()))
        dn = math.sqrt(sum(v * v for v in dv.values()))
        if qn == 0 or dn == 0:
            return 0.0
        return dot / (qn * dn)

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._chunks[collection] = chunks
        encoder = self._get_encoder()
        if not self.client or encoder is False or not chunks:
            return
        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            texts = [c["text"] for c in chunks]
            vectors = encoder.encode(texts, show_progress_bar=False)
            points = [
                PointStruct(id=i, vector=vector.tolist(), payload={**chunk["metadata"], "text": chunk["text"]})
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception:
            self.client = None

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        chunks = self._chunks.get(collection, [])
        if not chunks:
            return []
        encoder = self._get_encoder()
        if self.client and encoder is not False:
            try:
                query_vector = encoder.encode(query).tolist()
                hits = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=hit.payload["text"],
                        score=float(hit.score),
                        metadata={k: v for k, v in hit.payload.items() if k != "text"},
                        method="dense",
                    )
                    for hit in hits
                ]
            except Exception:
                self.client = None
        ranked = sorted(
            chunks,
            key=lambda chunk: self._fallback_similarity(query, chunk["text"]),
            reverse=True,
        )[:top_k]
        return [
            SearchResult(
                text=chunk["text"],
                score=self._fallback_similarity(query, chunk["text"]),
                metadata=chunk["metadata"],
                method="dense",
            )
            for chunk in ranked
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    fused: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = fused.setdefault(
                result.text,
                {"score": 0.0, "metadata": result.metadata, "result": result},
            )
            entry["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(fused.items(), key=lambda item: item[1]["score"], reverse=True)[:top_k]
    return [
        SearchResult(text=text, score=data["score"], metadata=data["metadata"], method="hybrid")
        for text, data in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
