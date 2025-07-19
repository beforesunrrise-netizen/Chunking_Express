"""
벡터 기반 검색 구현 (수정 완료)
Vector-based retriever implementation (Corrected)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from loguru import logger
import faiss

from src.data_structures import Query, Chunk
from src.embedders.base_embedder import BaseEmbedder
from .base_retriever import BaseRetriever


class VectorRetriever(BaseRetriever):
    """벡터 기반 검색기"""

    def __init__(self, embedder: BaseEmbedder, use_faiss: bool = True):
        self.embedder = embedder
        self.use_faiss = use_faiss
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunk_embeddings: Dict[str, np.ndarray] = {}
        self.chunk_map: Dict[int, str] = {}  # 인덱스 -> 청크 ID 매핑

    # <<< 수정된 retrieve 함수 >>>
    async def retrieve(self, query: Query, chunks: List[Chunk], k: int = 5) -> List[Chunk]:
        """쿼리에 대해 관련 청크 검색"""
        if not chunks:
            logger.warning("검색할 청크가 없어 빈 리스트를 반환합니다.")
            return []

        # 쿼리 임베딩
        query_embedding = await self.embedder.embed_text(query.question)

        # 항상 현재 청크들로 인덱스를 새로 구축
        await self.build_index(chunks)

        # 검색 수행
        if self.use_faiss and self.index is not None:
            retrieved_chunks = await self._faiss_search(query_embedding, chunks, k)
        else:
            retrieved_chunks = await self._brute_force_search(query_embedding, chunks, k)

        logger.info(f"검색 완료: {len(retrieved_chunks)}개 청크 반환")
        return retrieved_chunks

    async def build_index(self, chunks: List[Chunk]) -> None:
        """FAISS 인덱스 구축"""
        if not chunks:
            logger.warning("인덱스 구축할 청크가 없습니다.")
            return

        logger.info(f"{len(chunks)}개 청크로 인덱스 구축 시작")

        # 청크 임베딩 생성
        embeddings = await self.embedder.embed_chunks(chunks)
        if not embeddings:
            logger.warning("임베딩 생성에 실패하여 인덱스를 구축할 수 없습니다.")
            self.index = None
            return

        # 임베딩 저장
        self.chunk_embeddings.clear()
        self.chunk_map.clear()

        valid_embeddings = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is not None:
                self.chunk_embeddings[chunk.id] = embedding
                self.chunk_map[len(valid_embeddings)] = chunk.id
                valid_embeddings.append(embedding)

        if not valid_embeddings:
            logger.warning("유효한 임베딩이 없어 인덱스를 구축할 수 없습니다.")
            self.index = None
            return

        if self.use_faiss:
            # FAISS 인덱스 생성
            dimension = valid_embeddings[0].shape[0]
            self.index = faiss.IndexFlatL2(dimension)

            # 임베딩 추가
            embeddings_matrix = np.array(valid_embeddings).astype('float32')
            self.index.add(embeddings_matrix)

            logger.info(f"FAISS 인덱스 구축 완료: {self.index.ntotal}개 벡터")

    async def _faiss_search(
            self,
            query_embedding: np.ndarray,
            chunks: List[Chunk],
            k: int
    ) -> List[Chunk]:
        """FAISS를 사용한 벡터 검색"""
        if self.index is None or self.index.ntotal == 0:
            return []

        # 검색할 개수 조정
        k = min(k, self.index.ntotal)

        # FAISS 검색
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, k)

        # 결과 청크 수집
        retrieved_chunks = []
        chunk_dict = {chunk.id: chunk for chunk in chunks}

        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:
                continue

            chunk_id = self.chunk_map.get(idx)
            if chunk_id and chunk_id in chunk_dict:
                chunk = chunk_dict[chunk_id]
                similarity = 1 / (1 + dist)
                chunk.metadata["retrieval_score"] = float(similarity)
                chunk.metadata["retrieval_rank"] = i + 1
                retrieved_chunks.append(chunk)

        return retrieved_chunks

    async def _brute_force_search(
            self,
            query_embedding: np.ndarray,
            chunks: List[Chunk],
            k: int
    ) -> List[Chunk]:
        """전수 검색 (Brute Force)"""
        for chunk in chunks:
            if chunk.id not in self.chunk_embeddings:
                embedding = await self.embedder.embed_text(chunk.content)
                self.chunk_embeddings[chunk.id] = embedding

        similarities = []
        for chunk in chunks:
            embedding = self.chunk_embeddings.get(chunk.id)
            if embedding is not None:
                similarity = self.embedder.cosine_similarity(query_embedding, embedding)
                similarities.append((chunk, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        retrieved_chunks = []

        for i, (chunk, similarity) in enumerate(similarities[:k]):
            chunk.metadata["retrieval_score"] = float(similarity)
            chunk.metadata["retrieval_rank"] = i + 1
            retrieved_chunks.append(chunk)

        return retrieved_chunks

    def clear_index(self):
        """인덱스 초기화"""
        self.index = None
        self.chunk_embeddings.clear()
        self.chunk_map.clear()
        logger.info("검색 인덱스가 초기화되었습니다.")

    def get_index_stats(self) -> Dict[str, any]:
        """인덱스 통계 반환"""
        stats = {
            "num_embeddings": len(self.chunk_embeddings),
            "index_type": "FAISS" if self.use_faiss else "Brute Force",
            "memory_usage_mb": 0.0
        }

        if self.use_faiss and self.index:
            stats["faiss_vectors"] = self.index.ntotal

        if self.chunk_embeddings:
            sample_embedding = next(iter(self.chunk_embeddings.values()))
            embedding_size = sample_embedding.nbytes
            stats["memory_usage_mb"] = (
                    len(self.chunk_embeddings) * embedding_size / (1024 * 1024)
            )

        return stats