"""
벡터 기반 검색 구현 (정규화 문제 수정 완료)
Vector-based retriever implementation (Normalization issue corrected)
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
        self.index: Optional[faiss.Index] = None
        self.chunk_embeddings: Dict[str, np.ndarray] = {}
        self.chunk_map: Dict[int, str] = {}

    async def retrieve(self, query_text: str, chunks: List[Chunk], k: int = 5) -> List[Chunk]:
        """쿼리에 대해 관련 청크 검색"""
        if not chunks:
            logger.warning("검색할 청크가 없어 빈 리스트를 반환합니다.")
            return []

        question_string = query_text
        query_embedding = await self.embedder.embed_text(question_string)

        # 항상 현재 청크들로 인덱스를 새로 구축
        await self.build_index(chunks)

        if self.use_faiss and self.index is not None:
            retrieved_chunks = await self._faiss_search(query_embedding, chunks, k)
        else:
            # FAISS를 사용하지 않을 경우, 코사인 유사도로 직접 비교
            retrieved_chunks = await self._brute_force_search(query_embedding, chunks, k)

        logger.info(f"검색 완료: {len(retrieved_chunks)}개 청크 반환")
        return retrieved_chunks

    async def build_index(self, chunks: List[Chunk]) -> None:
        """FAISS 인덱스 구축"""
        if not chunks:
            return

        embeddings = await self.embedder.embed_chunks(chunks)
        if not embeddings:
            self.index = None
            return

        self.chunk_embeddings.clear()
        self.chunk_map.clear()

        valid_embeddings = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is not None:
                self.chunk_embeddings[chunk.id] = embedding
                self.chunk_map[len(valid_embeddings)] = chunk.id
                valid_embeddings.append(embedding)

        if not valid_embeddings or not self.use_faiss:
            self.index = None
            return

        dimension = valid_embeddings[0].shape[0]

        self.index = faiss.IndexFlatIP(dimension)

        embeddings_matrix = np.array(valid_embeddings).astype('float32')


        faiss.normalize_L2(embeddings_matrix)

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

        k = min(k, self.index.ntotal)

        query_vector = query_embedding.reshape(1, -1).astype('float32')

        # <<< 수정된 부분: 검색할 쿼리 벡터 정규화 >>>
        faiss.normalize_L2(query_vector)

        # 내적(유사도) 점수와 인덱스를 검색
        similarities, indices = self.index.search(query_vector, k)

        retrieved_chunks = []
        chunk_dict = {chunk.id: chunk for chunk in chunks}

        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < 0: continue
            chunk_id = self.chunk_map.get(idx)
            if chunk_id and chunk_id in chunk_dict:
                chunk = chunk_dict[chunk_id]
                chunk.metadata["retrieval_score"] = float(sim)
                chunk.metadata["retrieval_rank"] = i + 1
                retrieved_chunks.append(chunk)

        return retrieved_chunks

    async def _brute_force_search(
            self,
            query_embedding: np.ndarray,
            chunks: List[Chunk],
            k: int
    ) -> List[Chunk]:
        """전수 검색 (Brute Force) - 코사인 유사도 직접 계산"""
        similarities = []
        for chunk in chunks:
            embedding = self.chunk_embeddings.get(chunk.id)
            if embedding is not None:
                # BaseEmbedder에 cosine_similarity가 있다고 가정
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