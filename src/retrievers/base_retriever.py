"""
기본 검색 추상 클래스
Base retriever abstract class
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from src.data_structures import Query, Chunk


class BaseRetriever(ABC):
    """검색 추상 기본 클래스"""

    @abstractmethod
    async def retrieve(self, query: Query, chunks: List[Chunk], k: int = 5) -> List[Chunk]:
        """쿼리에 대해 관련 청크 검색"""
        pass

    @abstractmethod
    async def build_index(self, chunks: List[Chunk]) -> None:
        """검색을 위한 인덱스 구축"""
        pass

    def rerank(self, query: Query, chunks: List[Chunk], scores: List[float]) -> List[Chunk]:
        """검색 결과 재순위화"""
        # 점수와 청크를 쌍으로 묶어 정렬
        chunk_score_pairs = list(zip(chunks, scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 정렬된 청크만 반환
        return [chunk for chunk, _ in chunk_score_pairs]