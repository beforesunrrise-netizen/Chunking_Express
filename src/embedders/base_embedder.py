"""
기본 임베딩 추상 클래스
Base embedder abstract class
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

from src.data_structures import Chunk


class BaseEmbedder(ABC):
    """임베딩 추상 기본 클래스"""

    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """텍스트를 임베딩으로 변환"""
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """여러 텍스트를 임베딩으로 변환 (배치 처리)"""
        pass

    @abstractmethod
    async def embed_chunks(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """청크 리스트를 임베딩으로 변환"""
        pass

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """유클리드 거리 계산"""
        return np.linalg.norm(embedding1 - embedding2)