"""
기본 앙상블 추상 클래스
Base ensemble abstract class
"""

from abc import ABC, abstractmethod
from typing import List

from src.data_structures import RAGResponse


class BaseEnsemble(ABC):
    """앙상블 전략 추상 클래스"""

    @abstractmethod
    async def combine_responses(self, responses: List[RAGResponse]) -> RAGResponse:
        """여러 응답을 하나로 결합"""
        pass