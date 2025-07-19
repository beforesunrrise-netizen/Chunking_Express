"""
기본 평가 추상 클래스
Base evaluator abstract class
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from src.data_structures import RAGResponse, EvaluationResult


class BaseEvaluator(ABC):
    """평가 추상 기본 클래스"""

    @abstractmethod
    async def evaluate_responses(
        self,
        responses: List[RAGResponse],
        ground_truths: List[str]
    ) -> EvaluationResult:
        """응답들을 평가"""
        pass

    @abstractmethod
    async def evaluate_single_response(
        self,
        response: RAGResponse,
        ground_truth: str
    ) -> Dict[str, float]:
        """단일 응답 평가"""
        pass