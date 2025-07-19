"""
재순위 기반 앙상블
Reranking-based ensemble
"""

from typing import List, Tuple
import numpy as np
from rouge_score import rouge_scorer
from loguru import logger

from src.data_structures import RAGResponse
from .base_ensemble import BaseEnsemble


class RerankingEnsemble(BaseEnsemble):
    """재순위 기반 앙상블"""

    def __init__(self, reranking_method: str = "bert_score"):
        """
        Args:
            reranking_method: "bert_score", "rouge", "combined"
        """
        self.reranking_method = reranking_method
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    async def combine_responses(self, responses: List[RAGResponse]) -> RAGResponse:
        """재순위를 통해 최적 응답 선택"""
        if not responses:
            raise ValueError("응답 리스트가 비어있습니다.")

        if len(responses) == 1:
            return responses[0]

        # 재순위 점수 계산
        reranked_responses = await self._rerank_responses(responses)

        # 최고 점수 응답 선택
        best_response = reranked_responses[0][1]
        best_score = reranked_responses[0][0]

        # 메타데이터 업데이트
        best_response.metadata["ensemble_method"] = f"reranking_{self.reranking_method}"
        best_response.metadata["reranking_score"] = best_score
        best_response.metadata["reranking_scores"] = [score for score, _ in reranked_responses]

        logger.info(f"재순위 최고 점수: {best_score:.3f}")

        return best_response

    async def _rerank_responses(self, responses: List[RAGResponse]) -> List[Tuple[float, RAGResponse]]:
        """응답들을 재순위화"""
        scored_responses = []

        for response in responses:
            if self.reranking_method == "rouge":
                score = self._calculate_rouge_score(response)
            elif self.reranking_method == "bert_score":
                score = await self._calculate_bert_score(response)
            else:  # combined
                rouge_score = self._calculate_rouge_score(response)
                bert_score = await self._calculate_bert_score(response)
                score = 0.5 * rouge_score + 0.5 * bert_score

            scored_responses.append((score, response))

        # 점수 기준 내림차순 정렬
        scored_responses.sort(key=lambda x: x[0], reverse=True)

        return scored_responses

    def _calculate_rouge_score(self, response: RAGResponse) -> float:
        """ROUGE 점수 계산"""
        if not response.chunks_used:
            return 0.0

        # 청크 내용과 응답 간의 ROUGE 점수 계산
        rouge_scores = []

        for chunk in response.chunks_used:
            scores = self.rouge_scorer.score(chunk.content, response.response)
            # ROUGE-L F1 점수 사용
            rouge_scores.append(scores['rougeL'].fmeasure)

        # 평균 ROUGE 점수
        avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0

        # 응답 길이 페널티 (너무 짧거나 긴 응답 불이익)
        response_length = len(response.response.split())
        length_penalty = 1.0

        if response_length < 20:
            length_penalty = 0.7
        elif response_length > 500:
            length_penalty = 0.8

        return avg_rouge * length_penalty

    async def _calculate_bert_score(self, response: RAGResponse) -> float:
        """BERT Score 계산 (간소화된 버전)"""
        # 실제 구현에서는 bert_score 라이브러리 사용
        # 여기서는 간단한 휴리스틱 사용

        if not response.chunks_used:
            return 0.0

        # 응답과 청크 간의 단어 중복도 계산
        response_words = set(response.response.lower().split())

        overlap_scores = []
        for chunk in response.chunks_used:
            chunk_words = set(chunk.content.lower().split())
            if chunk_words:
                overlap = len(response_words.intersection(chunk_words))
                overlap_ratio = overlap / len(chunk_words)
                overlap_scores.append(overlap_ratio)

        # 평균 중복도
        avg_overlap = np.mean(overlap_scores) if overlap_scores else 0.0

        # 신뢰도 가중치
        confidence_weight = response.confidence

        return avg_overlap * confidence_weight