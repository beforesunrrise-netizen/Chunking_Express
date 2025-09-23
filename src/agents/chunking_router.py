"""
청킹 라우터 - 텍스트 특성을 기반으로 최적의 청킹 전략을 선택
Chunking Router - Selects optimal chunking strategy based on text characteristics
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from src.config import ChunkingStrategy, Language
from src.data_structures import Document, Query
from .text_analyzer import TextAnalyzer, TextCharacteristics


@dataclass
class StrategyRecommendation:
    """전략 추천 결과"""
    primary_strategy: ChunkingStrategy
    confidence: float  # 0.0-1.0
    reasoning: str
    alternative_strategies: List[Tuple[ChunkingStrategy, float]]  # (전략, 점수)
    performance_estimate: Dict[str, str]  # 예상 성능 특성


class ChunkingRouter:
    """청킹 전략 라우터"""

    def __init__(self, language: Language = Language.ENGLISH):
        self.language = language
        self.text_analyzer = TextAnalyzer(language)

        # 전략별 특성 정의
        self.strategy_profiles = {
            ChunkingStrategy.SEMANTIC: {
                "complexity_weight": 0.8,
                "structure_weight": 0.6,
                "quality_priority": 0.9,
                "speed_priority": 0.2,
                "api_cost": "high",
                "best_for": ["academic", "technical", "scientific"],
                "min_text_length": 500,
                "description": "의미 단위 기반 고품질 청킹"
            },
            ChunkingStrategy.KEYWORD: {
                "complexity_weight": 0.4,
                "structure_weight": 0.9,
                "quality_priority": 0.7,
                "speed_priority": 0.3,
                "api_cost": "medium",
                "best_for": ["business", "news", "structured"],
                "min_text_length": 300,
                "description": "메타데이터 기반 구조화 청킹"
            },
            ChunkingStrategy.FIXED_SIZE: {
                "complexity_weight": 0.0,
                "structure_weight": 0.0,
                "quality_priority": 0.4,
                "speed_priority": 1.0,
                "api_cost": "none",
                "best_for": ["simple", "fast_processing"],
                "min_text_length": 0,
                "description": "고정 크기 기반 빠른 청킹"
            },
            ChunkingStrategy.QUERY_AWARE: {
                "complexity_weight": 0.7,
                "structure_weight": 0.5,
                "quality_priority": 0.8,
                "speed_priority": 0.3,
                "api_cost": "high",
                "best_for": ["query_specific", "targeted_search"],
                "min_text_length": 200,
                "description": "질의 특화 청킹"
            },
            ChunkingStrategy.RECURSIVE: {
                "complexity_weight": 0.5,
                "structure_weight": 0.7,
                "quality_priority": 0.6,
                "speed_priority": 0.5,
                "api_cost": "low",
                "best_for": ["long_documents", "hierarchical"],
                "min_text_length": 1000,
                "description": "재귀적 계층 청킹"
            },
            ChunkingStrategy.TEXT_SIMILARITY: {
                "complexity_weight": 0.6,
                "structure_weight": 0.3,
                "quality_priority": 0.5,
                "speed_priority": 0.4,
                "api_cost": "medium",
                "best_for": ["repetitive", "similar_content"],
                "min_text_length": 400,
                "description": "텍스트 유사도 기반 청킹"
            }
        }

        # 컨텍스트별 가중치
        self.context_weights = {
            "quality_focused": {"quality": 1.0, "speed": 0.3, "cost": 0.2},
            "speed_focused": {"quality": 0.4, "speed": 1.0, "cost": 0.8},
            "balanced": {"quality": 0.7, "speed": 0.7, "cost": 0.6},
            "cost_conscious": {"quality": 0.5, "speed": 0.6, "cost": 1.0}
        }

    async def recommend_strategy(
        self,
        document: Document,
        query: Optional[Query] = None,
        context: str = "balanced",
        force_no_api: bool = False
    ) -> StrategyRecommendation:
        """최적의 청킹 전략을 추천합니다."""

        logger.info(f"문서 {document.id}에 대한 청킹 전략 추천 시작 (컨텍스트: {context})")

        # 텍스트 특성 분석
        characteristics = await self.text_analyzer.analyze_document(document)

        # 각 전략에 대한 점수 계산
        strategy_scores = self._calculate_strategy_scores(
            characteristics, query, context, force_no_api
        )

        # 최고 점수 전략 선택
        best_strategy, best_score = max(strategy_scores.items(), key=lambda x: x[1])

        # 대안 전략들 (점수 순으로 정렬)
        alternatives = sorted(
            [(s, score) for s, score in strategy_scores.items() if s != best_strategy],
            key=lambda x: x[1],
            reverse=True
        )[:3]  # 상위 3개

        # 추천 이유 생성
        reasoning = self._generate_reasoning(
            best_strategy, characteristics, query, context
        )

        # 성능 예상치
        performance_estimate = self._estimate_performance(best_strategy, characteristics)

        recommendation = StrategyRecommendation(
            primary_strategy=best_strategy,
            confidence=best_score,
            reasoning=reasoning,
            alternative_strategies=alternatives,
            performance_estimate=performance_estimate
        )

        logger.info(f"추천 결과: {best_strategy.value} (신뢰도: {best_score:.2f})")
        return recommendation

    def _calculate_strategy_scores(
        self,
        characteristics: TextCharacteristics,
        query: Optional[Query],
        context: str,
        force_no_api: bool
    ) -> Dict[ChunkingStrategy, float]:
        """각 전략의 점수를 계산합니다."""

        scores = {}
        context_weight = self.context_weights.get(context, self.context_weights["balanced"])

        for strategy, profile in self.strategy_profiles.items():
            # API 사용 제한 검사
            if force_no_api and profile["api_cost"] in ["high", "medium"]:
                scores[strategy] = 0.0
                continue

            # 기본 적합도 점수
            base_score = self._calculate_base_score(strategy, characteristics, profile)

            # 쿼리 특화 보너스
            query_bonus = self._calculate_query_bonus(strategy, query)

            # 컨텍스트 가중치 적용
            context_score = self._apply_context_weights(
                strategy, profile, context_weight
            )

            # 최종 점수 계산
            final_score = (base_score * 0.5 + query_bonus * 0.2 + context_score * 0.3)
            scores[strategy] = min(final_score, 1.0)

        return scores

    def _calculate_base_score(
        self,
        strategy: ChunkingStrategy,
        characteristics: TextCharacteristics,
        profile: Dict[str, Any]
    ) -> float:
        """기본 적합도 점수 계산"""

        score_factors = []

        # 텍스트 길이 적합성
        min_length = profile["min_text_length"]
        if characteristics.length >= min_length:
            length_score = min(characteristics.length / (min_length * 3), 1.0)
        else:
            length_score = characteristics.length / min_length * 0.5

        score_factors.append(length_score)

        # 복잡도 적합성
        complexity_match = 1.0 - abs(
            characteristics.complexity_score - profile["complexity_weight"]
        )
        score_factors.append(complexity_match)

        # 구조화 적합성
        structure_match = 1.0 - abs(
            characteristics.structure_score - profile["structure_weight"]
        )
        score_factors.append(structure_match)

        # 도메인 적합성
        domain_score = self._calculate_domain_match(characteristics, profile)
        score_factors.append(domain_score)

        # 기술 용어 적합성
        if characteristics.has_technical_terms:
            if strategy in [ChunkingStrategy.SEMANTIC, ChunkingStrategy.KEYWORD]:
                score_factors.append(0.8)
            else:
                score_factors.append(0.4)
        else:
            score_factors.append(0.6)

        return sum(score_factors) / len(score_factors)

    def _calculate_domain_match(
        self,
        characteristics: TextCharacteristics,
        profile: Dict[str, Any]
    ) -> float:
        """도메인 적합성 계산"""

        best_domains = profile["best_for"]
        max_domain_score = 0.0

        for domain in best_domains:
            if domain in characteristics.domain_indicators:
                domain_score = characteristics.domain_indicators[domain]
                max_domain_score = max(max_domain_score, domain_score)

        # 특별한 경우들
        if "simple" in best_domains:
            # 단순한 텍스트에 적합
            simplicity_score = 1.0 - characteristics.complexity_score
            max_domain_score = max(max_domain_score, simplicity_score)

        if "long_documents" in best_domains:
            # 긴 문서에 적합
            if characteristics.length > 2000:
                length_bonus = min(characteristics.length / 5000, 1.0)
                max_domain_score = max(max_domain_score, length_bonus)

        return max_domain_score

    def _calculate_query_bonus(
        self,
        strategy: ChunkingStrategy,
        query: Optional[Query]
    ) -> float:
        """쿼리 관련 보너스 점수"""

        if not query:
            return 0.5  # 중립

        # 쿼리가 있는 경우 query-aware 전략에 보너스
        if strategy == ChunkingStrategy.QUERY_AWARE:
            return 1.0

        # 복잡한 쿼리인 경우 의미 기반 전략에 보너스
        query_complexity = len(query.question.split()) / 10.0
        if strategy == ChunkingStrategy.SEMANTIC and query_complexity > 0.5:
            return 0.8

        return 0.5

    def _apply_context_weights(
        self,
        strategy: ChunkingStrategy,
        profile: Dict[str, Any],
        context_weight: Dict[str, float]
    ) -> float:
        """컨텍스트 가중치 적용"""

        # 품질 우선도
        quality_score = profile["quality_priority"] * context_weight["quality"]

        # 속도 우선도
        speed_score = profile["speed_priority"] * context_weight["speed"]

        # 비용 우선도 (API 비용이 낮을수록 높은 점수)
        cost_mapping = {"none": 1.0, "low": 0.8, "medium": 0.5, "high": 0.2}
        cost_efficiency = cost_mapping.get(profile["api_cost"], 0.5)
        cost_score = cost_efficiency * context_weight["cost"]

        return (quality_score + speed_score + cost_score) / 3.0

    def _generate_reasoning(
        self,
        strategy: ChunkingStrategy,
        characteristics: TextCharacteristics,
        query: Optional[Query],
        context: str
    ) -> str:
        """추천 이유 생성"""

        profile = self.strategy_profiles[strategy]
        reasons = []

        # 텍스트 특성 기반 이유
        if characteristics.complexity_score > 0.7:
            reasons.append("복잡한 텍스트 구조")
        elif characteristics.complexity_score < 0.3:
            reasons.append("단순한 텍스트 구조")

        if characteristics.structure_score > 0.6:
            reasons.append("잘 구조화된 문서")

        if characteristics.has_technical_terms:
            reasons.append("기술 용어 포함")

        # 도메인 기반 이유
        max_domain = max(characteristics.domain_indicators.items(), key=lambda x: x[1])
        if max_domain[1] > 0.3:
            reasons.append(f"{max_domain[0]} 도메인 특성")

        # 쿼리 기반 이유
        if query and strategy == ChunkingStrategy.QUERY_AWARE:
            reasons.append("질의 최적화 필요")

        # 컨텍스트 기반 이유
        if context == "speed_focused":
            reasons.append("빠른 처리 우선")
        elif context == "quality_focused":
            reasons.append("고품질 결과 우선")
        elif context == "cost_conscious":
            reasons.append("비용 효율성 우선")

        # 전략 설명 추가
        reasons.append(profile["description"])

        return " | ".join(reasons)

    def _estimate_performance(
        self,
        strategy: ChunkingStrategy,
        characteristics: TextCharacteristics
    ) -> Dict[str, str]:
        """성능 예상치 계산"""

        profile = self.strategy_profiles[strategy]

        # 처리 속도 예상
        if profile["speed_priority"] > 0.8:
            speed = "매우 빠름"
        elif profile["speed_priority"] > 0.6:
            speed = "빠름"
        elif profile["speed_priority"] > 0.4:
            speed = "보통"
        else:
            speed = "느림"

        # 품질 예상
        if profile["quality_priority"] > 0.8:
            quality = "매우 높음"
        elif profile["quality_priority"] > 0.6:
            quality = "높음"
        elif profile["quality_priority"] > 0.4:
            quality = "보통"
        else:
            quality = "낮음"

        # 비용 예상
        cost_map = {
            "none": "무료",
            "low": "저비용",
            "medium": "중비용",
            "high": "고비용"
        }
        cost = cost_map[profile["api_cost"]]

        # 청크 수 예상
        estimated_chunks = self._estimate_chunk_count(strategy, characteristics)

        return {
            "속도": speed,
            "품질": quality,
            "비용": cost,
            "예상_청크_수": f"약 {estimated_chunks}개"
        }

    def _estimate_chunk_count(
        self,
        strategy: ChunkingStrategy,
        characteristics: TextCharacteristics
    ) -> int:
        """청크 수 예상"""

        text_length = characteristics.length

        if strategy == ChunkingStrategy.FIXED_SIZE:
            # 고정 크기: 512자 기준
            return max(1, text_length // 512)
        elif strategy == ChunkingStrategy.SEMANTIC:
            # 의미 기반: 가변적이지만 보통 더 큰 청크
            return max(1, text_length // 800)
        elif strategy == ChunkingStrategy.KEYWORD:
            # 키워드 기반: 구조에 따라 다름
            base_chunks = max(1, text_length // 600)
            return base_chunks + 2  # 제목, 요약 등 추가
        else:
            # 기타: 중간 정도
            return max(1, text_length // 700)

    def get_strategy_comparison(
        self,
        document: Document,
        context: str = "balanced"
    ) -> Dict[str, Dict[str, Any]]:
        """모든 전략의 비교 정보 제공"""

        comparison = {}

        for strategy, profile in self.strategy_profiles.items():
            comparison[strategy.value] = {
                "설명": profile["description"],
                "적합한_도메인": profile["best_for"],
                "최소_텍스트_길이": profile["min_text_length"],
                "API_비용": profile["api_cost"],
                "품질_우선도": profile["quality_priority"],
                "속도_우선도": profile["speed_priority"]
            }

        return comparison