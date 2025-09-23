"""
청킹 에이전트 - 라우터의 추천을 받아 실제 청킹을 수행하는 통합 인터페이스
Chunking Agent - Unified interface that performs actual chunking based on router recommendations
"""

import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from src.config import ChunkingStrategy, Language
from src.data_structures import Document, Query, Chunk

# 청킹 전략 임포트
from src.chunkers import (
    SemanticChunker, KeywordChunker, QueryAwareChunker,
    FixedSizeChunker, RecursiveChunker, Text_Similarity
)

from .chunking_router import ChunkingRouter, StrategyRecommendation
from .text_analyzer import TextAnalyzer


@dataclass
class ChunkingResult:
    """청킹 결과"""
    chunks: List[Chunk]
    strategy_used: ChunkingStrategy
    recommendation: StrategyRecommendation
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    fallback_used: bool = False
    performance_metrics: Optional[Dict[str, Any]] = None


class ChunkingAgent:
    """지능형 청킹 에이전트"""

    def __init__(
        self,
        language: Language = Language.ENGLISH,
        chunk_size_limit: int = 512,
        overlap_ratio: float = 0.1,
        default_context: str = "balanced"
    ):
        self.language = language
        self.chunk_size_limit = chunk_size_limit
        self.overlap_ratio = overlap_ratio
        self.default_context = default_context

        # 라우터와 분석기 초기화
        self.router = ChunkingRouter(language)
        self.text_analyzer = TextAnalyzer(language)

        # 청킹 객체 캐시
        self._chunker_cache = {}

        # 성능 통계
        self.performance_stats = {
            "total_documents_processed": 0,
            "strategy_usage_count": {},
            "average_execution_times": {},
            "fallback_usage_count": 0,
            "success_rate": 0.0
        }

        logger.info(f"청킹 에이전트 초기화 완료 (언어: {language.value})")

    async def chunk_document(
        self,
        document: Document,
        query: Optional[Query] = None,
        context: Optional[str] = None,
        force_strategy: Optional[ChunkingStrategy] = None,
        force_no_api: bool = False
    ) -> ChunkingResult:
        """문서를 지능적으로 청킹합니다."""

        start_time = time.time()
        context = context or self.default_context

        logger.info(f"문서 {document.id} 청킹 시작 (컨텍스트: {context})")

        try:
            # 1. 전략 결정
            if force_strategy:
                # 강제 지정된 전략 사용
                strategy = force_strategy
                recommendation = StrategyRecommendation(
                    primary_strategy=strategy,
                    confidence=1.0,
                    reasoning="사용자 지정 전략",
                    alternative_strategies=[],
                    performance_estimate={}
                )
                logger.info(f"강제 지정된 전략 사용: {strategy.value}")
            else:
                # 라우터를 통한 전략 추천
                recommendation = await self.router.recommend_strategy(
                    document, query, context, force_no_api
                )
                strategy = recommendation.primary_strategy
                logger.info(f"추천된 전략: {strategy.value} (신뢰도: {recommendation.confidence:.2f})")

            # 2. 청킹 실행
            chunks = await self._execute_chunking(document, query, strategy)

            # 3. 결과 검증
            if not chunks or len(chunks) == 0:
                logger.warning(f"청킹 결과가 비어있음. 폴백 전략 사용")
                chunks, strategy = await self._execute_fallback(document, query)
                fallback_used = True
            else:
                fallback_used = False

            # 4. 성능 메트릭 계산
            execution_time = time.time() - start_time
            performance_metrics = self._calculate_performance_metrics(
                chunks, execution_time, strategy
            )

            # 5. 통계 업데이트
            self._update_statistics(strategy, execution_time, True, fallback_used)

            result = ChunkingResult(
                chunks=chunks,
                strategy_used=strategy,
                recommendation=recommendation,
                execution_time=execution_time,
                success=True,
                fallback_used=fallback_used,
                performance_metrics=performance_metrics
            )

            logger.success(f"문서 {document.id} 청킹 완료: {len(chunks)}개 청크 생성 "
                          f"(전략: {strategy.value}, 시간: {execution_time:.2f}초)")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"청킹 실패: {e}", exc_info=True)

            # 최후의 폴백 시도
            try:
                chunks, fallback_strategy = await self._execute_emergency_fallback(document)
                self._update_statistics(fallback_strategy, execution_time, True, True)

                return ChunkingResult(
                    chunks=chunks,
                    strategy_used=fallback_strategy,
                    recommendation=recommendation if 'recommendation' in locals() else None,
                    execution_time=execution_time,
                    success=True,
                    error_message=str(e),
                    fallback_used=True
                )
            except Exception as fallback_error:
                self._update_statistics(ChunkingStrategy.FIXED_SIZE, execution_time, False, True)

                return ChunkingResult(
                    chunks=[],
                    strategy_used=ChunkingStrategy.FIXED_SIZE,
                    recommendation=recommendation if 'recommendation' in locals() else None,
                    execution_time=execution_time,
                    success=False,
                    error_message=f"청킹 실패: {e}, 폴백도 실패: {fallback_error}",
                    fallback_used=True
                )

    async def _execute_chunking(
        self,
        document: Document,
        query: Optional[Query],
        strategy: ChunkingStrategy
    ) -> List[Chunk]:
        """지정된 전략으로 청킹을 실행합니다."""

        chunker = self._get_chunker(strategy)

        if strategy == ChunkingStrategy.QUERY_AWARE and query:
            chunks = await chunker.query_aware_chunk(document, query)
        else:
            chunks = await chunker.chunk_document(document)

        # 청크 검증 및 후처리
        return self._validate_and_process_chunks(chunks, document)

    def _get_chunker(self, strategy: ChunkingStrategy):
        """청킹 객체를 가져오거나 생성합니다 (캐싱 적용)."""

        if strategy in self._chunker_cache:
            return self._chunker_cache[strategy]

        # 전략별 청킹 객체 생성
        if strategy == ChunkingStrategy.SEMANTIC:
            chunker = SemanticChunker(self.language, self.chunk_size_limit)
        elif strategy == ChunkingStrategy.KEYWORD:
            chunker = KeywordChunker(self.language, self.chunk_size_limit)
        elif strategy == ChunkingStrategy.QUERY_AWARE:
            chunker = QueryAwareChunker(self.language, self.chunk_size_limit)
        elif strategy == ChunkingStrategy.FIXED_SIZE:
            chunker = FixedSizeChunker(self.language, self.chunk_size_limit, self.overlap_ratio)
        elif strategy == ChunkingStrategy.RECURSIVE:
            chunker = RecursiveChunker(self.language, self.chunk_size_limit)
        elif strategy == ChunkingStrategy.TEXT_SIMILARITY:
            chunker = Text_Similarity(self.language, self.chunk_size_limit)
        else:
            logger.warning(f"알 수 없는 전략 {strategy}, FixedSize 사용")
            chunker = FixedSizeChunker(self.language, self.chunk_size_limit, self.overlap_ratio)

        # 캐시에 저장
        self._chunker_cache[strategy] = chunker
        return chunker

    def _validate_and_process_chunks(
        self,
        chunks: List[Chunk],
        document: Document
    ) -> List[Chunk]:
        """청크를 검증하고 후처리합니다."""

        if not chunks:
            return chunks

        validated_chunks = []

        for chunk in chunks:
            # doc_id 설정 (누락된 경우)
            if not hasattr(chunk, 'doc_id') or not chunk.doc_id:
                chunk.doc_id = document.id

            # 빈 청크 제거
            if not chunk.content or not chunk.content.strip():
                logger.warning(f"빈 청크 제거: {chunk.id}")
                continue

            # 너무 짧은 청크 제거 (10자 미만)
            if len(chunk.content.strip()) < 10:
                logger.warning(f"너무 짧은 청크 제거: {chunk.id} ({len(chunk.content)}자)")
                continue

            validated_chunks.append(chunk)

        return validated_chunks

    async def _execute_fallback(
        self,
        document: Document,
        query: Optional[Query]
    ) -> Tuple[List[Chunk], ChunkingStrategy]:
        """폴백 전략을 실행합니다."""

        logger.info("폴백 전략 실행: FixedSizeChunker 사용")

        try:
            chunker = FixedSizeChunker(self.language, self.chunk_size_limit, self.overlap_ratio)
            chunks = await chunker.chunk_document(document)
            chunks = self._validate_and_process_chunks(chunks, document)

            if chunks:
                return chunks, ChunkingStrategy.FIXED_SIZE
        except Exception as e:
            logger.error(f"폴백 전략도 실패: {e}")

        # 최후의 수단: 단순 분할
        return await self._execute_emergency_fallback(document)

    async def _execute_emergency_fallback(
        self,
        document: Document
    ) -> Tuple[List[Chunk], ChunkingStrategy]:
        """비상 폴백 전략을 실행합니다."""

        logger.warning("비상 폴백 전략 실행: 단순 텍스트 분할")

        text = document.content
        max_chunk_size = self.chunk_size_limit

        chunks = []
        for i in range(0, len(text), max_chunk_size):
            chunk_text = text[i:i + max_chunk_size]

            chunk = Chunk(
                id=f"{document.id}_emergency_{len(chunks)}",
                content=chunk_text,
                document_id=document.id,
                start_idx=i,
                end_idx=min(i + max_chunk_size, len(text)),
                strategy=ChunkingStrategy.FIXED_SIZE,
                sequence_num=len(chunks),
                metadata={"emergency_fallback": True}
            )
            chunk.doc_id = document.id
            chunks.append(chunk)

        return chunks, ChunkingStrategy.FIXED_SIZE

    def _calculate_performance_metrics(
        self,
        chunks: List[Chunk],
        execution_time: float,
        strategy: ChunkingStrategy
    ) -> Dict[str, Any]:
        """성능 메트릭을 계산합니다."""

        if not chunks:
            return {}

        chunk_sizes = [len(chunk.content) for chunk in chunks]

        return {
            "chunk_count": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes),
            "execution_time_seconds": execution_time,
            "chunks_per_second": len(chunks) / execution_time if execution_time > 0 else 0,
            "strategy_used": strategy.value
        }

    def _update_statistics(
        self,
        strategy: ChunkingStrategy,
        execution_time: float,
        success: bool,
        fallback_used: bool
    ):
        """성능 통계를 업데이트합니다."""

        self.performance_stats["total_documents_processed"] += 1

        # 전략 사용 횟수
        strategy_name = strategy.value
        if strategy_name not in self.performance_stats["strategy_usage_count"]:
            self.performance_stats["strategy_usage_count"][strategy_name] = 0
        self.performance_stats["strategy_usage_count"][strategy_name] += 1

        # 평균 실행 시간
        if strategy_name not in self.performance_stats["average_execution_times"]:
            self.performance_stats["average_execution_times"][strategy_name] = []
        self.performance_stats["average_execution_times"][strategy_name].append(execution_time)

        # 폴백 사용 횟수
        if fallback_used:
            self.performance_stats["fallback_usage_count"] += 1

        # 성공률 계산
        total_processed = self.performance_stats["total_documents_processed"]
        total_successes = sum(
            1 for _ in range(total_processed) if success
        )  # 단순화된 계산
        self.performance_stats["success_rate"] = total_successes / total_processed

    async def batch_chunk_documents(
        self,
        documents: List[Document],
        queries: Optional[List[Query]] = None,
        context: str = "balanced"
    ) -> List[ChunkingResult]:
        """문서들을 배치로 청킹합니다."""

        logger.info(f"{len(documents)}개 문서 배치 청킹 시작")

        results = []
        for i, document in enumerate(documents):
            query = queries[i] if queries and i < len(queries) else None
            result = await self.chunk_document(document, query, context)
            results.append(result)

        logger.info(f"배치 청킹 완료: {len(results)}개 결과")
        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보를 반환합니다."""

        # 평균 실행 시간 계산
        avg_times = {}
        for strategy, times in self.performance_stats["average_execution_times"].items():
            if times:
                avg_times[strategy] = sum(times) / len(times)

        return {
            "총_처리_문서수": self.performance_stats["total_documents_processed"],
            "전략별_사용횟수": self.performance_stats["strategy_usage_count"],
            "전략별_평균실행시간": avg_times,
            "폴백_사용횟수": self.performance_stats["fallback_usage_count"],
            "성공률": f"{self.performance_stats['success_rate']:.2%}",
            "폴백_사용률": (
                f"{self.performance_stats['fallback_usage_count'] / max(1, self.performance_stats['total_documents_processed']):.2%}"
            )
        }

    async def analyze_document_characteristics(self, document: Document) -> Dict[str, Any]:
        """문서 특성 분석 결과를 반환합니다."""

        characteristics = await self.text_analyzer.analyze_document(document)

        return {
            "텍스트_길이": characteristics.length,
            "문장_수": characteristics.sentence_count,
            "구조화_점수": f"{characteristics.structure_score:.2f}",
            "복잡도_점수": f"{characteristics.complexity_score:.2f}",
            "주요_도메인": max(characteristics.domain_indicators.items(), key=lambda x: x[1])[0],
            "기술용어_포함": characteristics.has_technical_terms,
            "가독성_점수": f"{characteristics.readability_score:.2f}"
        }

    def reset_statistics(self):
        """통계를 초기화합니다."""

        self.performance_stats = {
            "total_documents_processed": 0,
            "strategy_usage_count": {},
            "average_execution_times": {},
            "fallback_usage_count": 0,
            "success_rate": 0.0
        }
        logger.info("성능 통계가 초기화되었습니다")