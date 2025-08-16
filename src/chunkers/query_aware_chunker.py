"""
쿼리 인식 청킹 전략 (수정된 버전)
Query-aware chunking strategy (Fixed version)
"""

import json
import time
from typing import List, Dict, Any
import openai
from loguru import logger

from src.config import Language, ChunkingStrategy
from src.data_structures import Document, Query, Chunk
from .base_chunker import BaseChunker
from openai import AsyncOpenAI
from src.config import APIConfig

class QueryAwareChunker(BaseChunker):
    """쿼리 인식 청킹 전략 구현"""

    def __init__(self, language: Language, chunk_size_limit: int = 512):
        super().__init__(language, chunk_size_limit)
        self.strategy = ChunkingStrategy.QUERY_AWARE
        config = APIConfig()
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.min_relevance_score = 0.3

    async def chunk_document(self, document: Document) -> List[Chunk]:
        """일반 문서 청킹 (쿼리 없이)"""
        logger.info("쿼리 없이 호출됨 - 의미 기반 청킹으로 대체")
        from .semantic_chunker import SemanticChunker
        semantic_chunker = SemanticChunker(self.language, self.chunk_size_limit)
        return await semantic_chunker.chunk_document(document)

    async def query_aware_chunk(self, document: Document, query: Query) -> List[Chunk]:
        """쿼리에 최적화된 청킹"""
        start_time = time.time()

        try:
            # 1. 쿼리 분석
            query_analysis = await self._analyze_query(query.question)

            # 2. 관련 섹션 식별 및 청킹
            chunks = await self._extract_query_relevant_chunks(
                document,
                query.question,
                query_analysis
            )

            # 3. 관련성 점수 기반 필터링
            filtered_chunks = self._filter_by_relevance(chunks)

            # 4. 청크 최적화 (병합/분할)
            optimized_chunks = self._optimize_chunks(filtered_chunks)

            processing_time = time.time() - start_time
            metrics = self.calculate_metrics(optimized_chunks, processing_time)
            logger.info(
                f"쿼리 인식 청킹 완료 - {metrics.num_chunks}개 청크, "
                f"평균 관련성: {self._calculate_avg_relevance(optimized_chunks):.2f}"
            )

            return optimized_chunks

        except Exception as e:
            logger.error(f"쿼리 인식 청킹 실패: {e}")
            # 예외 발생 시 fallback으로 일반 의미 청킹 사용
            fallback_chunks = await self.chunk_document(document)
            return fallback_chunks

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """쿼리 분석 및 정보 추출"""
        prompt = self._create_query_analysis_prompt(query)

        logger.debug("--- 함수 정밀 검사 시작 ---")
        try:
            logger.debug(f"호출하려는 함수 타입: {type(self.client.chat.completions.create)}")
        except Exception as inspect_e:
            logger.error(f"함수 검사 중 오류 발생: {inspect_e}")
        logger.debug("--- 함수 정밀 검사 종료 ---")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_query_analysis_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"쿼리 분석 실패: {e}")
            return {
                "question_type": "information",
                "key_entities": [],
                "keywords": query.split()[:5],
                "required_info": ["general"],
                "complexity": "moderate"
            }

        except Exception as e:
            logger.error(f"쿼리 분석 실패: {e}")
            # --- 여기가 중요합니다 ---
            # None을 반환하는 대신, 비어있더라도 반드시 dictionary를 반환해야 합니다.
            return {
                "question_type": "information",
                "key_entities": [],
                "keywords": query.split()[:5],
                "required_info": ["general"],
                "complexity": "moderate"
            }

    async def _extract_query_relevant_chunks(
            self,
            document: Document,
            query: str,
            query_analysis: Dict[str, Any]
    ) -> List[Chunk]:
        """쿼리와 관련된 청크 추출"""
        prompt = self._create_extraction_prompt(
            document.content,
            query,
            query_analysis
        )

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_extraction_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            extracted_chunks_data = result.get("chunks", [])

            if not extracted_chunks_data:
                logger.warning(f"LLM이 관련 청크를 찾지 못했습니다. Fallback 청킹을 사용합니다. Doc ID: {document.id}")
                return self._fallback_chunking(document, query_analysis)

            return self._create_chunks_from_extraction(result, document, query_analysis)

        except Exception as e:
            logger.error(f"청크 추출 실패: {e}")
            return self._fallback_chunking(document, query_analysis)

    def _fallback_chunking(self, document: Document, query_analysis: Dict[str, Any]) -> List[Chunk]:
        """예외 발생 시 사용할 fallback 청킹"""
        content = document.content
        chunk_size = self.chunk_size_limit
        chunks = []

        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i + chunk_size]

            metadata = {
                "relevance_score": 0.5,
                "relevance_reason": "fallback chunking",
                "answer_contribution": "general content",
                "keywords": query_analysis.get("keywords", []),
                "query_analysis": query_analysis,
                "coverage": "unknown"
            }

            chunk = self.create_chunk(
                content=chunk_content,
                document_id=document.id,
                start_idx=i,
                end_idx=min(i + chunk_size, len(content)),
                sequence_num=len(chunks),
                metadata=metadata
            )
            chunks.append(chunk)

        return chunks

    def _filter_by_relevance(self, chunks: List[Chunk]) -> List[Chunk]:
        """관련성 점수 기반 필터링"""
        filtered = []

        for chunk in chunks:
            relevance_score = chunk.metadata.get("relevance_score", 0)
            if relevance_score >= self.min_relevance_score:
                filtered.append(chunk)
            else:
                logger.debug(f"청크 필터링됨 - 낮은 관련성: {relevance_score:.2f}")

        if not filtered and chunks:
            sorted_chunks = sorted(chunks, key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
            filtered = sorted_chunks[:3]
            logger.info(f"모든 청크가 필터링되어 상위 {len(filtered)}개 청크 유지")

        return filtered

    def _optimize_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """청크 최적화 (병합/분할)"""
        if not chunks:
            return chunks

        optimized = []
        current_group = []
        current_size = 0

        sorted_chunks = sorted(chunks, key=lambda x: (x.start_idx, -x.metadata.get("relevance_score", 0)))

        for chunk in sorted_chunks:
            chunk_size = len(chunk.content)

            if current_group and self._can_merge(current_group[-1], chunk):
                if current_size + chunk_size <= self.chunk_size_limit * 1.5:
                    current_group.append(chunk)
                    current_size += chunk_size
                else:
                    merged = self._merge_chunk_group(current_group)
                    optimized.append(merged)
                    current_group = [chunk]
                    current_size = chunk_size
            else:
                if current_group:
                    merged = self._merge_chunk_group(current_group)
                    optimized.append(merged)
                current_group = [chunk]
                current_size = chunk_size

        if current_group:
            merged = self._merge_chunk_group(current_group)
            optimized.append(merged)

        for i, chunk in enumerate(optimized):
            chunk.sequence_num = i

        return optimized

    def _can_merge(self, chunk1: Chunk, chunk2: Chunk) -> bool:
        """두 청크가 병합 가능한지 확인"""
        if chunk1.end_idx >= chunk2.start_idx - 100:
            score1 = chunk1.metadata.get("relevance_score", 0)
            score2 = chunk2.metadata.get("relevance_score", 0)
            if abs(score1 - score2) < 0.3:
                return True
        return False

    def _merge_chunk_group(self, chunks: List[Chunk]) -> Chunk:
        """청크 그룹을 하나로 병합"""
        if len(chunks) == 1:
            return chunks[0]

        merged_content = " ".join(chunk.content for chunk in chunks)
        avg_relevance = sum(chunk.metadata.get("relevance_score", 0) for chunk in chunks) / len(chunks)
        all_keywords = list(set(kw for chunk in chunks for kw in chunk.metadata.get("keywords", [])))
        all_topics = list(set(chunk.metadata.get("topic", "") for chunk in chunks if chunk.metadata.get("topic")))

        merged_metadata = {
            "relevance_score": avg_relevance,
            "keywords": all_keywords,
            "topics": all_topics,
            "merged_from": len(chunks),
            "original_chunks": [chunk.id for chunk in chunks]
        }

        return self.create_chunk(
            content=merged_content,
            document_id=chunks[0].document_id,
            start_idx=chunks[0].start_idx,
            end_idx=chunks[-1].end_idx,
            sequence_num=chunks[0].sequence_num,
            metadata=merged_metadata
        )

    def _calculate_avg_relevance(self, chunks: List[Chunk]) -> float:
        """평균 관련성 점수 계산"""
        if not chunks:
            return 0.0
        total_relevance = sum(chunk.metadata.get("relevance_score", 0) for chunk in chunks)
        return total_relevance / len(chunks)

    def _get_query_analysis_system_prompt(self) -> str:
        """쿼리 분석 시스템 프롬프트"""
        return "Analyze the question to extract key information. Identify question type, required information, and key keywords."

    def _get_extraction_system_prompt(self) -> str:
        """청크 추출 시스템 프롬프트"""
        return "Extract document parts necessary to answer the question. Evaluate relevance of each part and include necessary context. Prioritize information directly helpful for answering."

    def _create_query_analysis_prompt(self, query: str) -> str:
        """쿼리 분석 프롬프트"""
        return f"""Analyze the following question:

Question: {query}

Response format (JSON):
{{
    "question_type": "information/comparison/explanation/other",
    "key_entities": ["entity1", "entity2"],
    "keywords": ["keyword1", "keyword2"],
    "required_info": ["info1", "info2"],
    "complexity": "simple/moderate/complex"
}}"""

    def _create_extraction_prompt(
            self,
            content: str,
            query: str,
            query_analysis: Dict[str, Any]
    ) -> str:
        """청크 추출 프롬프트"""
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        keywords = ", ".join(query_analysis.get("keywords", []))

        return f"""Extract relevant parts from the document to answer the question.

Question: {query}
Key keywords: {keywords}

Document:
{content}

Response format (JSON):
{{
    "chunks": [
        {{
            "content": "relevant content",
            "start_idx": 0,
            "end_idx": 100,
            "relevance_score": 0.0-1.0,
            "relevance_reason": "relevance reason",
            "answer_contribution": "information contributing to answer",
            "keywords": ["included_keywords"]
        }}
    ],
    "coverage": "answer coverage for the question (0-100%)"
}}"""

    def _create_chunks_from_extraction(
            self,
            extraction_result: Dict[str, Any],
            document: Document,
            query_analysis: Dict[str, Any]
    ) -> List[Chunk]:
        """추출 결과로부터 청크 생성"""
        chunks = []

        for i, chunk_data in enumerate(extraction_result.get("chunks", [])):
            metadata = {
                "relevance_score": chunk_data.get("relevance_score", 0.5),
                "relevance_reason": chunk_data.get("relevance_reason", ""),
                "answer_contribution": chunk_data.get("answer_contribution", ""),
                "keywords": chunk_data.get("keywords", []),
                "query_analysis": query_analysis,
                "coverage": extraction_result.get("coverage", "0%")
            }

            chunk = self.create_chunk(
                content=chunk_data.get("content", ""),
                document_id=document.id,
                start_idx=chunk_data.get("start_idx", 0),
                end_idx=chunk_data.get("end_idx", len(chunk_data.get("content", ""))),
                sequence_num=i,
                metadata=metadata
            )
            chunks.append(chunk)

        return chunks
