"""
쿼리 인식 청킹 전략
Query-aware chunking strategy
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
        config = APIConfig()  # 환경변수에서 가져온 config 사용
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.min_relevance_score = 0.3  # 최소 관련성 점수

    async def chunk_document(self, document: Document) -> List[Chunk]:
        """일반 문서 청킹 (쿼리 없이)"""
        # 쿼리 인식 청킹은 쿼리가 필요하므로, 일반적인 의미 기반 청킹으로 대체
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

            # 메트릭 계산 및 로깅
            processing_time = time.time() - start_time
            metrics = self.calculate_metrics(optimized_chunks, processing_time)
            logger.info(
                f"쿼리 인식 청킹 완료 - {metrics.num_chunks}개 청크, "
                f"평균 관련성: {self._calculate_avg_relevance(optimized_chunks):.2f}"
            )

            return optimized_chunks

        except Exception as e:
            logger.error(f"쿼리 인식 청킹 실패: {e}")
            return await self.chunk_document(document)

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """쿼리 분석 및 정보 추출"""
        prompt = self._create_query_analysis_prompt(query)

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
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

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
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
        return self._create_chunks_from_extraction(result, document, query_analysis)

    def _filter_by_relevance(self, chunks: List[Chunk]) -> List[Chunk]:
        """관련성 점수 기반 필터링"""
        filtered = []

        for chunk in chunks:
            relevance_score = chunk.metadata.get("relevance_score", 0)
            if relevance_score >= self.min_relevance_score:
                filtered.append(chunk)
            else:
                logger.debug(f"청크 필터링됨 - 낮은 관련성: {relevance_score:.2f}")

        return filtered

    def _optimize_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """청크 최적화 (병합/분할)"""
        if not chunks:
            return chunks

        optimized = []
        current_group = []
        current_size = 0

        # 관련성과 위치 기반으로 정렬
        sorted_chunks = sorted(chunks, key=lambda x: (x.start_idx, -x.metadata.get("relevance_score", 0)))

        for chunk in sorted_chunks:
            chunk_size = len(chunk.content)

            # 현재 그룹과 병합 가능한지 확인
            if current_group and self._can_merge(current_group[-1], chunk):
                if current_size + chunk_size <= self.chunk_size_limit * 1.5:
                    current_group.append(chunk)
                    current_size += chunk_size
                else:
                    # 그룹 완성 및 새 그룹 시작
                    merged = self._merge_chunk_group(current_group)
                    optimized.append(merged)
                    current_group = [chunk]
                    current_size = chunk_size
            else:
                # 이전 그룹 완성
                if current_group:
                    merged = self._merge_chunk_group(current_group)
                    optimized.append(merged)

                # 새 그룹 시작
                current_group = [chunk]
                current_size = chunk_size

        # 마지막 그룹 처리
        if current_group:
            merged = self._merge_chunk_group(current_group)
            optimized.append(merged)

        # 시퀀스 번호 재할당
        for i, chunk in enumerate(optimized):
            chunk.sequence_num = i

        return optimized

    def _can_merge(self, chunk1: Chunk, chunk2: Chunk) -> bool:
        """두 청크가 병합 가능한지 확인"""
        # 위치가 인접하거나 겹치는 경우
        if chunk1.end_idx >= chunk2.start_idx - 100:  # 100자 갭 허용
            # 관련성 점수가 비슷한 경우
            score1 = chunk1.metadata.get("relevance_score", 0)
            score2 = chunk2.metadata.get("relevance_score", 0)
            if abs(score1 - score2) < 0.3:
                return True

        return False

    def _merge_chunk_group(self, chunks: List[Chunk]) -> Chunk:
        """청크 그룹을 하나로 병합"""
        if len(chunks) == 1:
            return chunks[0]

        # 내용 병합
        merged_content = " ".join(chunk.content for chunk in chunks)

        # 메타데이터 병합
        avg_relevance = sum(chunk.metadata.get("relevance_score", 0) for chunk in chunks) / len(chunks)
        all_keywords = []
        all_topics = []

        for chunk in chunks:
            keywords = chunk.metadata.get("keywords", [])
            all_keywords.extend(keywords)

            topic = chunk.metadata.get("topic", "")
            if topic:
                all_topics.append(topic)

        # 중복 제거
        all_keywords = list(set(all_keywords))

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
        if self.language == Language.KOREAN:
            return """질문을 분석하여 핵심 정보를 추출하세요.
질문의 유형, 요구되는 정보, 핵심 키워드를 파악하세요."""
        else:
            return """Analyze the question to extract key information.
Identify question type, required information, and key keywords."""

    def _get_extraction_system_prompt(self) -> str:
        """청크 추출 시스템 프롬프트"""
        if self.language == Language.KOREAN:
            return """질문에 답하기 위해 필요한 문서 부분을 추출하세요.
각 부분의 관련성을 평가하고, 필요한 문맥을 포함하세요.
답변에 직접적으로 도움이 되는 정보를 우선시하세요."""
        else:
            return """Extract document parts necessary to answer the question.
Evaluate relevance of each part and include necessary context.
Prioritize information directly helpful for answering."""

    def _create_query_analysis_prompt(self, query: str) -> str:
        """쿼리 분석 프롬프트"""
        if self.language == Language.KOREAN:
            return f"""다음 질문을 분석하세요:

    질문: {query}

    응답 형식 (JSON): # <--- 수정
    {{
        "question_type": "정보요청/비교/설명/기타",
        "key_entities": ["주요_개체1", "주요_개체2"],
        "keywords": ["키워드1", "키워드2"],
        "required_info": ["필요정보1", "필요정보2"],
        "complexity": "simple/moderate/complex"
    }}"""
        else:
            return f"""Analyze the following question:

    Question: {query}

    Response format (JSON): # <--- 수정
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

        if self.language == Language.KOREAN:
            return f"""다음 질문에 답하기 위해 문서에서 관련 부분을 추출하세요.

    질문: {query}
    핵심 키워드: {keywords}

    문서:
    {content}

    응답 형식 (JSON): # <--- 수정
    {{
        "chunks": [
            {{
                "content": "관련 내용",
                "start_idx": 시작_위치,
                "end_idx": 끝_위치,
                "relevance_score": 0.0-1.0,
                "relevance_reason": "관련성 이유",
                "answer_contribution": "답변에 기여하는 정보",
                "keywords": ["포함된_키워드"]
            }}
        ],
        "coverage": "질문에 대한 답변 커버리지 (0-100%)"
    }}"""
        else:
            return f"""Extract relevant parts from the document to answer the question.

    Question: {query}
    Key keywords: {keywords}

    Document:
    {content}

    Response format (JSON): # <--- 수정
    {{
        "chunks": [
            {{
                "content": "relevant content",
                "start_idx": start_position,
                "end_idx": end_position,
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