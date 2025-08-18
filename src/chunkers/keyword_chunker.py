"""
문서 메타데이터 추출 기반 청킹 전략
Document Metadata Extraction-based Chunking Strategy
"""

import json
import time
from typing import List, Dict, Any
from loguru import logger

from src.config import Language, ChunkingStrategy, APIConfig
from src.data_structures import Document, Query, Chunk
from src.chunkers.base_chunker import BaseChunker
from openai import AsyncOpenAI


class KeywordChunker(BaseChunker):
    """문서 메타데이터 추출 기반 청킹 전략 구현"""

    def __init__(self, language: Language, chunk_size_limit: int = 1024):
        super().__init__(language, chunk_size_limit)
        self.strategy = ChunkingStrategy.KEYWORD
        config = APIConfig()
        self.client = AsyncOpenAI(api_key=config.openai_api_key)

    async def chunk_document(self, document: Document) -> List[Chunk]:
        """문서를 제목, 요약, 핵심 메타정보로 변환하여 청킹"""
        start_time = time.time()
        try:
            metadata = await self._extract_metadata(document.content)
            chunks = self._create_metadata_chunks(document, metadata)

            for chunk in chunks:
                chunk.doc_id = document.id

            processing_time = time.time() - start_time
            metrics = self.calculate_metrics(chunks, processing_time)
            logger.info(
                f"메타데이터 청킹 완료 - {metrics.num_chunks}개 청크 생성"
            )
            return chunks
        except Exception as e:
            logger.error(f"메타데이터 청킹 실패: {e}")
            return self._fallback_chunking(document)

    async def query_aware_chunk(self, document: Document, query: Query) -> List[Chunk]:
        """쿼리를 고려한 메타데이터 청킹"""
        try:
            metadata = await self._extract_query_focused_metadata(
                document.content,
                query.question
            )
            chunks = self._create_metadata_chunks(document, metadata)
            for chunk in chunks:
                chunk.doc_id = document.id
            return chunks
        except Exception as e:
            logger.error(f"쿼리 인식 메타데이터 청킹 실패: {e}")
            return await self.chunk_document(document)

    async def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """GPT-4를 사용하여 문서 메타데이터 추출"""
        try:
            prompt = self._create_extraction_prompt(content)
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"메타데이터 추출 실패: {e}")
            return self._create_fallback_metadata(content)

    async def _extract_query_focused_metadata(
            self,
            content: str,
            query: str
    ) -> Dict[str, Any]:
        """쿼리 중심 메타데이터 추출"""
        try:
            prompt = self._create_query_extraction_prompt(content, query)
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._get_query_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"쿼리 중심 메타데이터 추출 실패: {e}")
            return await self._extract_metadata(content)

    def _create_metadata_chunks(
            self,
            document: Document,
            metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """메타데이터를 청크로 변환"""
        chunks = []
        is_korean = self.language == Language.KOREAN

        title_prefix = "제목: " if is_korean else "Title: "
        summary_prefix = "요약: " if is_korean else "Summary: "
        key_point_prefix = "핵심포인트" if is_korean else "Key Point"

        # 1. 제목 청크
        if metadata.get("title"):
            title_chunk = self.create_chunk(
                content=f"{title_prefix}{metadata['title']}",
                document_id=document.id,
                start_idx=0,
                end_idx=len(metadata["title"]),
                sequence_num=0,
                metadata={"type": "title", "importance": 1.0}
            )
            chunks.append(title_chunk)

        # 2. 요약 청크
        if metadata.get("summary"):
            summary_chunk = self.create_chunk(
                content=f"{summary_prefix}{metadata['summary']}",
                document_id=document.id,
                start_idx=0,
                end_idx=len(metadata["summary"]),
                sequence_num=1,
                metadata={"type": "summary", "importance": 0.9}
            )
            chunks.append(summary_chunk)

        # 3. 핵심 포인트 청크
        if metadata.get("key_points"):
            for i, point in enumerate(metadata["key_points"]):
                point_chunk = self.create_chunk(
                    content=f"{key_point_prefix} {i + 1}: {point}",
                    document_id=document.id,
                    start_idx=0,
                    end_idx=len(point),
                    sequence_num=2 + i,
                    metadata={"type": "key_point", "importance": 0.8, "index": i}
                )
                chunks.append(point_chunk)

        # 4. 핵심 메타정보 청크
        if metadata.get("meta_info"):
            meta_content = self._format_meta_info(metadata["meta_info"])
            meta_chunk = self.create_chunk(
                content=meta_content,
                document_id=document.id,
                start_idx=0,
                end_idx=len(meta_content),
                sequence_num=len(chunks),
                metadata={"type": "meta_info", "importance": 0.7}
            )
            chunks.append(meta_chunk)

        return chunks

    def _format_meta_info(self, meta_info: Dict[str, Any]) -> str:
        """메타정보 포맷팅"""
        is_korean = self.language == Language.KOREAN

        header = "핵심 메타정보:\n" if is_korean else "Core Metadata:\n"
        entities_label = "- 주요 개체: " if is_korean else "- Main Entities: "
        topics_label = "- 주제: " if is_korean else "- Topics: "
        keywords_label = "- 키워드: " if is_korean else "- Keywords: "

        formatted = header
        if meta_info.get("entities"):
            formatted += f"{entities_label}{', '.join(meta_info['entities'])}\n"
        if meta_info.get("topics"):
            formatted += f"{topics_label}{', '.join(meta_info['topics'])}\n"
        if meta_info.get("keywords"):
            formatted += f"{keywords_label}{', '.join(meta_info['keywords'])}\n"

        return formatted

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트"""
        if self.language == Language.KOREAN:
            return """문서를 분석하여 다음 메타데이터를 추출하세요:
1. 제목: 문서의 핵심 주제를 나타내는 명확한 제목
2. 요약: 200-300자의 핵심 내용 요약
3. 핵심 포인트: 3-5개의 주요 논점
4. 메타정보: 주요 개체, 주제, 키워드"""
        else:
            return """Extract the following metadata from the document:
1. Title: Clear title representing the core topic
2. Summary: 200-300 character summary
3. Key Points: 3-5 main arguments
4. Meta Info: Entities, topics, keywords"""

    def _get_query_system_prompt(self) -> str:
        """쿼리 중심 시스템 프롬프트"""
        if self.language == Language.KOREAN:
            return """질문과 관련된 메타데이터를 우선적으로 추출하세요.
질문의 의도를 파악하고 관련 정보를 중점적으로 추출합니다."""
        else:
            return """Extract metadata prioritizing relevance to the question.
Understand query intent and focus on relevant information."""

    def _create_extraction_prompt(self, content: str) -> str:
        """메타데이터 추출 프롬프트"""
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        if self.language == Language.KOREAN:
            return f"""다음 문서에서 메타데이터를 추출하세요.

문서:
{content}

JSON 형식:
{{
    "title": "문서 제목",
    "summary": "핵심 요약",
    "key_points": ["포인트1", "포인트2", "포인트3"],
    "meta_info": {{
        "entities": ["주요 개체들"],
        "topics": ["주제1", "주제2"],
        "keywords": ["키워드1", "키워드2"]
    }}
}}"""
        else:
            return f"""Extract metadata from the document.

Document:
{content}

JSON format:
{{
    "title": "Document title",
    "summary": "Core summary",
    "key_points": ["point1", "point2", "point3"],
    "meta_info": {{
        "entities": ["main entities"],
        "topics": ["topic1", "topic2"],
        "keywords": ["keyword1", "keyword2"]
    }}
}}"""

    def _create_query_extraction_prompt(self, content: str, query: str) -> str:
        """쿼리 중심 추출 프롬프트"""
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        if self.language == Language.KOREAN:
            return f"""질문과 관련된 메타데이터를 추출하세요.

질문: {query}

문서:
{content}

JSON 형식 (질문과의 관련성 우선):
{{
    "title": "질문 관련 제목",
    "summary": "질문 중심 요약",
    "key_points": ["관련 포인트1", "관련 포인트2"],
    "meta_info": {{
        "entities": ["관련 개체들"],
        "topics": ["관련 주제들"],
        "keywords": ["관련 키워드들"]
    }}
}}"""
        else:
            return f"""Extract metadata related to the question.

Question: {query}

Document:
{content}

JSON format (prioritizing query relevance):
{{
    "title": "Query-related title",
    "summary": "Query-focused summary",
    "key_points": ["relevant point1", "relevant point2"],
    "meta_info": {{
        "entities": ["relevant entities"],
        "topics": ["relevant topics"],
        "keywords": ["relevant keywords"]
    }}
}}"""

    def _create_fallback_metadata(self, content: str) -> Dict[str, Any]:
        """폴백 메타데이터 생성"""
        sentences = self.split_by_sentences(content)
        is_korean = self.language == Language.KOREAN

        title = sentences[0][:100] if sentences else ("제목 없음" if is_korean else "No Title")

        return {
            "title": title,
            "summary": " ".join(sentences[:3])[:300] if len(sentences) > 3 else content[:300],
            "key_points": [sent[:100] for sent in sentences[:3]] if sentences else [],
            "meta_info": {
                "entities": [],
                "topics": ["일반"] if is_korean else ["General"],
                "keywords": []
            }
        }

    def _fallback_chunking(self, document: Document) -> List[Chunk]:
        """폴백 청킹"""
        logger.warning("메타데이터 추출 실패, 폴백 청킹 사용")
        fallback_metadata = self._create_fallback_metadata(document.content)
        chunks = self._create_metadata_chunks(document, fallback_metadata)
        for chunk in chunks:
            chunk.doc_id = document.id
        return chunks