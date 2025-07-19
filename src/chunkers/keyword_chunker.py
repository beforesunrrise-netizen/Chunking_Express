"""
키워드 중요도 기반 청킹 전략
Keyword importance-based chunking strategy
"""

import json
import time
from typing import List, Dict, Any, Tuple
import openai
from loguru import logger

from src.config import Language, ChunkingStrategy
from src.data_structures import Document, Query, Chunk
from .base_chunker import BaseChunker
from openai import AsyncOpenAI
from src.config import APIConfig

class KeywordChunker(BaseChunker):
    """키워드 중요도 기반 청킹 전략 구현"""

    def __init__(self, language: Language, chunk_size_limit: int = 512):
        super().__init__(language, chunk_size_limit)
        self.strategy = ChunkingStrategy.KEYWORD
        config = APIConfig()  # 환경변수에서 가져온 config 사용
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.context_window = 2  # 핵심 문장 주변 문장 수

    async def chunk_document(self, document: Document) -> List[Chunk]:
        """키워드 기반 청킹"""
        start_time = time.time()

        try:
            # 1. 핵심 문장 추출
            key_sentences = await self._extract_key_sentences(document.content)

            # 2. 핵심 문장 주변 문맥 포함하여 청크 생성
            chunks = self._create_chunks_with_context(document, key_sentences)

            # 3. 메트릭 계산 및 로깅
            processing_time = time.time() - start_time
            metrics = self.calculate_metrics(chunks, processing_time)
            logger.info(f"키워드 기반 청킹 완료 - {metrics.num_chunks}개 청크, 평균 크기: {metrics.avg_chunk_size:.1f}")

            return chunks

        except Exception as e:
            logger.error(f"키워드 기반 청킹 실패: {e}")
            return self._fallback_chunking(document)

    async def query_aware_chunk(self, document: Document, query: Query) -> List[Chunk]:
        """쿼리를 고려한 키워드 청킹"""
        try:
            # 쿼리 관련 키워드 추출
            query_keywords = await self._extract_query_keywords(query.question)

            # 쿼리 키워드를 고려한 핵심 문장 추출
            key_sentences = await self._extract_key_sentences_with_query(
                document.content,
                query_keywords
            )

            # 청크 생성
            chunks = self._create_chunks_with_context(document, key_sentences)

            return chunks

        except Exception as e:
            logger.error(f"쿼리 인식 키워드 청킹 실패: {e}")
            return await self.chunk_document(document)

    async def _extract_key_sentences(self, content: str) -> List[Dict[str, Any]]:
        """GPT-4를 사용하여 핵심 문장 추출"""
        prompt = self._create_key_sentence_prompt(content)

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": self._get_key_sentence_system_prompt()
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
        return result.get("key_sentences", [])

    async def _extract_query_keywords(self, query: str) -> List[str]:
        """쿼리에서 핵심 키워드 추출"""
        prompt = self._create_keyword_extraction_prompt(query)

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract key keywords from the query. Return as JSON."
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
        return result.get("keywords", [])

    async def _extract_key_sentences_with_query(
        self,
        content: str,
        query_keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """쿼리 키워드를 고려하여 핵심 문장 추출"""
        prompt = self._create_query_aware_key_sentence_prompt(content, query_keywords)

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": self._get_query_aware_key_sentence_system_prompt()
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
        return result.get("key_sentences", [])

    def _create_chunks_with_context(
        self,
        document: Document,
        key_sentences: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """핵심 문장과 주변 문맥을 포함한 청크 생성"""
        # 전체 문서를 문장으로 분할
        all_sentences = self.split_by_sentences(document.content)
        sentence_positions = self._calculate_sentence_positions(all_sentences, document.content)

        chunks = []
        used_sentences = set()  # 중복 방지

        for key_data in key_sentences:
            key_sentence = key_data.get("sentence", "")
            importance = key_data.get("importance_score", 0.5)

            # 핵심 문장의 인덱스 찾기
            key_idx = self._find_sentence_index(key_sentence, all_sentences)
            if key_idx == -1:
                continue

            # 이미 사용된 문장인지 확인
            if key_idx in used_sentences:
                continue

            # 문맥 포함 범위 결정
            start_idx = max(0, key_idx - self.context_window)
            end_idx = min(len(all_sentences), key_idx + self.context_window + 1)

            # 청크 내용 구성
            chunk_sentences = all_sentences[start_idx:end_idx]
            chunk_content = " ".join(chunk_sentences)

            # 사용된 문장 표시
            for i in range(start_idx, end_idx):
                used_sentences.add(i)

            # 위치 정보 계산
            if start_idx < len(sentence_positions) and end_idx - 1 < len(sentence_positions):
                start_pos = sentence_positions[start_idx][0]
                end_pos = sentence_positions[end_idx - 1][1]
            else:
                start_pos = 0
                end_pos = len(chunk_content)

            # 청크 생성
            chunk = self.create_chunk(
                content=chunk_content,
                document_id=document.id,
                start_idx=start_pos,
                end_idx=end_pos,
                sequence_num=len(chunks),
                metadata={
                    "key_sentence": key_sentence,
                    "importance_score": importance,
                    "context_window": self.context_window,
                    "keywords": key_data.get("keywords", [])
                }
            )
            chunks.append(chunk)

        # 청크를 위치 순으로 정렬
        chunks.sort(key=lambda x: x.start_idx)

        # 시퀀스 번호 재할당
        for i, chunk in enumerate(chunks):
            chunk.sequence_num = i

        return chunks

    def _calculate_sentence_positions(
        self,
        sentences: List[str],
        full_text: str
    ) -> List[Tuple[int, int]]:
        """각 문장의 시작과 끝 위치 계산"""
        positions = []
        current_pos = 0

        for sentence in sentences:
            start_pos = full_text.find(sentence, current_pos)
            if start_pos != -1:
                end_pos = start_pos + len(sentence)
                positions.append((start_pos, end_pos))
                current_pos = end_pos
            else:
                # 찾을 수 없는 경우 추정
                positions.append((current_pos, current_pos + len(sentence)))
                current_pos += len(sentence)

        return positions

    def _find_sentence_index(self, target: str, sentences: List[str]) -> int:
        """목표 문장의 인덱스 찾기"""
        target_clean = target.strip().lower()

        for i, sentence in enumerate(sentences):
            if sentence.strip().lower() == target_clean:
                return i
            # 부분 일치도 고려
            if target_clean in sentence.strip().lower() or sentence.strip().lower() in target_clean:
                return i

        return -1

    def _get_key_sentence_system_prompt(self) -> str:
        """핵심 문장 추출 시스템 프롬프트"""
        if self.language == Language.KOREAN:
            return """문서에서 가장 중요하고 정보가 풍부한 핵심 문장들을 추출하세요.
각 문장의 중요도를 0-1 점수로 평가하고, 관련 키워드를 추출하세요.
전체 문서를 대표할 수 있는 다양한 주제의 문장을 선택하세요."""
        else:
            return """Extract the most important and information-rich key sentences from the document.
Evaluate each sentence's importance on a 0-1 scale and extract relevant keywords.
Select sentences representing diverse topics that can represent the entire document."""

    def _get_query_aware_key_sentence_system_prompt(self) -> str:
        """쿼리 인식 핵심 문장 추출 시스템 프롬프트"""
        if self.language == Language.KOREAN:
            return """주어진 키워드와 관련된 핵심 문장들을 추출하세요.
키워드와의 관련성과 정보의 중요도를 모두 고려하여 평가하세요."""
        else:
            return """Extract key sentences related to the given keywords.
Consider both relevance to keywords and information importance in evaluation."""

    def _create_key_sentence_prompt(self, content: str) -> str:
        """핵심 문장 추출 프롬프트"""
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        if self.language == Language.KOREAN:
            return f"""다음 문서에서 핵심 문장들을 추출하세요.
    각 문장의 중요도와 키워드를 함께 제공하세요.

    문서:
    {content}

    응답 형식 (JSON): # <--- 수정
    {{
        "key_sentences": [
            {{
                "sentence": "핵심 문장",
                "importance_score": 0.0-1.0,
                "keywords": ["키워드1", "키워드2"],
                "topic": "주제"
            }}
        ],
        "total_sentences": 추출된_문장_수
    }}"""
        else:
            return f"""Extract key sentences from the following document.
    Provide importance score and keywords for each sentence.

    Document:
    {content}

    Response format (JSON): # <--- 수정
    {{
        "key_sentences": [
            {{
                "sentence": "key sentence",
                "importance_score": 0.0-1.0,
                "keywords": ["keyword1", "keyword2"],
                "topic": "topic"
            }}
        ],
        "total_sentences": number_of_extracted_sentences
    }}"""

    def _create_keyword_extraction_prompt(self, query: str) -> str:
        """키워드 추출 프롬프트"""
        if self.language == Language.KOREAN:
            return f"""다음 질문에서 핵심 키워드를 추출하세요.

    질문: {query}

    응답 형식 (JSON): # <--- 수정
    {{
        "keywords": ["키워드1", "키워드2", ...]
    }}"""
        else:
            return f"""Extract key keywords from the following question.

    Question: {query}

    Response format (JSON): # <--- 수정
    {{
        "keywords": ["keyword1", "keyword2", ...]
    }}"""

    def _create_query_aware_key_sentence_prompt(
            self,
            content: str,
            keywords: List[str]
    ) -> str:
        """쿼리 인식 핵심 문장 추출 프롬프트"""
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        keywords_str = ", ".join(keywords)

        if self.language == Language.KOREAN:
            return f"""다음 키워드와 관련된 핵심 문장들을 추출하세요.

    키워드: {keywords_str}

    문서:
    {content}

    응답 형식 (JSON): # <--- 수정
    {{
        "key_sentences": [
            {{
                "sentence": "핵심 문장",
                "importance_score": 0.0-1.0,
                "keywords": ["매칭된_키워드"],
                "relevance_reason": "관련성 이유"
            }}
        ]
    }}"""
        else:
            return f"""Extract key sentences related to the following keywords.

    Keywords: {keywords_str}

    Document:
    {content}

    Response format (JSON): # <--- 수정
    {{
        "key_sentences": [
            {{
                "sentence": "key sentence",
                "importance_score": 0.0-1.0,
                "keywords": ["matched_keywords"],
                "relevance_reason": "relevance reason"
            }}
        ]
    }}"""

    def _fallback_chunking(self, document: Document) -> List[Chunk]:
        """폴백 청킹 방법"""
        logger.warning("키워드 기반 청킹 실패, 폴백 방법 사용")

        # 고정 크기 청킹
        chunks = []
        text = document.content
        chunk_size = self.chunk_size_limit

        for i in range(0, len(text), chunk_size // 2):  # 50% 오버랩
            end = min(i + chunk_size, len(text))
            chunk_content = text[i:end]

            if len(chunk_content.strip()) > 50:  # 최소 길이 체크
                chunk = self.create_chunk(
                    content=chunk_content,
                    document_id=document.id,
                    start_idx=i,
                    end_idx=end,
                    sequence_num=len(chunks),
                    metadata={"fallback": True}
                )
                chunks.append(chunk)

        return chunks