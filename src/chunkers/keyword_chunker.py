"""
키워드 중요도 기반 청킹 전략 (수정된 버전)
Keyword importance-based chunking strategy (Fixed version)
"""

import json
import time
import difflib
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
        config = APIConfig()
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.context_window = 2

    async def chunk_document(self, document: Document) -> List[Chunk]:
        """키워드 기반 청킹"""
        start_time = time.time()

        try:
            # 1. 핵심 문장 추출
            key_sentences = await self._extract_key_sentences(document.content)

            # 2. 핵심 문장 주변 문맥 포함하여 청크 생성
            chunks = self._create_chunks_with_context(document, key_sentences)

            logger.debug(f"청크 생성 완료, doc_id 추가 전: {len(chunks)}개")

            # doc_id 추가 후 확인
            logger.debug(f"모든 청크에 doc_id 추가 완료: {all(hasattr(c, 'doc_id') for c in chunks)}")

            # 3. 메트릭 계산 및 로깅
            processing_time = time.time() - start_time
            metrics = self.calculate_metrics(chunks, processing_time)
            logger.info(f"키워드 기반 청킹 완료 - {metrics.num_chunks}개 청크, 평균 크기: {metrics.avg_chunk_size:.1f}")

            return chunks

        except Exception as e:
            logger.error(f"키워드 기반 청킹 실패: {e}")
            fallback_chunks = self._fallback_chunking(document)
            # fallback 청크에도 doc_id 속성 추가
            for chunk in fallback_chunks:
                if not hasattr(chunk, 'doc_id'):
                    chunk.doc_id = document.id
            return fallback_chunks

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

            # *** 수정된 부분: doc_id 속성 추가 ***
            for chunk in chunks:
                if not hasattr(chunk, 'doc_id'):
                    chunk.doc_id = document.id

            return chunks

        except Exception as e:
            logger.error(f"쿼리 인식 키워드 청킹 실패: {e}")
            return await self.chunk_document(document)

    async def _extract_key_sentences(self, content: str) -> List[Dict[str, Any]]:
        """GPT-4를 사용하여 핵심 문장 추출"""
        try:
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

        except Exception as e:
            logger.error(f"핵심 문장 추출 실패: {e}")
            # *** 수정된 부분: Fallback으로 휴리스틱 기반 추출 ***
            return self._heuristic_key_sentences(content)

    def _heuristic_key_sentences(self, content: str) -> List[Dict[str, Any]]:
        """API 실패 시 사용할 휴리스틱 기반 핵심 문장 추출"""
        sentences = self.split_by_sentences(content)

        key_sentences = []
        for i, sentence in enumerate(sentences):
            sentence_clean = sentence.strip()

            # 충분한 정보가 있고 너무 짧지 않은 문장 선택
            if len(sentence_clean) > 50 and len(sentence_clean) < 300:
                key_sentences.append({
                    "sentence": sentence_clean,
                    "importance_score": 0.7,  # 기본 점수
                    "keywords": sentence_clean.split()[:5],  # 처음 5개 단어
                    "topic": "general"
                })

                if len(key_sentences) >= 5:  # 최대 5개
                    break

        # 최소 1개는 보장
        if not key_sentences and sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 20:
                key_sentences.append({
                    "sentence": first_sentence,
                    "importance_score": 0.5,
                    "keywords": first_sentence.split()[:3],
                    "topic": "fallback"
                })

        return key_sentences

    async def _extract_query_keywords(self, query: str) -> List[str]:
        """쿼리에서 핵심 키워드 추출"""
        try:
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

        except Exception as e:
            logger.error(f"쿼리 키워드 추출 실패: {e}")
            # Fallback: 쿼리를 단어로 분할
            return query.split()[:5]

    async def _extract_key_sentences_with_query(
        self,
        content: str,
        query_keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """쿼리 키워드를 고려하여 핵심 문장 추출"""
        try:
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

        except Exception as e:
            logger.error(f"쿼리 인식 핵심 문장 추출 실패: {e}")
            # Fallback: 일반 핵심 문장 추출 사용
            return await self._extract_key_sentences(content)

    def _create_chunks_with_context(
        self,
        document: Document,
        key_sentences: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """핵심 문장과 주변 문맥을 포함한 청크 생성"""

        # *** 수정된 부분: 빈 key_sentences 처리 ***
        if not key_sentences:
            logger.warning("핵심 문장이 없어 fallback 청킹 사용")
            return self._create_fallback_chunks_internal(document)

        # 전체 문서를 문장으로 분할
        all_sentences = self.split_by_sentences(document.content)
        sentence_positions = self._calculate_sentence_positions(all_sentences, document.content)

        chunks = []
        used_sentences = set()

        for key_data in key_sentences:
            key_sentence = key_data.get("sentence", "")
            importance = key_data.get("importance_score", 0.5)

            # 핵심 문장의 인덱스 찾기 (개선된 매칭)
            key_idx = self._find_sentence_index_improved(key_sentence, all_sentences)
            if key_idx == -1:
                logger.debug(f"문장 매칭 실패: {key_sentence[:50]}...")
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

        # *** 수정된 부분: 청크가 없으면 최소한의 청크 생성 ***
        if not chunks:
            logger.warning("키워드 기반 청크 생성 실패, 최소한의 청크 생성")
            return self._create_fallback_chunks_internal(document)

        # 청크를 위치 순으로 정렬
        chunks.sort(key=lambda x: x.start_idx)

        # 시퀀스 번호 재할당
        for i, chunk in enumerate(chunks):
            chunk.sequence_num = i

        return chunks

    def _create_fallback_chunks_internal(self, document: Document) -> List[Chunk]:
        """내부적으로 사용할 fallback 청킹 (doc_id 미포함)"""
        chunks = []
        text = document.content
        chunk_size = self.chunk_size_limit

        for i in range(0, len(text), chunk_size // 2):
            end = min(i + chunk_size, len(text))
            chunk_content = text[i:end]

            if len(chunk_content.strip()) > 50:
                chunk = self.create_chunk(
                    content=chunk_content,
                    document_id=document.id,
                    start_idx=i,
                    end_idx=end,
                    sequence_num=len(chunks),
                    metadata={"fallback": True, "type": "internal"}
                )
                chunks.append(chunk)

        return chunks

    def _find_sentence_index_improved(self, target: str, sentences: List[str]) -> int:
        """개선된 문장 매칭 (유사도 기반)"""
        target_clean = target.strip().lower()

        # 1. 정확한 매칭 시도
        for i, sentence in enumerate(sentences):
            if sentence.strip().lower() == target_clean:
                return i

        # 2. 유사도 기반 매칭
        best_match_idx = -1
        best_similarity = 0.0

        for i, sentence in enumerate(sentences):
            similarity = difflib.SequenceMatcher(None, target_clean, sentence.strip().lower()).ratio()
            if similarity > best_similarity and similarity > 0.8:  # 80% 이상 유사
                best_similarity = similarity
                best_match_idx = i

        # 3. 포함 관계 확인
        if best_match_idx == -1:
            for i, sentence in enumerate(sentences):
                sentence_clean = sentence.strip().lower()
                if target_clean in sentence_clean or sentence_clean in target_clean:
                    if len(target_clean) > 20 or len(sentence_clean) > 20:  # 너무 짧은 문장 제외
                        return i

        return best_match_idx

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
        """핵심 문장 추출 프롬프트 (주석 제거됨)"""
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        if self.language == Language.KOREAN:
            return f"""다음 문서에서 핵심 문장들을 추출하세요.
각 문장의 중요도와 키워드를 함께 제공하세요.

문서:
{content}

응답 형식 (JSON):
{{
    "key_sentences": [
        {{
            "sentence": "핵심 문장",
            "importance_score": 0.8,
            "keywords": ["키워드1", "키워드2"],
            "topic": "주제"
        }}
    ],
    "total_sentences": 5
}}"""
        else:
            return f"""Extract key sentences from the following document.
Provide importance score and keywords for each sentence.

Document:
{content}

Response format (JSON):
{{
    "key_sentences": [
        {{
            "sentence": "key sentence",
            "importance_score": 0.8,
            "keywords": ["keyword1", "keyword2"],
            "topic": "topic"
        }}
    ],
    "total_sentences": 5
}}"""

    def _create_keyword_extraction_prompt(self, query: str) -> str:
        """키워드 추출 프롬프트 (주석 제거됨)"""
        if self.language == Language.KOREAN:
            return f"""다음 질문에서 핵심 키워드를 추출하세요.

질문: {query}

응답 형식 (JSON):
{{
    "keywords": ["키워드1", "키워드2", "키워드3"]
}}"""
        else:
            return f"""Extract key keywords from the following question.

Question: {query}

Response format (JSON):
{{
    "keywords": ["keyword1", "keyword2", "keyword3"]
}}"""

    def _create_query_aware_key_sentence_prompt(
            self,
            content: str,
            keywords: List[str]
    ) -> str:
        """쿼리 인식 핵심 문장 추출 프롬프트 (주석 제거됨)"""
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        keywords_str = ", ".join(keywords)

        if self.language == Language.KOREAN:
            return f"""다음 키워드와 관련된 핵심 문장들을 추출하세요.

키워드: {keywords_str}

문서:
{content}

응답 형식 (JSON):
{{
    "key_sentences": [
        {{
            "sentence": "핵심 문장",
            "importance_score": 0.8,
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

Response format (JSON):
{{
    "key_sentences": [
        {{
            "sentence": "key sentence",
            "importance_score": 0.8,
            "keywords": ["matched_keywords"],
            "relevance_reason": "relevance reason"
        }}
    ]
}}"""

    def _fallback_chunking(self, document: Document) -> List[Chunk]:
        """폴백 청킹 방법 (doc_id 미포함)"""
        logger.warning("키워드 기반 청킹 실패, 폴백 방법 사용")

        chunks = []
        text = document.content
        chunk_size = self.chunk_size_limit

        for i in range(0, len(text), chunk_size // 2):
            end = min(i + chunk_size, len(text))
            chunk_content = text[i:end]

            if len(chunk_content.strip()) > 50:
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