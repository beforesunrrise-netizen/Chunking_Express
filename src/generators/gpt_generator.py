"""
GPT 기반 응답 생성기
GPT-based response generator
"""

import time
from typing import List, Dict
import openai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Language, config
from src.data_structures import Query, Chunk, RAGResponse
from .base_generator import BaseGenerator
from openai import AsyncOpenAI
from src.config import APIConfig

class GPTGenerator(BaseGenerator):
    """GPT 기반 응답 생성기"""

    def __init__(self, language: Language, model: str = None):
        self.language = language
        self.model = model or config.model.gpt_model
        config_data = APIConfig()
        self.client = openai.AsyncOpenAI(api_key=config_data.openai_api_key)
        self.temperature = config.model.temperature
        self.max_tokens = config.model.max_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def generate_response(
            self,
            query: Query,
            chunks: List[Chunk],
            **kwargs
    ) -> RAGResponse:
        """쿼리와 청크를 기반으로 응답 생성"""
        start_time = time.time()

        # 컨텍스트 준비
        context = self.format_context(chunks)

        # 프롬프트 생성
        messages = self._create_messages(query.question, context)

        try:
            # GPT 호출
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # 응답 추출
            generated_text = response.choices[0].message.content

            # 신뢰도 계산 (간단한 휴리스틱)
            confidence = self._calculate_confidence(
                generated_text,
                chunks,
                response.choices[0].finish_reason
            )

            generation_time = time.time() - start_time

            # 응답 객체 생성
            rag_response = RAGResponse(
                query=query,
                query_id=query.id,
                response=generated_text,
                chunks_used=chunks,
                confidence=confidence,
                strategy=chunks[0].strategy if chunks else None,
                generation_time=generation_time,
                model_used=self.model,
                metadata={
                    "usage": response.usage.dict() if response.usage else {},
                    "finish_reason": response.choices[0].finish_reason
                }
            )

            logger.info(
                f"응답 생성 완료 - "
                f"길이: {len(generated_text)}, "
                f"시간: {generation_time:.2f}초, "
                f"신뢰도: {confidence:.2f}"
            )

            return rag_response

        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            # 오류 응답 반환
            return self._create_error_response(query, chunks, str(e))

    def _create_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        """GPT 메시지 생성"""
        system_prompt = self._get_system_prompt()
        user_prompt = self._create_user_prompt(question, context)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 생성"""
        if self.language == Language.KOREAN:
            return """당신은 주어진 참조 문서를 바탕으로 정확하고 도움이 되는 답변을 제공하는 AI 어시스턴트입니다.

다음 지침을 따르세요:
1. 제공된 참조 문서의 정보만을 사용하여 답변하세요.
2. 참조 문서에 없는 정보는 추가하지 마세요.
3. 확실하지 않은 경우 "제공된 정보에서는 찾을 수 없습니다"라고 답하세요.
4. 답변은 명확하고 구조화된 형태로 제공하세요.
5. 가능한 경우 관련 참조 번호를 언급하세요."""
        else:
            return """You are an AI assistant that provides accurate and helpful answers based on given reference documents.

Follow these guidelines:
1. Use only information from the provided reference documents to answer.
2. Do not add information not present in the reference documents.
3. If uncertain, respond with "Cannot find in the provided information."
4. Provide clear and structured answers.
5. Mention relevant reference numbers when possible."""

    def _create_user_prompt(self, question: str, context: str) -> str:
        """사용자 프롬프트 생성"""
        if self.language == Language.KOREAN:
            return f"""참조 문서:
{context}

질문: {question}

위의 참조 문서를 바탕으로 질문에 답변해주세요."""
        else:
            return f"""Reference Documents:
{context}

Question: {question}

Please answer the question based on the reference documents above."""

    def _calculate_confidence(
            self,
            response: str,
            chunks: List[Chunk],
            finish_reason: str
    ) -> float:
        """응답 신뢰도 계산"""
        confidence = 1.0

        # 완성도 체크
        if finish_reason != "stop":
            confidence *= 0.8

        # 응답 길이 체크
        if len(response) < 20:
            confidence *= 0.7
        elif len(response) > 1000:
            confidence *= 0.9

        # 청크 사용률 체크
        if not chunks:
            confidence *= 0.5
        elif len(chunks) == 1:
            confidence *= 0.8

        # 청크 관련성 점수 평균 (쿼리 인식 청킹의 경우)
        relevance_scores = [
            chunk.metadata.get("relevance_score", 1.0)
            for chunk in chunks
        ]
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            confidence *= avg_relevance

        return min(max(confidence, 0.0), 1.0)

    def _create_error_response(
            self,
            query: Query,
            chunks: List[Chunk],
            error_message: str
    ) -> RAGResponse:
        """오류 응답 생성"""
        if self.language == Language.KOREAN:
            error_text = f"죄송합니다. 응답을 생성하는 중 오류가 발생했습니다: {error_message}"
        else:
            error_text = f"Sorry, an error occurred while generating the response: {error_message}"

        return RAGResponse(
            query=query,
            query_id=query.id,
            response=error_text,
            chunks_used=chunks,
            confidence=0.0,
            strategy=chunks[0].strategy if chunks else None,
            generation_time=0.0,
            model_used=self.model,
            metadata={"error": error_message}
        )