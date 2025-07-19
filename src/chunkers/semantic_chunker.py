import json
import time
from typing import List, Dict, Any, Optional

import openai
from loguru import logger

from src.config import Language, ChunkingStrategy, APIConfig
from src.data_structures import Document, Query, Chunk
from .base_chunker import BaseChunker
from openai import AsyncOpenAI

class SemanticChunker(BaseChunker):
    """
    GPT-4o 모델을 활용하여 문서의 의미 구조를 파악하고,
    이를 기반으로 문서를 영어(English)로 청킹하는 클래스입니다.
    """

    # --- 수정된 부분: __init__에서 language 매개변수 제거 ---
    def __init__(self, chunk_size_limit: int = 512):
        # 부모 클래스에는 기본 언어로 ENGLISH를 고정하여 전달합니다.
        super().__init__(Language.ENGLISH, chunk_size_limit)
        self.strategy = ChunkingStrategy.SEMANTIC
        config = APIConfig()
        self.client = AsyncOpenAI(api_key=config.openai_api_key)


    async def chunk_document(self, document: Document) -> List[Chunk]:
        """문서를 의미 단위로 청킹합니다."""
        start_time = time.time()
        logger.info(f"문서 '{document.id}'에 대한 의미 기반 청킹을 시작합니다.")
        system_prompt = self._get_system_prompt()
        user_prompt = self._create_semantic_prompt(document.content)
        response_data = await self._get_gpt_response(system_prompt, user_prompt)
        if not response_data:
            logger.warning(f"GPT 응답 실패. 문서 '{document.id}'에 대한 폴백 청킹을 실행합니다.")
            return self._fallback_chunking(document)
        chunks = self._create_chunks_from_response(response_data, document)
        processing_time = time.time() - start_time
        num_chunks = len(chunks)
        avg_chunk_size = sum(len(c.content) for c in chunks) / num_chunks if num_chunks > 0 else 0
        logger.info(
            f"의미 기반 청킹 완료. "
            f"소요 시간: {processing_time:.2f}초, "
            f"생성된 청크 수: {num_chunks}개, "
            f"평균 청크 크기: {avg_chunk_size:.1f}자"
        )
        return chunks

    async def query_aware_chunk(self, document: Document, query: Query) -> List[Chunk]:
        """쿼리(질문)를 고려하여 문서를 의미 단위로 청킹합니다."""
        logger.info(f"문서 '{document.id}'에 대한 쿼리 인식 청킹을 시작합니다. (쿼리: '{query.question}')")
        system_prompt = self._get_query_aware_system_prompt()
        user_prompt = self._create_query_aware_prompt(document.content, query.question)
        response_data = await self._get_gpt_response(system_prompt, user_prompt)
        if not response_data:
            logger.warning(f"쿼리 인식 청킹 실패. 문서 '{document.id}'에 대한 폴백 청킹을 실행합니다.")
            return self._fallback_chunking(document)
        return self._create_chunks_from_response(response_data, document)

    # --- Core Logic (변경 없음) ---

    async def _get_gpt_response(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """GPT 모델에 요청을 보내고 JSON 응답을 파싱합니다."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            logger.error("GPT 응답 내용이 비어 있습니다.")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"GPT 응답 JSON 파싱 실패: {e}")
            logger.error(f"파싱 실패한 내용: {content}")
            return None
        except Exception as e:
            logger.error(f"GPT API 호출 중 오류 발생: {e}")
            return None

    def _create_chunks_from_response(self, response_data: Dict[str, Any], document: Document) -> List[Chunk]:
        """API 응답(JSON)을 Chunk 객체 리스트로 변환합니다."""
        chunks = []
        for i, chunk_data in enumerate(response_data.get("chunks", [])):
            content = chunk_data.get("content", "")
            if not content:
                continue

            # --- ID 생성 및 전달 로직 제거 ---

            start_idx = chunk_data.get("start_idx", 0)
            end_idx = chunk_data.get("end_idx", start_idx + len(content))

            metadata = {
                "topic": chunk_data.get("topic"),
                "keywords": chunk_data.get("keywords"),
                "relevance_score": chunk_data.get("relevance_score"),
                "reason": chunk_data.get("reason")
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}

            # create_chunk 호출 시 id를 전달하지 않으면, 부모 클래스(BaseChunker)가 알아서 만들어줍니다.
            chunk = self.create_chunk(
                content=content,
                document_id=document.id,
                start_idx=start_idx,
                end_idx=end_idx,
                sequence_num=i,
                metadata=metadata
            )
            chunks.append(chunk)
        return chunks

    def _fallback_chunking(self, document: Document) -> List[Chunk]:
        """API 실패 시 대체 작동할 단순 문장 분할 기반 청킹입니다."""
        sentences = self.split_by_sentences(document.content)
        chunks = []
        current_chunk_content = ""
        current_start_idx = 0
        for sentence in sentences:
            if len(current_chunk_content) + len(sentence) > self.chunk_size_limit and current_chunk_content:
                chunk = self.create_chunk(
                    content=current_chunk_content.strip(),
                    document_id=document.id,
                    start_idx=current_start_idx,
                    end_idx=current_start_idx + len(current_chunk_content),
                    sequence_num=len(chunks),
                    metadata={"fallback": True, "reason": "API call failed"}
                )
                chunks.append(chunk)
                current_start_idx += len(current_chunk_content)
                current_chunk_content = sentence
            else:
                current_chunk_content += (" " + sentence) if current_chunk_content else sentence
        if current_chunk_content:
            chunk = self.create_chunk(
                content=current_chunk_content.strip(),
                document_id=document.id,
                start_idx=current_start_idx,
                end_idx=len(document.content),
                sequence_num=len(chunks),
                metadata={"fallback": True, "reason": "API call failed"}
            )
            chunks.append(chunk)
        return chunks

    # --- 프롬프트 함수 수정: 영어 프롬프트만 직접 반환 ---

    def _get_system_prompt(self) -> str:
        """의미 기반 청킹을 위한 시스템 프롬프트를 생성합니다."""
        return """You are an expert at dividing documents into semantically related units.
Identify the document's topics, structure, and context to create natural semantic divisions.
Each chunk should be independently understandable, and important information should not be separated.
Always return results in the specified JSON format."""

    def _get_query_aware_system_prompt(self) -> str:
        """쿼리 인식 청킹을 위한 시스템 프롬프트를 생성합니다."""
        return """You are an expert at dividing documents in a way that is optimized for answering a specific question.
Understand the user's query and divide the document focusing on the most critical information needed to answer it.
Each chunk must clearly indicate its relevance to the question.
Always return results in the specified JSON format."""

    def _create_semantic_prompt(self, content: str) -> str:
        """의미 기반 청킹을 위한 사용자 프롬프트를 생성합니다."""
        content = content[:10000]
        return f"""Divide the following document into semantically related units.
Each chunk should be limited to approximately {self.chunk_size_limit} characters, but avoid breaking semantic meaning.

Document:
{content}

Response format (JSON):
{{
    "chunks": [
        {{
            "content": "chunk content",
            "start_idx": "start position in document (integer)",
            "end_idx": "end position in document (integer)",
            "topic": "core topic or summary of the chunk",
            "keywords": ["keyword1", "keyword2"]
        }}
    ]
}}"""

    def _create_query_aware_prompt(self, content: str, question: str) -> str:
        """쿼리 인식 청킹을 위한 사용자 프롬프트를 생성합니다."""
        content = content[:10000]
        return f"""Optimally divide the document below to best answer the following question.
Structure the chunks around information directly relevant to the question and evaluate the relevance of each chunk.

Question: {question}

Document:
{content}

Response format (JSON):
{{
    "chunks": [
        {{
            "content": "relevant content",
            "start_idx": "start position in document (integer)",
            "end_idx": "end position in document (integer)",
            "relevance_score": "a relevance score between 0.0 and 1.0 (float)",
            "reason": "a brief explanation of why this chunk is relevant"
        }}
    ]
}}"""