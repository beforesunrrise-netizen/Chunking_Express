"""
임베딩 기반 의미 청크 방식 (LangChain의 SemanticChunker 유사)

이 모듈은 외부 라이브러리 없이 의미 단위로 문장을 묶어 청크로 분할하는 간단한 구현입니다.
- 문장을 3개씩 묶고
- 의미가 유사하면 병합합니다.
- 임베딩 대신 difflib.SequenceMatcher로 텍스트 유사도를 계산합니다.
- LangChain이나 OpenAI 의존성 없이 가볍게 사용할 수 있습니다.
"""

import difflib
from typing import List, Sequence

from src.config import Language, ChunkingStrategy
from src.data_structures import Document, Query, Chunk
from .base_chunker import BaseChunker


class Text_Similarity(BaseChunker):
    def __init__(
        self,
        language: Language,
        chunk_size_limit: int = 512,
        group_size: int = 3,
        similarity_threshold: float = 0.75,
    ) -> None:
        super().__init__(language, chunk_size_limit)

        # ✅ 명확히 전략 지정
        self.strategy = ChunkingStrategy.TEXT_SIMILARITY

        self.group_size = max(1, group_size)
        self.similarity_threshold = similarity_threshold


    def _group_sentences(self, sentences: Sequence[str]) -> List[List[str]]:
        """문장 리스트를 group_size 개수만큼 묶어서 그룹화"""
        groups: List[List[str]] = []
        idx = 0
        total = len(sentences)
        while idx < total:
            groups.append(list(sentences[idx: idx + self.group_size]))
            idx += self.group_size
        return groups

    def _compute_similarity(self, a: str, b: str) -> float:
        """두 문자열 사이의 유사도 계산 (0.0 ~ 1.0)"""
        return difflib.SequenceMatcher(None, a, b).ratio()

    async def chunk_document(self, document: Document) -> List[Chunk]:
        """
        문서를 문장 단위로 나눈 뒤 의미가 유사한 그룹끼리 병합하여 청크 생성

        1. 문장을 자름
        2. group_size 개수씩 묶음
        3. 그룹별 텍스트를 합치고
        4. 유사도가 일정 기준 이상이면 병합
        5. 병합된 텍스트에서 위치를 계산하여 Chunk 생성
        """
        # 문장 분할
        sentences = self.split_by_sentences(document.content or "")
        groups = self._group_sentences(sentences)

        # 각 그룹을 텍스트로 병합
        group_texts = [" ".join(group).strip() for group in groups if any(s.strip() for s in group)]

        merged_texts: List[str] = []
        if not group_texts:
            group_texts = []

        # 인접 그룹 유사하면 병합
        current = ""
        for text in group_texts:
            if not current:
                current = text
                continue
            similarity = self._compute_similarity(current, text)
            if similarity >= self.similarity_threshold:
                current = f"{current} {text}"  # 병합
            else:
                merged_texts.append(current)  # 병합 종료, 다음 그룹 시작
                current = text
        if current:
            merged_texts.append(current)

        # 병합된 텍스트를 기준으로 Chunk 객체 생성
        chunks: List[Chunk] = []
        offset = 0
        seq = 0
        content = document.content or ""
        for text in merged_texts:
            start_idx = content.find(text, offset)
            if start_idx == -1:
                start_idx = content.find(text)
            end_idx = start_idx + len(text)
            offset = end_idx

            if len(text.strip()) <= 10:
                continue  # 너무 짧으면 무시

            chunk = self.create_chunk(
                content=text,
                document_id=document.id,
                start_idx=start_idx,
                end_idx=end_idx,
                sequence_num=seq,
                metadata={
                    "group_size": self.group_size,
                    "similarity_threshold": self.similarity_threshold
                },
            )
            chunks.append(chunk)
            seq += 1
        return chunks

    async def query_aware_chunk(self, document: Document, query: Query) -> List[Chunk]:
        """
        쿼리를 고려하지 않고 의미 기반 청크 분할 수행
        """
        return await self.chunk_document(document)
