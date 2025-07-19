"""
기본 생성 추상 클래스
Base generator abstract class
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.data_structures import Query, Chunk, RAGResponse


class BaseGenerator(ABC):
    """생성 추상 기본 클래스"""

    @abstractmethod
    async def generate_response(
            self,
            query: Query,
            chunks: List[Chunk],
            **kwargs
    ) -> RAGResponse:
        """쿼리와 청크를 기반으로 응답 생성"""
        pass

    def format_context(self, chunks: List[Chunk]) -> str:
        """청크들을 컨텍스트 문자열로 포맷팅"""
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[참조 {i + 1}]\n{chunk.content}")

        return "\n\n".join(context_parts)