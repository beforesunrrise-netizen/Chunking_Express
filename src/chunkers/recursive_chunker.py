from typing import List

from src.config import Language, ChunkingStrategy
from src.data_structures import Document, Query, Chunk
from .base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    고정된 길이로 텍스트를 나누되, 청크 간에 일부 내용을 겹치게 처리하는 클래스입니다.
    LangChain의 RecursiveCharacterTextSplitter와 유사한 방식입니다.
    """

    def __init__(self, language: Language, chunk_size_limit: int = 512, chunk_overlap: int = 50) -> None:
        super().__init__(language, chunk_size_limit)

        # ✅ 명확하게 전략 지정
        self.strategy = ChunkingStrategy.RECURSIVE

        # 겹치는 길이는 0 이상, 청크 크기의 절반 이하로 제한
        self.chunk_overlap = max(0, min(chunk_overlap, chunk_size_limit // 2))

    async def chunk_document(self, document: Document) -> List[Chunk]:
        content = document.content or ""
        chunk_size = self.chunk_size_limit
        overlap = self.chunk_overlap
        step = max(1, chunk_size - overlap)

        chunks: List[Chunk] = []
        seq = 0
        pos = 0
        length = len(content)

        while pos < length:
            end = min(pos + chunk_size, length)
            chunk_text = content[pos:end]
            if len(chunk_text.strip()) > 10:
                chunk = self.create_chunk(
                    content=chunk_text,
                    document_id=document.id,
                    start_idx=pos,
                    end_idx=end,
                    sequence_num=seq,
                    metadata={"chunk_overlap": overlap}
                )
                chunks.append(chunk)
                seq += 1
            pos += step

        return chunks

    async def query_aware_chunk(self, document: Document, query: Query) -> List[Chunk]:
        return await self.chunk_document(document)
