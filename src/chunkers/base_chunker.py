"""
기본 청킹 추상 클래스
Base chunker abstract class
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time
import numpy as np
from loguru import logger

from src.config import Language, ChunkingStrategy
from src.data_structures import Document, Query, Chunk, ChunkingMetrics


class BaseChunker(ABC):
    """청킹 전략 추상 기본 클래스"""

    def __init__(self, language: Language, chunk_size_limit: int = 512):
        self.language = language
        self.chunk_size_limit = chunk_size_limit
        self.strategy = ChunkingStrategy.SEMANTIC  # 기본값, 하위 클래스에서 재정의

    @abstractmethod
    async def chunk_document(self, document: Document) -> List[Chunk]:
        """문서를 청크로 분할"""
        pass

    @abstractmethod
    async def query_aware_chunk(self, document: Document, query: Query) -> List[Chunk]:
        """쿼리 인식 청킹"""
        pass

    def validate_chunks(self, chunks: List[Chunk]) -> bool:
        """청크 유효성 검증"""
        if not chunks:
            logger.warning("빈 청크 리스트")
            return False

        for chunk in chunks:
            if len(chunk.content) == 0:
                logger.warning(f"빈 청크 발견: {chunk.id}")
                return False

            if len(chunk.content) > self.chunk_size_limit * 2:
                logger.warning(f"청크 크기 초과: {chunk.id} - {len(chunk.content)} 문자")
                return False

        return True

    def calculate_metrics(self, chunks: List[Chunk], processing_time: float) -> ChunkingMetrics:
        """청킹 메트릭 계산"""
        if not chunks:
            return ChunkingMetrics(
                num_chunks=0,
                avg_chunk_size=0,
                min_chunk_size=0,
                max_chunk_size=0,
                chunk_size_std=0,
                processing_time=processing_time
            )

        chunk_sizes = [len(chunk.content) for chunk in chunks]

        # 겹침 비율 계산
        total_chunk_length = sum(chunk_sizes)
        doc_length = chunks[0].end_idx if chunks else 0
        overlap_ratio = max(0, (total_chunk_length - doc_length) / doc_length) if doc_length > 0 else 0

        return ChunkingMetrics(
            num_chunks=len(chunks),
            avg_chunk_size=np.mean(chunk_sizes),
            min_chunk_size=min(chunk_sizes),
            max_chunk_size=max(chunk_sizes),
            chunk_size_std=np.std(chunk_sizes),
            overlap_ratio=overlap_ratio,
            processing_time=processing_time
        )

    def create_chunk(
            self,
            content: str,
            document_id: str,
            start_idx: int,
            end_idx: int,
            sequence_num: int,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """청크 객체 생성 헬퍼"""
        chunk_id = f"{document_id}_{self.strategy.value}_{sequence_num}"

        chunk = Chunk(
            id=chunk_id,
            content=content,
            document_id=document_id,
            start_idx=start_idx,
            end_idx=end_idx,
            strategy=self.strategy,
            sequence_num=sequence_num,
            metadata=metadata or {}
        )

        # doc_id 속성 추가
        chunk.doc_id = document_id

        return chunk

    def split_by_sentences(self, text: str) -> List[str]:
        """문장 단위로 텍스트 분할"""
        if self.language == Language.KOREAN:
            # 한국어 문장 분할 (간단한 규칙 기반)
            import re
            sentences = re.split(r'[.!?]+\s*', text)
            return [s.strip() for s in sentences if s.strip()]
        else:
            # 영어 문장 분할 (NLTK 사용)
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)

    async def chunk_documents_batch(self, documents: List[Document]) -> Dict[str, List[Chunk]]:
        """문서 배치 청킹"""
        results = {}

        for doc in documents:
            start_time = time.time()
            chunks = await self.chunk_document(doc)
            processing_time = time.time() - start_time

            if self.validate_chunks(chunks):
                results[doc.id] = chunks
                metrics = self.calculate_metrics(chunks, processing_time)
                logger.info(f"문서 {doc.id} 청킹 완료: {metrics.num_chunks}개 청크 생성")
            else:
                logger.error(f"문서 {doc.id} 청킹 실패")
                results[doc.id] = []

        return results

    def merge_small_chunks(self, chunks: List[Chunk], min_size: int = 100) -> List[Chunk]:
        """작은 청크들을 병합"""
        if not chunks:
            return chunks

        merged_chunks = []
        current_chunk = None

        for chunk in sorted(chunks, key=lambda x: x.sequence_num):
            if current_chunk is None:
                current_chunk = chunk
            elif len(current_chunk.content) < min_size and \
                 len(current_chunk.content) + len(chunk.content) < self.chunk_size_limit:  # <-- 크기 제한 체크 추가

                # 현재 청크가 작으면 다음 청크와 병합
                current_chunk = self.create_chunk(
                    content=current_chunk.content + " " + chunk.content,
                    document_id=current_chunk.document_id,
                    start_idx=current_chunk.start_idx,
                    end_idx=chunk.end_idx,
                    sequence_num=current_chunk.sequence_num,
                    metadata={
                        **current_chunk.metadata,
                        "merged": True,
                        "original_chunks": 2
                    }
                )
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            merged_chunks.append(current_chunk)

        return merged_chunks