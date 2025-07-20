"""
고정 크기 청킹 전략 (Baseline)
Fixed-size chunking strategy (Baseline)
"""

import time
from typing import List, Optional, Dict, Any
from loguru import logger

from src.config import Language, ChunkingStrategy
from src.data_structures import Document, Query, Chunk
from .base_chunker import BaseChunker


class FixedSizeChunker(BaseChunker):
    """고정 크기 청킹 전략 (Baseline) 구현"""

    def __init__(self, language: Language, chunk_size_limit: int = 512, overlap_ratio: float = 0.1):
        super().__init__(language, chunk_size_limit)
        self.strategy = ChunkingStrategy.FIXED_SIZE
        self.overlap_ratio = overlap_ratio  # 겹침 비율 (기본 10%)
        self.overlap_size = int(chunk_size_limit * overlap_ratio)

    async def chunk_document(self, document: Document) -> List[Chunk]:
        """고정 크기 기반 청킹"""
        start_time = time.time()

        try:
            chunks = self._create_fixed_size_chunks(document)

            # 메트릭 계산 및 로깅
            processing_time = time.time() - start_time
            metrics = self.calculate_metrics(chunks, processing_time)
            logger.info(
                f"고정 크기 청킹 완료 - {metrics.num_chunks}개 청크, "
                f"평균 크기: {metrics.avg_chunk_size:.1f}, "
                f"겹침 비율: {metrics.overlap_ratio:.2f}"
            )

            return chunks

        except Exception as e:
            logger.error(f"고정 크기 청킹 실패: {e}")
            return []

    async def query_aware_chunk(self, document: Document, query: Query) -> List[Chunk]:
        """쿼리 인식 청킹 (고정 크기는 쿼리와 무관하므로 일반 청킹과 동일)"""
        logger.info("고정 크기 청킹은 쿼리와 무관하므로 일반 청킹 수행")
        return await self.chunk_document(document)

    def _create_fixed_size_chunks(self, document: Document) -> List[Chunk]:
        """고정 크기 청크 생성"""
        text = document.content
        chunks = []

        # 문서가 너무 짧은 경우 그대로 하나의 청크로 처리
        if len(text) <= self.chunk_size_limit:
            chunk = self.create_chunk(
                content=text,
                document_id=document.id,
                start_idx=0,
                end_idx=len(text),
                sequence_num=0,
                metadata={
                    "chunk_method": "single_chunk",
                    "overlap_size": 0
                }
            )
            return [chunk]

        # 청크 크기에서 겹침 크기를 뺀 값이 실제 스텝 크기
        step_size = self.chunk_size_limit - self.overlap_size

        for i in range(0, len(text), step_size):
            start_idx = i
            end_idx = min(i + self.chunk_size_limit, len(text))

            # 마지막 청크가 너무 작으면 이전 청크와 병합
            if end_idx == len(text) and i > 0:
                remaining_size = len(text) - i
                if remaining_size < self.chunk_size_limit * 0.3:  # 30% 미만이면 병합
                    # 이전 청크와 병합
                    if chunks:
                        prev_chunk = chunks[-1]
                        merged_content = text[prev_chunk.start_idx:end_idx]

                        # 이전 청크를 업데이트
                        chunks[-1] = self.create_chunk(
                            content=merged_content,
                            document_id=document.id,
                            start_idx=prev_chunk.start_idx,
                            end_idx=end_idx,
                            sequence_num=prev_chunk.sequence_num,
                            metadata={
                                "chunk_method": "merged_last",
                                "overlap_size": self.overlap_size,
                                "merged": True
                            }
                        )
                    break

            chunk_content = text[start_idx:end_idx]

            # 빈 청크는 건너뛰기
            if not chunk_content.strip():
                continue

            # 문장 경계에서 자르기 시도 (선택사항)
            if self._should_adjust_boundary(chunk_content, end_idx, text):
                adjusted_end = self._find_sentence_boundary(text, start_idx, end_idx)
                if adjusted_end > start_idx:
                    end_idx = adjusted_end
                    chunk_content = text[start_idx:end_idx]

            chunk = self.create_chunk(
                content=chunk_content,
                document_id=document.id,
                start_idx=start_idx,
                end_idx=end_idx,
                sequence_num=len(chunks),
                metadata={
                    "chunk_method": "fixed_size",
                    "overlap_size": self.overlap_size if i > 0 else 0,
                    "step_size": step_size,
                    "boundary_adjusted": end_idx != min(i + self.chunk_size_limit, len(text))
                }
            )

            chunks.append(chunk)

        return chunks

    def _should_adjust_boundary(self, chunk_content: str, end_idx: int, full_text: str) -> bool:
        """문장 경계 조정이 필요한지 확인"""
        # 마지막 청크이거나 이미 문장이 완전한 경우 조정 불필요
        if end_idx >= len(full_text):
            return False

        # 청크가 문장 중간에서 끝나는 경우 조정 고려
        if not chunk_content.rstrip().endswith(('.', '!', '?', '다', '요', '음')):
            return True

        return False

    def _find_sentence_boundary(self, text: str, start_idx: int, max_end_idx: int) -> int:
        """문장 경계 찾기"""
        # 한국어와 영어 문장 끝 패턴
        if self.language == Language.KOREAN:
            sentence_endings = ['.', '!', '?', '다.', '요.', '음.', '다!', '요!', '다?', '요?']
        else:
            sentence_endings = ['.', '!', '?']

        # 역방향으로 문장 끝 찾기
        search_start = min(max_end_idx, len(text))
        search_limit = max(start_idx + self.chunk_size_limit // 2, search_start - 100)  # 너무 많이 줄이지 않도록

        for i in range(search_start - 1, search_limit - 1, -1):
            for ending in sentence_endings:
                if text[i:i + len(ending)] == ending:
                    # 문장 끝 다음 위치 반환
                    return i + len(ending)

        # 문장 경계를 찾지 못한 경우 원래 위치 반환
        return max_end_idx

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """청크 통계 정보 반환"""
        if not chunks:
            return {}

        chunk_sizes = [len(chunk.content) for chunk in chunks]
        overlap_sizes = [chunk.metadata.get("overlap_size", 0) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "avg_overlap_size": sum(overlap_sizes) / len(overlap_sizes),
            "total_text_length": sum(chunk_sizes),
            "effective_coverage": sum(chunk_sizes) - sum(overlap_sizes),
            "overlap_ratio": sum(overlap_sizes) / sum(chunk_sizes) if sum(chunk_sizes) > 0 else 0,
            "boundary_adjusted_chunks": sum(1 for chunk in chunks
                                            if chunk.metadata.get("boundary_adjusted", False)),
            "merged_chunks": sum(1 for chunk in chunks
                                 if chunk.metadata.get("merged", False))
        }