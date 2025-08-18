"""
OpenAI 임베딩 구현 (성능 최적화 버전)
- 대용량 배치 처리 최적화
- 비동기 병렬 처리 강화
- 메모리 효율성 개선
- Rate limiting 대응 강화
"""

import hashlib
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import openai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Language, config
from src.data_structures import Chunk
from .base_embedder import BaseEmbedder
from src.config import APIConfig
from src.storage.chunk_storage import ChunkEmbeddingStorage
import re
import random

def _extract_retry_after_seconds(err_msg: str, default_seconds: float = 0.25) -> float:
    """
    OpenAI 오류 메시지에 포함된 'Try again in XXms'를 파싱해 초 단위로 반환.
    없으면 기본값(지수백오프와 함께) 사용.
    """
    if not err_msg:
        return default_seconds
    m = re.search(r"Try again in\s+(\d+)ms", err_msg)
    if m:
        # 약간의 지터 추가
        return max(int(m.group(1)) / 1000.0, default_seconds) + random.uniform(0.05, 0.15)
    return default_seconds

class OptimizedOpenAIEmbedder(BaseEmbedder):
    """성능 최적화된 OpenAI 임베딩 구현"""

    def __init__(self, language: Language, model: str = None):
        self.language = language
        self.model = model or config.model.embedding_model
        config_data = APIConfig()
        self.client = openai.AsyncOpenAI(api_key=config_data.openai_api_key)
        self.dimension = config.model.embedding_dimension

        # 최적화된 배치 설정
        self.batch_size = 100  # OpenAI API 최대 배치 크기
        self.max_concurrent_batches = 1  # 동시 배치 요청 수 제한
        self.rate_limit_delay = 0.25  # 배치 간 지연시간 (초)

        # 메모리 효율적 캐싱
        self.cache_dir = Path("./cache")
        self.cache_path = self.cache_dir / f"embedding_cache_{self.model.replace('/', '_')}.json"
        self.embedding_cache = self._load_cache()

        # 배치 처리용 세마포어
        self.batch_semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        logger.info(f"최적화된 임베딩 엔진 초기화 완료: {len(self.embedding_cache)}개 캐시 항목")

    def _load_cache(self) -> Dict[str, List[float]]:
        """메모리 효율적 캐시 로드"""
        self.cache_dir.mkdir(exist_ok=True)
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                logger.info(f"캐시 로드 성공: {len(cache_data)}개 항목")
                return cache_data
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"캐시 파일 오류: {e}. 새 캐시 생성")
        return {}

    def _save_cache(self):
        """비동기 캐시 저장 (블로킹 방지)"""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.embedding_cache, f, separators=(',', ':'))  # 압축 저장
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")

    def _get_cache_key(self, text: str) -> str:
        """빠른 해시 키 생성"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()  # SHA256보다 빠른 MD5

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _batch_embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        async with self.batch_semaphore:  # 동시 요청 수 제한
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float"
                )
                # 고정 대기(배치 간) — 소폭 상향
                await asyncio.sleep(self.rate_limit_delay)
                return [data.embedding for data in response.data]

            except openai.RateLimitError as e:
                # 동적 대기: 오류 메시지에 포함된 ms를 사용
                wait_s = _extract_retry_after_seconds(str(e), default_seconds=0.5)
                logger.warning(f"Rate limit 도달, {wait_s:.2f}s 대기 후 재시도: {e}")
                await asyncio.sleep(wait_s)
                raise

            except Exception as e:
                logger.error(f"배치 임베딩 실패: {e}")
                raise

    async def embed_texts_optimized(self, texts: List[str]) -> List[np.ndarray]:
        """대용량 텍스트 배치 최적화 임베딩"""
        if not texts:
            return []

        logger.info(f"임베딩 시작: {len(texts)}개 텍스트")

        # 1단계: 캐시 분류 (빠른 처리)
        embeddings = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []
        cache_hits = 0

        for i, text in enumerate(texts):
            if not text or text.isspace():
                embeddings[i] = np.zeros(self.dimension)
                continue

            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                embeddings[i] = np.array(self.embedding_cache[cache_key])
                cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if cache_hits > 0:
            cache_rate = (cache_hits / len(texts)) * 100
            logger.info(f"캐시 적중률: {cache_rate:.1f}% ({cache_hits}/{len(texts)})")

        # 2단계: API 호출이 필요한 텍스트만 배치 처리
        if uncached_texts:
            logger.info(f"새로운 임베딩 생성: {len(uncached_texts)}개")

            # 대용량 배치를 여러 개의 작은 배치로 분할
            batch_tasks = []
            for i in range(0, len(uncached_texts), self.batch_size):
                batch_texts = uncached_texts[i:i + self.batch_size]
                batch_indices = uncached_indices[i:i + self.batch_size]

                batch_tasks.append(
                    self._process_batch(batch_texts, batch_indices, embeddings)
                )

            # 배치들을 병렬로 처리 (세마포어로 동시성 제어)
            await asyncio.gather(*batch_tasks, return_exceptions=True)

            # 캐시 업데이트 (한 번에)
            self._save_cache()

        logger.success(f"임베딩 완료: {len(texts)}개 텍스트")
        return embeddings

    async def _process_batch(
        self,
        batch_texts: List[str],
        batch_indices: List[int],
        embeddings: List
    ):
        """개별 배치 처리"""
        try:
            batch_embeddings = await self._batch_embed_with_retry(batch_texts)

            # 결과 저장 및 캐시 업데이트
            for text, embedding, idx in zip(batch_texts, batch_embeddings, batch_indices):
                embeddings[idx] = np.array(embedding)
                cache_key = self._get_cache_key(text)
                self.embedding_cache[cache_key] = embedding

        except Exception as e:
            logger.error(f"배치 처리 실패: {e}")
            # 실패한 배치는 0벡터로 채움
            for idx in batch_indices:
                embeddings[idx] = np.zeros(self.dimension)

    async def embed_chunks(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """청크 리스트 최적화 임베딩"""
        if not chunks:
            return []

        # 텍스트 추출
        texts = []
        valid_indices = []

        for i, chunk in enumerate(chunks):
            if hasattr(chunk, 'content') and chunk.content:
                texts.append(chunk.content)
                valid_indices.append(i)

        if not texts:
            logger.warning("임베딩할 유효한 텍스트가 없습니다")
            return [np.zeros(self.dimension) for _ in chunks]

        # 최적화된 임베딩 생성
        embeddings = await self.embed_texts_optimized(texts)

        # 청크 객체에 임베딩 할당
        chunk_embeddings = [np.zeros(self.dimension) for _ in chunks]

        for embedding, chunk_idx in zip(embeddings, valid_indices):
            if embedding is not None:
                chunks[chunk_idx].embedding = embedding.tolist()
                chunk_embeddings[chunk_idx] = embedding

        return chunk_embeddings

    # 기존 메서드들 유지
    async def embed_text(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩 (하위 호환성)"""
        embeddings = await self.embed_texts_optimized([text])
        return embeddings[0] if embeddings else np.zeros(self.dimension)

    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """텍스트 리스트 임베딩 (하위 호환성)"""
        return await self.embed_texts_optimized(texts)

    def clear_cache(self):
        """캐시 초기화"""
        self.embedding_cache.clear()
        if self.cache_path.exists():
            self.cache_path.unlink()
        logger.info("임베딩 캐시가 초기화되었습니다")

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        cache_size_mb = 0
        if self.cache_path.exists():
            cache_size_mb = self.cache_path.stat().st_size / (1024 * 1024)

        return {
            "cache_file": str(self.cache_path),
            "cache_entries": len(self.embedding_cache),
            "cache_size_mb": round(cache_size_mb, 2),
            "batch_size": self.batch_size,
            "max_concurrent_batches": self.max_concurrent_batches
        }


class OptimizedOpenAIEmbedderWithStorage(OptimizedOpenAIEmbedder):
    """스토리지 기능이 추가된 최적화 임베더"""

    def __init__(
        self,
        language: Language,
        model: str = None,
        storage_path: str = "./src/data",
        enable_storage: bool = True
    ):
        super().__init__(language, model)
        self.enable_storage = enable_storage
        if self.enable_storage:
            self.storage = ChunkEmbeddingStorage(storage_path)
            logger.info(f"청킹 임베딩 저장소 초기화: {storage_path}")

    async def embed_and_store_chunks(
        self,
        chunks: List[Chunk],
        chunk_type: str,
        document_id: str,
        additional_metadata: Optional[Dict] = None,
        force_recompute: bool = False
    ) -> List[np.ndarray]:
        """청크 임베딩 및 저장 (최적화 버전)"""

        if not self.enable_storage:
            return await self.embed_chunks(chunks)

        # 1. 기존 데이터 확인 (force_recompute가 False인 경우)
        if not force_recompute:
            try:
                existing_data = self.storage.load_chunk_embeddings(document_id, chunk_type)
                if existing_data and chunk_type in existing_data.get("chunk_types", {}):
                    existing_info = existing_data["chunk_types"][chunk_type]
                    stored_chunks = existing_info.get("chunks", [])
                    stored_embeddings = existing_info.get("embeddings", [])

                    if len(stored_chunks) == len(chunks) and len(stored_embeddings) == len(chunks):
                        logger.info(f"기존 임베딩 재사용: {document_id} ({chunk_type}) - {len(chunks)}개")
                        # 청크 객체에 임베딩 할당
                        for chunk, embedding in zip(chunks, stored_embeddings):
                            chunk.embedding = embedding
                        return [np.array(emb) for emb in stored_embeddings]
            except Exception as e:
                logger.warning(f"기존 데이터 로드 실패: {e}")

        # 2. 새로운 임베딩 생성 (최적화된 배치 처리)
        logger.info(f"새 임베딩 생성: {document_id} ({chunk_type}) - {len(chunks)}개")
        embeddings = await self.embed_chunks(chunks)

        # 3. 저장
        if self.enable_storage and embeddings:
            try:
                metadata = additional_metadata or {}
                metadata.update({
                    "model": self.model,
                    "chunk_count": len(chunks),
                    "embedding_dimension": self.dimension
                })

                success = self.storage.save_chunk_embeddings(
                    chunks=chunks,
                    embeddings=embeddings,
                    chunk_type=chunk_type,
                    document_id=document_id,
                    language=self.language,
                    additional_metadata=metadata
                )

                if success:
                    logger.debug(f"임베딩 저장 성공: {document_id} ({chunk_type})")
                else:
                    logger.warning(f"임베딩 저장 실패: {document_id} ({chunk_type})")

            except Exception as e:
                logger.error(f"저장 중 오류: {e}")

        return embeddings

    # 기존 메서드들 유지...
    async def search_similar_chunks(
        self,
        query: str,
        document_id: Optional[str] = None,
        chunk_type: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """유사 청크 검색"""
        if not self.enable_storage:
            logger.warning("저장소가 비활성화되어 검색할 수 없습니다")
            return []

        query_embedding = await self.embed_text(query)
        return self.storage.search_similar_chunks(
            query_embedding=query_embedding,
            document_id=document_id,
            chunk_type=chunk_type,
            top_k=top_k,
            threshold=threshold
        )

    def get_storage_stats(self) -> Dict[str, Any]:
        """저장소 통계"""
        stats = {
            "cache_stats": self.get_cache_stats(),
            "storage_enabled": self.enable_storage
        }
        if self.enable_storage:
            stats["storage_stats"] = self.storage.get_statistics()
        return stats

    def clear_document_embeddings(self, document_id: str) -> bool:
        """문서 임베딩 삭제"""
        if not self.enable_storage:
            return False
        return self.storage.clear_document(document_id)


# 하위 호환성을 위한 별칭
OpenAIEmbedder = OptimizedOpenAIEmbedder
OpenAIEmbedderWithStorage = OptimizedOpenAIEmbedderWithStorage