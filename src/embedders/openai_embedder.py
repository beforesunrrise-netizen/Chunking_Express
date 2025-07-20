"""
OpenAI 임베딩 구현 (파일 기반 영구 캐싱 및 청킹별 저장 기능 추가)
OpenAI embedder implementation (with file-based persistent caching and chunk storage)
"""

import hashlib
import json
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


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 임베딩 구현 (파일 캐싱 기능 포함)"""

    def __init__(self, language: Language, model: str = None):
        self.language = language
        self.model = model or config.model.embedding_model
        config_data = APIConfig()
        self.client = openai.AsyncOpenAI(api_key=config_data.openai_api_key)
        self.dimension = config.model.embedding_dimension
        self.batch_size = 100  # OpenAI API 배치 제한

        # --- 파일 기반 캐싱 로직 추가 ---
        self.cache_dir = Path("./cache")
        self.cache_path = self.cache_dir / f"embedding_cache_{self.model.replace('/', '_')}.json"
        self.embedding_cache = self._load_cache()
        logger.info(f"임베딩 캐시 로드 완료: {self.cache_path} ({len(self.embedding_cache)}개 항목)")
        # ---------------------------------

    def _load_cache(self) -> Dict[str, List[float]]:
        """로컬 JSON 파일에서 캐시를 로드합니다."""
        self.cache_dir.mkdir(exist_ok=True)
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"캐시 파일 손상됨: {self.cache_path}. 새 캐시를 생성합니다.")
                    return {}
        return {}

    def _save_cache(self):
        """캐시를 로컬 JSON 파일에 저장합니다."""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.embedding_cache, f)  # indent 제거로 파일 크기 최적화
        logger.debug(f"임베딩 캐시 저장 완료. 총 {len(self.embedding_cache)}개 항목.")

    def _get_cache_key(self, text: str) -> str:
        """텍스트 내용을 기반으로 고유한 SHA-256 해시 키를 생성합니다."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def embed_text(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩 (캐싱 적용)"""
        key = self._get_cache_key(text)
        if key in self.embedding_cache:
            return np.array(self.embedding_cache[key])

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            self.embedding_cache[key] = embedding
            self._save_cache()  # 변경 시마다 저장
            return np.array(embedding)

        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return np.zeros(self.dimension)

    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """여러 텍스트 배치 임베딩 (캐싱 적용)"""
        if not texts:
            return []

        embeddings = [None] * len(texts)
        texts_to_fetch_online = []
        indices_to_fetch_online = []

        # 1. 캐시 확인
        for i, text in enumerate(texts):
            key = self._get_cache_key(text)
            if key in self.embedding_cache:
                embeddings[i] = np.array(self.embedding_cache[key])
            else:
                # 빈 문자열이나 공백만 있는 경우 API 호출 방지
                if text and not text.isspace():
                    texts_to_fetch_online.append(text)
                    indices_to_fetch_online.append(i)
                else:
                    embeddings[i] = np.zeros(self.dimension)

        # 2. 캐시에 없는 텍스트에 대해서만 API 호출 (배치 처리)
        if texts_to_fetch_online:
            logger.info(f"{len(texts_to_fetch_online)}개의 새로운 텍스트에 대해 임베딩 API 호출 중...")

            made_change = False
            for i in range(0, len(texts_to_fetch_online), self.batch_size):
                batch_texts = texts_to_fetch_online[i:i + self.batch_size]
                batch_indices = indices_to_fetch_online[i:i + self.batch_size]

                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=batch_texts
                    )

                    # 3. 결과 저장 및 캐시 업데이트
                    for j, embedding_data in enumerate(response.data):
                        original_index = batch_indices[j]
                        text_to_cache = batch_texts[j]
                        key_to_cache = self._get_cache_key(text_to_cache)

                        embedding_list = embedding_data.embedding
                        embeddings[original_index] = np.array(embedding_list)
                        self.embedding_cache[key_to_cache] = embedding_list
                        made_change = True

                except Exception as e:
                    logger.error(f"배치 임베딩 실패 (배치 인덱스 {i}): {e}")
                    for original_index in batch_indices:
                        embeddings[original_index] = np.zeros(self.dimension)

            # API 호출로 변경이 있었던 경우에만 파일 저장
            if made_change:
                self._save_cache()

        return embeddings

    async def embed_chunks(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """청크 리스트 임베딩"""
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embed_texts(texts)

        # 청크 객체에 임베딩 저장
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is not None:
                chunk.embedding = embedding.tolist()

        return embeddings

    def clear_cache(self):
        """캐시 파일과 메모리 캐시를 모두 초기화합니다."""
        self.embedding_cache.clear()
        if self.cache_path.exists():
            self.cache_path.unlink()
        logger.info(f"임베딩 캐시가 초기화되었습니다: {self.cache_path}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return {
            "cache_file": str(self.cache_path),
            "cache_size": len(self.embedding_cache)
        }


class OpenAIEmbedderWithStorage(OpenAIEmbedder):
    """청킹별 임베딩 저장 기능이 추가된 OpenAI Embedder"""

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
            logger.info(f"청킹 임베딩 저장소 초기화 완료: {storage_path}")

    async def embed_and_store_chunks(
        self,
        chunks: List[Chunk],
        chunk_type: str,
        document_id: str,
        additional_metadata: Optional[Dict] = None,
        force_recompute: bool = False
    ) -> List[np.ndarray]:
        """청크를 임베딩하고 저장"""

        # 저장 기능이 비활성화된 경우
        if not self.enable_storage:
            return await self.embed_chunks(chunks)

        # 1. 이미 저장된 임베딩이 있는지 확인 (force_recompute가 False인 경우)
        if not force_recompute:
            existing_data = self.storage.load_chunk_embeddings(document_id, chunk_type)
            if existing_data and chunk_type in existing_data.get("chunk_types", {}):
                existing_chunks = existing_data["chunk_types"][chunk_type]
                if len(existing_chunks.get("chunks", [])) == len(chunks):
                    logger.info(f"기존 임베딩 사용: {document_id} ({chunk_type})")
                    return [np.array(emb) for emb in existing_chunks["embeddings"]]

        # 2. 새로운 임베딩 생성
        logger.info(f"새 임베딩 생성 중: {document_id} ({chunk_type})")
        embeddings = await self.embed_chunks(chunks)

        # 3. 저장
        if self.enable_storage:
            metadata = additional_metadata or {}
            metadata["model"] = self.model

            success = self.storage.save_chunk_embeddings(
                chunks=chunks,
                embeddings=embeddings,
                chunk_type=chunk_type,
                document_id=document_id,
                language=self.language,
                additional_metadata=metadata
            )

            if not success:
                logger.warning("임베딩 저장 실패, 하지만 임베딩은 반환됩니다.")

        return embeddings

    async def search_similar_chunks(
        self,
        query: str,
        document_id: Optional[str] = None,
        chunk_type: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """쿼리와 유사한 청크 검색"""
        if not self.enable_storage:
            logger.warning("저장소가 비활성화되어 있어 검색할 수 없습니다.")
            return []

        # 쿼리 임베딩 생성
        query_embedding = await self.embed_text(query)

        # 유사 청크 검색
        return self.storage.search_similar_chunks(
            query_embedding=query_embedding,
            document_id=document_id,
            chunk_type=chunk_type,
            top_k=top_k,
            threshold=threshold
        )

    def get_storage_stats(self) -> Dict[str, Any]:
        """저장소 통계 반환"""
        stats = {
            "cache_stats": self.get_cache_stats(),
            "storage_enabled": self.enable_storage
        }

        if self.enable_storage:
            stats["storage_stats"] = self.storage.get_statistics()

        return stats

    def clear_document_embeddings(self, document_id: str) -> bool:
        """특정 문서의 모든 임베딩 삭제"""
        if not self.enable_storage:
            logger.warning("저장소가 비활성화되어 있습니다.")
            return False

        return self.storage.clear_document(document_id)