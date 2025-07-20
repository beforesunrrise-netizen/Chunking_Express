"""
청킹별 임베딩 저장 및 관리 시스템
Chunk embedding storage and management system
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from loguru import logger

from src.data_structures import Chunk
from src.config import Language, ChunkingStrategy


class ChunkEmbeddingStorage:
    """청킹된 데이터의 임베딩을 효율적으로 저장/관리하는 클래스"""

    def __init__(self, base_path: str = "./src/data"):
        self.base_path = Path(base_path)
        self.embeddings_dir = self.base_path / "embeddings"
        self.metadata_dir = self.base_path / "metadata"

        # 디렉토리 생성
        self._create_directories()

        # 메타데이터 캐시
        self.metadata_cache = self._load_metadata()

    def _create_directories(self):
        """필요한 디렉토리 구조 생성"""
        # ChunkingStrategy enum 값들을 사용하여 디렉토리 생성
        chunk_types = [strategy.value for strategy in ChunkingStrategy]

        dirs = [
            self.embeddings_dir / "by_document",
            self.embeddings_dir / "by_language" / "ko",
            self.embeddings_dir / "by_language" / "en",
            self.metadata_dir
        ]

        # 각 청킹 타입별 디렉토리 생성
        for chunk_type in chunk_types:
            dirs.append(self.embeddings_dir / "by_chunk_type" / chunk_type)

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"임베딩 저장 디렉토리 구조 생성 완료: {self.base_path}")

    def _load_metadata(self) -> Dict[str, Any]:
        """메타데이터 로드"""
        metadata_file = self.metadata_dir / "embedding_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"메타데이터 로드 실패: {e}")
                return self._get_default_metadata()
        return self._get_default_metadata()

    def _get_default_metadata(self) -> Dict[str, Any]:
        """기본 메타데이터 구조"""
        return {
            "documents": {},
            "statistics": {
                "total_chunks": 0,
                "total_embeddings": 0,
                "total_documents": 0,
                "by_chunk_type": {},
                "by_language": {"ko": 0, "en": 0}
            },
            "last_updated": datetime.now().isoformat()
        }

    def _save_metadata(self):
        """메타데이터 저장"""
        try:
            self.metadata_cache["last_updated"] = datetime.now().isoformat()
            metadata_file = self.metadata_dir / "embedding_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")

    def get_chunk_hash(self, chunk: Chunk) -> str:
        """청크의 고유 해시 생성"""
        # 청크 내용, 문서 ID, 위치 정보를 조합하여 고유 ID 생성
        unique_str = f"{chunk.doc_id}:{chunk.start_idx}:{chunk.end_idx}:{chunk.content}"
        return hashlib.sha256(unique_str.encode('utf-8')).hexdigest()[:16]

    def save_chunk_embeddings(
            self,
            chunks: List[Chunk],
            embeddings: List[np.ndarray],
            chunk_type: str,
            document_id: str,
            language: Language,
            additional_metadata: Optional[Dict] = None
    ) -> bool:
        """청크와 임베딩을 저장"""

        if len(chunks) != len(embeddings):
            logger.error(f"청크 수({len(chunks)})와 임베딩 수({len(embeddings)})가 일치하지 않습니다.")
            return False

        try:
            # 1. 문서별 임베딩 저장 (NPZ 형식)
            doc_path = self.embeddings_dir / "by_document" / f"{document_id}.npz"

            # 기존 데이터가 있으면 로드
            existing_data = {}
            if doc_path.exists():
                try:
                    existing_data = dict(np.load(doc_path, allow_pickle=True))
                except Exception as e:
                    logger.warning(f"기존 임베딩 로드 실패: {e}")

            # 청크 타입별로 저장
            chunk_data = []
            embedding_matrix = []

            for chunk, embedding in zip(chunks, embeddings):
                chunk_hash = self.get_chunk_hash(chunk)
                chunk_data.append({
                    "hash": chunk_hash,
                    "content": chunk.content,  # 처음 200자만 저장 (메타데이터용)
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx,
                    "sequence_num": chunk.sequence_num,
                    "metadata": chunk.metadata or {},
                    "chunk_type": chunk_type
                })
                embedding_matrix.append(embedding)

            # 기존 데이터에 추가 또는 업데이트
            existing_data[f"{chunk_type}_embeddings"] = np.array(embedding_matrix)
            existing_data[f"{chunk_type}_chunks"] = chunk_data

            # NPZ 파일로 저장
            np.savez_compressed(doc_path, **existing_data)

            # 2. 청크 타입별 메타데이터 저장
            type_path = self.embeddings_dir / "by_chunk_type" / chunk_type / f"{document_id}.json"
            chunk_metadata = {
                "document_id": document_id,
                "chunk_type": chunk_type,
                "language": language.value,
                "num_chunks": len(chunks),
                "chunks": chunk_data,
                "created_at": datetime.now().isoformat(),
                "embedding_model": additional_metadata.get("model", "unknown") if additional_metadata else "unknown",
                "additional_metadata": additional_metadata or {}
            }

            with open(type_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)

            # 3. 언어별 심볼릭 링크 또는 참조 저장
            lang_path = self.embeddings_dir / "by_language" / language.value / f"{document_id}_{chunk_type}.json"
            with open(lang_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "document_id": document_id,
                    "chunk_type": chunk_type,
                    "doc_path": str(doc_path),
                    "type_path": str(type_path)
                }, f, indent=2)

            # 4. 메타데이터 업데이트
            self._update_metadata(document_id, chunk_type, len(chunks), language)

            logger.info(f"저장 완료: {document_id} - {len(chunks)}개 청크 ({chunk_type})")
            return True

        except Exception as e:
            logger.error(f"청크 임베딩 저장 실패: {e}")
            return False

    def load_chunk_embeddings(
            self,
            document_id: str,
            chunk_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """저장된 청크와 임베딩 로드"""

        doc_path = self.embeddings_dir / "by_document" / f"{document_id}.npz"
        if not doc_path.exists():
            logger.warning(f"임베딩 파일을 찾을 수 없습니다: {document_id}")
            return {}

        try:
            # NPZ 파일 로드
            data = np.load(doc_path, allow_pickle=True)

            result = {
                "document_id": document_id,
                "chunk_types": {}
            }

            # 특정 청크 타입 또는 모든 타입 로드
            if chunk_type:
                if f"{chunk_type}_embeddings" in data:
                    result["chunk_types"][chunk_type] = {
                        "embeddings": data[f"{chunk_type}_embeddings"],
                        "chunks": data[f"{chunk_type}_chunks"].tolist()
                    }
            else:
                # 모든 청크 타입 로드
                for key in data.keys():
                    if key.endswith("_embeddings"):
                        chunk_type_name = key.replace("_embeddings", "")
                        result["chunk_types"][chunk_type_name] = {
                            "embeddings": data[f"{chunk_type_name}_embeddings"],
                            "chunks": data[f"{chunk_type_name}_chunks"].tolist()
                        }

            return result

        except Exception as e:
            logger.error(f"임베딩 로드 실패: {e}")
            return {}

    def _update_metadata(self, document_id: str, chunk_type: str, num_chunks: int, language: Language):
        """메타데이터 업데이트"""
        # 문서별 메타데이터
        if document_id not in self.metadata_cache["documents"]:
            self.metadata_cache["documents"][document_id] = {
                "chunk_types": {},
                "total_chunks": 0,
                "language": language.value,
                "created_at": datetime.now().isoformat()
            }

        doc_meta = self.metadata_cache["documents"][document_id]
        doc_meta["chunk_types"][chunk_type] = num_chunks
        doc_meta["total_chunks"] = sum(doc_meta["chunk_types"].values())
        doc_meta["updated_at"] = datetime.now().isoformat()

        # 전체 통계 업데이트
        stats = self.metadata_cache["statistics"]
        stats["total_documents"] = len(self.metadata_cache["documents"])
        stats["total_chunks"] = sum(
            doc["total_chunks"]
            for doc in self.metadata_cache["documents"].values()
        )
        stats["total_embeddings"] = stats["total_chunks"]

        # 청크 타입별 통계
        if chunk_type not in stats["by_chunk_type"]:
            stats["by_chunk_type"][chunk_type] = 0
        stats["by_chunk_type"][chunk_type] += num_chunks

        # 언어별 통계
        stats["by_language"][language.value] += num_chunks

        self._save_metadata()

    def search_similar_chunks(
            self,
            query_embedding: np.ndarray,
            document_id: Optional[str] = None,
            chunk_type: Optional[str] = None,
            top_k: int = 10,
            threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """유사한 청크 검색"""
        all_results = []

        # 검색 대상 결정
        if document_id:
            docs_to_search = [document_id]
        else:
            docs_to_search = list(self.metadata_cache["documents"].keys())

        for doc_id in docs_to_search:
            doc_data = self.load_chunk_embeddings(doc_id, chunk_type)
            if not doc_data or not doc_data.get("chunk_types"):
                continue

            for c_type, type_data in doc_data["chunk_types"].items():
                embeddings = type_data["embeddings"]
                chunks = type_data["chunks"]

                # 코사인 유사도 계산
                query_norm = np.linalg.norm(query_embedding)
                embeddings_norm = np.linalg.norm(embeddings, axis=1)

                # 0으로 나누기 방지
                valid_indices = embeddings_norm > 0
                if not np.any(valid_indices):
                    continue

                similarities = np.zeros(len(embeddings))
                similarities[valid_indices] = np.dot(
                    embeddings[valid_indices],
                    query_embedding
                ) / (embeddings_norm[valid_indices] * query_norm)

                # 임계값 이상인 결과만 선택
                high_sim_indices = np.where(similarities >= threshold)[0]

                for idx in high_sim_indices:
                    if idx < len(chunks):
                        all_results.append({
                            "document_id": doc_id,
                            "chunk_type": c_type,
                            "chunk": chunks[idx],
                            "similarity": float(similarities[idx]),
                            "embedding_index": int(idx)
                        })

        # 전체 결과에서 상위 k개 반환
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """저장된 임베딩 통계 반환"""
        stats = self.metadata_cache["statistics"].copy()
        stats["storage_info"] = {
            "base_path": str(self.base_path),
            "embeddings_path": str(self.embeddings_dir),
            "total_size_mb": self._calculate_storage_size()
        }
        return stats

    def _calculate_storage_size(self) -> float:
        """저장 공간 크기 계산 (MB)"""
        total_size = 0
        for file_path in self.embeddings_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return round(total_size / (1024 * 1024), 2)

    def clear_document(self, document_id: str) -> bool:
        """특정 문서의 모든 임베딩 삭제"""
        try:
            # 문서 임베딩 파일 삭제
            doc_path = self.embeddings_dir / "by_document" / f"{document_id}.npz"
            if doc_path.exists():
                doc_path.unlink()

            # 청크 타입별 메타데이터 삭제
            for chunk_type_dir in (self.embeddings_dir / "by_chunk_type").iterdir():
                if chunk_type_dir.is_dir():
                    type_file = chunk_type_dir / f"{document_id}.json"
                    if type_file.exists():
                        type_file.unlink()

            # 언어별 참조 삭제
            for lang_dir in (self.embeddings_dir / "by_language").iterdir():
                if lang_dir.is_dir():
                    for ref_file in lang_dir.glob(f"{document_id}_*.json"):
                        ref_file.unlink()

            # 메타데이터에서 제거
            if document_id in self.metadata_cache["documents"]:
                del self.metadata_cache["documents"][document_id]
                self._save_metadata()

            logger.info(f"문서 임베딩 삭제 완료: {document_id}")
            return True

        except Exception as e:
            logger.error(f"문서 임베딩 삭제 실패: {e}")
            return False