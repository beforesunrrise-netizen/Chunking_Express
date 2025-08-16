"""
RAG 시스템 데이터 구조 정의
Data structures for RAG system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from src.config import Language, ChunkingStrategy, EnsembleMethod


@dataclass
class Document:
    """문서 데이터 구조"""
    id: str
    content: str
    language: Language
    title: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __len__(self) -> int:
        return len(self.content)

    def word_count(self) -> int:
        """단어 수 계산"""
        if self.language == Language.KOREAN:
            # 한국어는 공백으로 어절 단위 계산
            return len(self.content.split())
        else:
            # 영어는 공백으로 단어 단위 계산
            return len(self.content.split())


@dataclass
class Query:
    """쿼리 데이터 구조"""
    id: str
    question: str
    language: Language
    expected_answer: str = ""
    context_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __len__(self) -> int:
        return len(self.question)


@dataclass
class Chunk:
    """청크 데이터 구조"""
    id: str
    content: str
    document_id: str  # 이미 있음
    start_idx: int
    end_idx: int
    strategy: ChunkingStrategy
    sequence_num: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __len__(self) -> int:
        return len(self.content)

    def get_position_info(self) -> Dict[str, int]:
        """위치 정보 반환"""
        return {
            "start": self.start_idx,
            "end": self.end_idx,
            "length": self.end_idx - self.start_idx,
            "sequence": self.sequence_num
        }


@dataclass
class RAGResponse:
    """RAG 응답 구조"""
    query: Query
    query_id: str
    response: str
    chunks_used: List[Chunk]
    confidence: float
    strategy: ChunkingStrategy
    generation_time: float = 0.0
    model_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


    def get_chunk_ids(self) -> List[str]:
        """사용된 청크 ID 목록 반환"""
        return [chunk.id for chunk in self.chunks_used]

    def get_response_length(self) -> int:
        """응답 길이 반환"""
        return len(self.response)


@dataclass
class EvaluationScore:
    """평가 점수 구조"""
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def is_better_than(self, other: 'EvaluationScore') -> bool:
        """다른 점수와 비교"""
        if "auroc" in self.metric_name.lower():
            return self.score > other.score  # AUROC는 높을수록 좋음
        elif "rmse" in self.metric_name.lower():
            return self.score < other.score  # RMSE는 낮을수록 좋음
        else:
            return self.score > other.score  # 기본적으로 높을수록 좋음


@dataclass
class EvaluationResult:
    """평가 결과 구조"""
    strategy: str
    language: Language
    hallucination_auroc: float
    context_relevance_rmse: float
    utilization_rmse: float
    recall_at_k: float   # <--- 수정된 부분
    mrr: float           # <--- 수정된 부분
    num_samples: int = 0
    ensemble_method: Optional[EnsembleMethod] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "strategy": self.strategy,
            "language": self.language.value,
            "hallucination_auroc": self.hallucination_auroc,
            "context_relevance_rmse": self.context_relevance_rmse,
            "utilization_rmse": self.utilization_rmse,
            "recall_at_k": self.recall_at_k, # <--- 수정된 부분
            "mrr": self.mrr,                 # <--- 수정된 부분
            "num_samples": self.num_samples,
            "ensemble_method": self.ensemble_method.value if self.ensemble_method else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    def get_overall_score(self) -> float:
        """전체 점수 계산 (정규화된 점수)"""
        # AUROC는 높을수록 좋고, RMSE는 낮을수록 좋음
        auroc_score = self.hallucination_auroc
        rmse_score = 1 - min(self.context_relevance_rmse, 1)
        util_score = 1 - min(self.utilization_rmse, 1)

        # 새로운 지표 추가 (Recall@K, MRR) - 둘 다 높을수록 좋음
        recall_score = self.recall_at_k
        mrr_score = self.mrr

        # 가중 평균 (모든 지표의 중요도를 동일하게 설정)
        return (auroc_score + rmse_score + util_score + recall_score + mrr_score) / 5


@dataclass
class ExperimentRun:
    """실험 실행 정보"""
    run_id: str
    config: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[EvaluationResult] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: EvaluationResult):
        """결과 추가"""
        self.results.append(result)

    def add_error(self, error: Exception, context: str = ""):
        """오류 추가"""
        self.errors.append({
            "error": str(error),
            "type": type(error).__name__,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })

    def get_duration(self) -> Optional[float]:
        """실행 시간 계산 (초)"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def get_summary(self) -> Dict[str, Any]:
        """실험 요약 정보"""
        return {
            "run_id": self.run_id,
            "duration_seconds": self.get_duration(),
            "num_results": len(self.results),
            "num_errors": len(self.errors),
            "total_cost": self.total_cost,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class ChunkingMetrics:
    """청킹 메트릭"""
    num_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    chunk_size_std: float
    overlap_ratio: float = 0.0
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "num_chunks": self.num_chunks,
            "avg_chunk_size": self.avg_chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "chunk_size_std": self.chunk_size_std,
            "overlap_ratio": self.overlap_ratio,
            "processing_time": self.processing_time
        }