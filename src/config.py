"""
RAG 청킹 전략 비교 연구 - 설정 파일
Configuration file for RAG Chunking Strategy Comparison Study
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import logging


class Language(Enum):
    """지원 언어"""
    ENGLISH = "en"
    KOREAN = "kr"


class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    QUERY_AWARE = "query_aware"
    RECURSIVE = 'recursive'
    TEXT_SIMILARITY = 'text_similarity'


class EnsembleMethod(Enum):
    """앙상블 방법"""
    VOTING = "voting"
    RERANKING = "reranking"
    FUSION = "fusion"


@dataclass
class ModelConfig:
    """모델 설정"""
    """gpt model"""
    gpt_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.1
    max_tokens: int = 2000
    embedding_dimension: int = 3072

# sample_size --> 문서수와 겹쳐야한다
@dataclass
class ExperimentConfig:
    """실험 설정"""
    sample_size: int = 100
    chunk_size_limit: int = 512
    context_window: int = 2
    top_k_retrieval: int = 5
    batch_size: int = 10
    num_workers: int = 4


@dataclass
class EvaluationConfig:
    """평가 설정"""
    metrics: List[str] = None
    significance_level: float = 0.05
    bootstrap_iterations: int = 1000

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "hallucination_auroc",
                "context_relevance_rmse",
                "utilization_rmse"
            ]


@dataclass
class PathConfig:
    """경로 설정"""
    root_dir: Path = Path(__file__).parent
    data_dir: Path = None
    results_dir: Path = None
    logs_dir: Path = None
    cache_dir: Path = None
    embedding_storage_path: Path = None  # ◀◀◀ 이 줄을 추가하세요.

    def __post_init__(self):
        self.data_dir = self.root_dir / "data"
        self.results_dir = self.root_dir / "results"
        self.logs_dir = self.root_dir / "logs"
        self.cache_dir = self.root_dir / "cache"


        self.embedding_storage_path = Path("/Users/jaeyoung/Desktop/Projects/Chunking_Express/src/data")


        for dir_path in [
            self.data_dir, self.results_dir, self.logs_dir,
            self.cache_dir, self.embedding_storage_path
        ]:
            dir_path.mkdir(exist_ok=True)


@dataclass
class APIConfig:
    """API 설정"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "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")
    openai_org_id: Optional[str] = os.getenv("OPENAI_ORG_ID")
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_rotation: str = "1 day"
    retention: str = "7 days"

    def setup_logging(self, log_dir: Path):
        """로깅 설정 초기화"""
        from loguru import logger

        # 기본 로거 제거
        logger.remove()

        # 콘솔 출력
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level=self.level
        )

        # 파일 출력
        logger.add(
            log_dir / "rag_experiment_{time}.log",
            rotation=self.file_rotation,
            retention=self.retention,
            level=self.level,
            format=self.format
        )

        return logger


@dataclass
class DatasetConfig:
    """데이터셋 설정"""
    data_path: str = "rag_squad_train_100_samples.json"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42


@dataclass
class CostConfig:
    """비용 관련 설정"""
    # GPT-4o-mini pricing (per 1K tokens)
    gpt4_input_cost: float = 0.005
    gpt4_output_cost: float = 0.015

    # Embedding pricing (per 1K tokens)
    embedding_cost: float = 0.00013

    # 예산 제한
    max_budget: float = 100.0  # USD
    warning_threshold: float = 50.0  # USD


class Config:
    """전체 설정 관리 클래스"""

    def __init__(self):
        self.model = ModelConfig()
        self.experiment = ExperimentConfig()
        self.evaluation = EvaluationConfig()
        self.paths = PathConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.dataset = DatasetConfig()
        self.cost = CostConfig()

        # 로거 설정
        self.logger = self.logging.setup_logging(self.paths.logs_dir)

        # API 키 확인
        if not self.api.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

    def get_model_config(self, language: Language) -> Dict:
        """언어별 모델 설정 반환"""
        if language == Language.KOREAN:
            return {
                "gpt_model": self.model.gpt_model,
                "embedding_model": "text-embedding-3-large",  # 다국어 지원
                "temperature": self.model.temperature
            }
        else:
            return {
                "gpt_model": self.model.gpt_model,
                "embedding_model": self.model.embedding_model,
                "temperature": self.model.temperature
            }

    def get_data_path(self, language: Language) -> Path:
        p = Path(self.dataset.data_path)

        # 절대경로면 그대로
        if p.is_absolute():
            return p

        # 상대경로인데 이미 data/로 시작하면 root_dir 기준으로만 결합
        if str(p).startswith(("data/", "data\\")):
            return (self.paths.root_dir / p).resolve()

        # 그 외엔 data_dir/파일명
        return (self.paths.data_dir / p.name).resolve()

    def estimate_cost(self, input_tokens: int, output_tokens: int, embedding_tokens: int) -> float:
        """예상 비용 계산"""
        input_cost = (input_tokens / 1000) * self.cost.gpt4_input_cost
        output_cost = (output_tokens / 1000) * self.cost.gpt4_output_cost
        embedding_cost = (embedding_tokens / 1000) * self.cost.embedding_cost

        total_cost = input_cost + output_cost + embedding_cost

        if total_cost > self.cost.warning_threshold:
            self.logger.warning(f"예상 비용이 경고 임계값을 초과합니다: ${total_cost:.2f}")

        return total_cost

    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환"""
        return {
            "model": self.model.__dict__,
            "experiment": self.experiment.__dict__,
            "evaluation": self.evaluation.__dict__,
            "dataset": self.dataset.__dict__,
            "cost": self.cost.__dict__
        }


# 전역 설정 인스턴스
config = Config()