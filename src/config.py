"""
RAG 청킹 전략 비교 연구 - 설정 파일
Configuration file for RAG Chunking Strategy Comparison Study
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum, auto
from src.env.config_yml import api_config


class Language(Enum):
    """지원 언어"""
    ENGLISH = "en"
    KOREAN = "kr"


class ChunkingStrategy(Enum):
    """청킹 전략"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    QUERY_AWARE = "query_aware"
    RECURSIVE = "recursive"
    TEXT_SIMILARITY = "text_similarity"
    ADAPTIVE = "adaptive"  # LangGraph 기반 적응형 전략 추가

# +++ FIX: Added the missing EnsembleMethod Enum +++
class EnsembleMethod(Enum):
    """Defines the methods for ensembling or combining chunking strategies."""
    RANK_FUSION = auto()
    WEIGHTED_AVERAGE = auto()
    MAJORITY_VOTE = auto()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++

class DatasetType(Enum):
    """데이터셋 타입"""
    SQUAD = "squad"
    SQUAD_V2 = "squad_v2"
    NEWSQA = "newsqa"
    TECHQA = "techqa"
    COVID_QA = "covid_qa"
    NARRATIVEQA = "deepmind/narrativeqa"
    CUSTOM = "custom"


@dataclass
class DatasetConfig:
    """데이터셋 설정"""
    # 기본 데이터셋
    default_dataset: DatasetType = DatasetType.SQUAD

    # 데이터셋별 설정
    dataset_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "squad": {
            "path": "squad",
            "split": "train",
            "sample_size": 100,
            "format": "standardized"
        },
        "squad_v2": {
            "path": "squad_v2",
            "split": "train",
            "sample_size": 100,
            "format": "standardized"
        },
        "newsqa": {
            "path": "newsqa",
            "split": "train",
            "sample_size": 100,
            "format": "standardized"
        },
        "techqa": {
            "path": "techqa",
            "split": "train",
            "sample_size": 100,
            "format": "standardized"
        },
        "covid_qa": {
            "path": "covid_qa",
            "split": "train",
            "sample_size": 100,
            "format": "standardized"
        },
        "deepmind/narrativeqa": {
            "path": "deepmind/narrativeqa",
            "split": "train",
            "sample_size": 100,
            "format": "standardized"
        }
    })

    # 통합 데이터셋 설정
    combined_dataset_path: str = "combined_qa_dataset.jsonl"
    use_combined: bool = False

    # 데이터 분할
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

    def get_dataset_config(self, dataset_name: str) -> Dict:
        """특정 데이터셋 설정 반환"""
        if dataset_name in self.dataset_configs:
            return self.dataset_configs[dataset_name]
        return self.dataset_configs.get(self.default_dataset.value, {})


@dataclass
class AdaptiveChunkingConfig:
    """적응형 청킹 설정"""
    enable_adaptive: bool = True
    use_llm_analysis: bool = True
    llm_model: str = "gpt-4"

    # 카테고리별 전략 매핑
    category_strategies: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "science": {
            "high_complexity": ["semantic", "recursive"],
            "medium_complexity": ["semantic"],
            "low_complexity": ["fixed_size"]
        },
        "technology": {
            "high_complexity": ["keyword", "semantic"],
            "medium_complexity": ["keyword"],
            "low_complexity": ["fixed_size"]
        },
        "business": {
            "high_complexity": ["semantic", "keyword"],
            "medium_complexity": ["keyword"],
            "low_complexity": ["fixed_size"]
        },
        "medical": {
            "high_complexity": ["semantic", "keyword"],
            "medium_complexity": ["semantic"],
            "low_complexity": ["keyword"]
        },
        "news": {
            "high_complexity": ["semantic"],
            "medium_complexity": ["fixed_size"],
            "low_complexity": ["fixed_size"]
        },
        "literature": {
            "high_complexity": ["text_similarity", "semantic"],
            "medium_complexity": ["semantic"],
            "low_complexity": ["fixed_size"]
        }
    })

    # 성능 학습 설정
    enable_learning: bool = True
    history_file: str = "chunking_history.json"
    min_history_size: int = 10

    # 품질 임계값
    quality_threshold: float = 0.6
    enable_backup_strategies: bool = True


@dataclass
class ModelConfig:
    """모델 설정"""
    gpt_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.1
    max_tokens: int = 2000
    embedding_dimension: int = 3072


@dataclass
class ExperimentConfig:
    """실험 설정"""
    sample_size: int = 100
    chunk_size_limit: int = 512
    overlap_ratio: float = 0.1
    context_window: int = 2
    top_k_retrieval: int = 5
    batch_size: int = 10
    num_workers: int = 4

    # 실험 모드
    experiment_mode: str = "adaptive"  # adaptive, traditional, comparison
    compare_baseline: bool = False

    # 병렬 처리
    max_concurrent: int = 5
    use_async: bool = True


@dataclass
class EvaluationConfig:
    """평가 설정"""
    metrics: List[str] = None
    significance_level: float = 0.05
    bootstrap_iterations: int = 5

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "mrr",  # Mean Reciprocal Rank
                "hit_at_k",  # Hit@K
                "recall",
                "precision",
                "f1_score"
            ]


@dataclass
class PathConfig:
    """경로 설정"""
    root_dir: Path = Path(__file__).parent
    data_dir: Path = None
    results_dir: Path = None
    logs_dir: Path = None
    cache_dir: Path = None
    embedding_storage_path: Path = None

    def __post_init__(self):
        self.data_dir = self.root_dir / "data"
        self.results_dir = self.root_dir / "results"
        self.logs_dir = self.root_dir / "logs"
        self.cache_dir = self.root_dir / "cache"
        self.embedding_storage_path = Path(
            os.getenv("EMBEDDING_STORE_PATH",
                      "/Users/jaeyoung/Desktop/Projects/Chunking_Express/src/data")
        )

        for dir_path in [
            self.data_dir,
            self.results_dir,
            self.logs_dir,
            self.cache_dir,
            self.embedding_storage_path,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfig:
    """API 설정"""
    openai_api_key: str = api_config.openai_api_key
    openai_org_id: Optional[str] = api_config.openai_org_id
    request_timeout: int = api_config.request_timeout
    max_retries: int = api_config.max_retries
    retry_delay: float = api_config.retry_delay


@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_rotation: str = "1 day"
    retention: str = "7 days"

    def setup_logging(self, log_dir: Path):
        from loguru import logger
        logger.remove()

        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level=self.level,
        )

        logger.add(
            log_dir / "rag_experiment_{time}.log",
            rotation=self.file_rotation,
            retention=self.retention,
            level=self.level,
            format=self.format,
        )

        return logger


@dataclass
class CostConfig:
    """비용 관련 설정"""
    gpt4_input_cost: float = 0.005
    gpt4_output_cost: float = 0.015
    embedding_cost: float = 0.00013
    max_budget: float = 100.0
    warning_threshold: float = 50.0


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
        self.adaptive = AdaptiveChunkingConfig()
        self.cost = CostConfig()

        # 로거 설정
        self.logger = self.logging.setup_logging(self.paths.logs_dir)

        # API 키 확인
        if not self.api.openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

    def get_dataset_path(self, dataset_name: str) -> Path:
        """데이터셋 경로 반환"""
        config = self.dataset.get_dataset_config(dataset_name)
        path = config.get("path", dataset_name)

        # 통합 데이터셋 사용 시
        if self.dataset.use_combined:
            return self.paths.data_dir / self.dataset.combined_dataset_path

        # 개별 데이터셋
        return self.paths.data_dir / f"{path.replace('/', '_')}_processed.jsonl"

    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환"""
        return {
            "model": self.model.__dict__,
            "experiment": self.experiment.__dict__,
            "evaluation": self.evaluation.__dict__,
            "dataset": {
                "default": self.dataset.default_dataset.value,
                "configs": self.dataset.dataset_configs,
                "use_combined": self.dataset.use_combined
            },
            "adaptive": {
                "enabled": self.adaptive.enable_adaptive,
                "llm_analysis": self.adaptive.use_llm_analysis,
                "category_strategies": self.adaptive.category_strategies
            },
            "cost": self.cost.__dict__,
        }


# 전역 설정 인스턴스
config = Config()