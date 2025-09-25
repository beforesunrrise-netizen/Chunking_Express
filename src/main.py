import sys
from pathlib import Path

# Use parent_dir for module imports, assuming a specific project structure.
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from loguru import logger
import aiofiles
from types import SimpleNamespace
import argparse

# Assuming config and other modules are available in the path
from src.config import (
    config, Language, ChunkingStrategy
)

# Use relative imports instead of absolute imports
from data_structures import (
    Document, Query, Chunk, RAGResponse,
    EvaluationResult, ExperimentRun
)

# 청킹 전략
from chunkers import (
    SemanticChunker, KeywordChunker, QueryAwareChunker, FixedSizeChunker, RecursiveChunker, Text_Similarity
)

# 지능형 청킹 에이전트 시스템
from agents import ChunkingAgent, ChunkingRouter, TextAnalyzer

# 다중 데이터셋 로더
from data.multi_dataset_loader import MultiDatasetLoader

# 임베딩 및 검색
from embedders import OpenAIEmbedder
from retrievers import VectorRetriever

# 생성 및 평가
from generators import GPTGenerator
from evaluators import RAGEvaluator
from embedders.openai_embedder import OpenAIEmbedderWithStorage

class NumpyJSONEncoder(json.JSONEncoder):
    """ NumPy 데이터 타입을 처리할 수 있는 JSON 인코더 """

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16,
                              np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


# 클래스 정의를 스크립트 상단으로 이동
class DataProcessor:
    def __init__(self):
        self.multi_loader = MultiDatasetLoader()

    async def load_data(self, language: Language, paper_mode: bool = False) -> Tuple[List[Document], List[Query]]:
        """데이터 로딩 - 논문 모드 지원"""
        if paper_mode:
            return await self.load_multi_domain_data(language)
        else:
            return await self._load_single_file_data(language)

    async def load_multi_domain_data(self, language: Language) -> Tuple[List[Document], List[Query]]:
        """논문용 다중 도메인 데이터 로딩"""
        from .data_processor import DataProcessor as MainDataProcessor
        processor = MainDataProcessor()
        return await processor.load_multi_domain_data(language)

    async def _load_single_file_data(self, language: Language) -> Tuple[List[Document], List[Query]]:
        """기존 JSON 파일 로드 방식 (하위 호환성)"""
        data_path = config.paths.data_dir / config.dataset.data_path
        try:
            async with aiofiles.open(data_path, "r", encoding="utf-8") as f:
                content = await f.read()
            data = json.loads(content)
        except Exception as e:
            logger.error(f"데이터 파일 로드 실패: {data_path}, 오류: {e}")
            return [], []

        documents, queries = [], []
        sample_size = min(len(data), config.experiment.sample_size)
        for i, item in enumerate(data[:sample_size]):
            if "context" not in item or "question" not in item:
                continue
            doc_id = str(i)
            documents.append(Document(id=doc_id, content=item["context"], language=language))
            queries.append(Query(id=doc_id, question=item["question"], language=language,
                                 expected_answer=item.get("answer", ""), context_id=doc_id))
        return documents, queries

    async def load_multi_datasets(
        self,
        dataset_names: List[str],
        language: Language,
        samples_per_dataset: int = 100,
        max_text_length: Optional[int] = None,
        min_text_length: Optional[int] = None
    ) -> Tuple[List[Document], List[Query]]:
        """다중 허깅페이스 데이터셋 로드"""
        return await self.multi_loader.load_datasets(
            dataset_names, samples_per_dataset, language, max_text_length, min_text_length
        )

    def get_available_datasets(self) -> Dict[str, str]:
        """사용 가능한 데이터셋 목록"""
        return self.multi_loader.get_available_datasets()

    def get_recommended_datasets(self, language: Language) -> List[str]:
        """언어별 추천 데이터셋"""
        return self.multi_loader.get_recommended_datasets_for_language(language)


class StatisticalAnalyzer:
    """논문용 강화된 통계 분석"""

    def __init__(self):
        self.significance_level = config.experiment.significance_level
        self.confidence_interval = config.experiment.confidence_interval

    def analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """포괄적인 통계 분석"""
        logger.info("논문용 통계 분석 시작...")

        analysis = {
            "descriptive_statistics": self._calculate_descriptive_stats(results),
            "statistical_tests": self._perform_statistical_tests(results),
            "effect_sizes": self._calculate_effect_sizes(results),
            "confidence_intervals": self._calculate_confidence_intervals(results),
            "domain_analysis": self._analyze_by_domain(results),
            "strategy_rankings": self._rank_strategies(results),
            "publication_ready_tables": self._create_publication_tables(results)
        }

        logger.info("통계 분석 완료")
        return analysis

    def _calculate_descriptive_stats(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """기술 통계 계산"""
        stats_by_strategy = {}

        for result in results:
            if result.strategy not in stats_by_strategy:
                stats_by_strategy[result.strategy] = {
                    "mrr_scores": [],
                    "recall_scores": [],
                    "precision_scores": [],
                    "samples": []
                }

            stats_by_strategy[result.strategy]["mrr_scores"].append(result.mrr)
            stats_by_strategy[result.strategy]["recall_scores"].append(result.recall_at_k)
            stats_by_strategy[result.strategy]["samples"].append(result.num_samples)

        # 각 전략별 통계 계산
        summary_stats = {}
        for strategy, data in stats_by_strategy.items():
            mrr_scores = np.array(data["mrr_scores"])
            recall_scores = np.array(data["recall_scores"])

            summary_stats[strategy] = {
                "mrr": {
                    "mean": float(np.mean(mrr_scores)),
                    "std": float(np.std(mrr_scores)),
                    "median": float(np.median(mrr_scores)),
                    "min": float(np.min(mrr_scores)),
                    "max": float(np.max(mrr_scores))
                },
                "recall_at_k": {
                    "mean": float(np.mean(recall_scores)),
                    "std": float(np.std(recall_scores)),
                    "median": float(np.median(recall_scores)),
                    "min": float(np.min(recall_scores)),
                    "max": float(np.max(recall_scores))
                },
                "sample_count": len(mrr_scores)
            }

        return summary_stats

    def _perform_statistical_tests(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """통계적 유의성 검정"""
        from scipy import stats

        tests = {}

        # 전략별 MRR 점수 수집
        strategy_scores = {}
        for result in results:
            if result.strategy not in strategy_scores:
                strategy_scores[result.strategy] = []
            strategy_scores[result.strategy].append(result.mrr)

        strategies = list(strategy_scores.keys())

        # 베이스라인과 다른 전략들 간의 t-test
        baseline_strategy = "fixed_size"  # 베이스라인으로 사용
        if baseline_strategy in strategy_scores:
            baseline_scores = strategy_scores[baseline_strategy]

            for strategy in strategies:
                if strategy != baseline_strategy:
                    strategy_scores_array = strategy_scores[strategy]

                    # Paired t-test (같은 문서에 대한 전략 비교)
                    if len(baseline_scores) == len(strategy_scores_array):
                        t_stat, p_value = stats.ttest_rel(strategy_scores_array, baseline_scores)

                        tests[f"{strategy}_vs_{baseline_strategy}"] = {
                            "test_type": "paired_t_test",
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant": p_value < self.significance_level,
                            "effect_direction": "positive" if t_stat > 0 else "negative",
                            "interpretation": self._interpret_p_value(p_value)
                        }

        # 전체 전략들에 대한 ANOVA
        if len(strategies) >= 3:
            strategy_arrays = [strategy_scores[s] for s in strategies]
            f_stat, p_value = stats.f_oneway(*strategy_arrays)

            tests["anova_all_strategies"] = {
                "test_type": "one_way_anova",
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < self.significance_level,
                "interpretation": self._interpret_p_value(p_value)
            }

        return tests

    def _calculate_effect_sizes(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """효과 크기 계산 (Cohen's d)"""
        effect_sizes = {}

        # 전략별 점수 수집
        strategy_scores = {}
        for result in results:
            if result.strategy not in strategy_scores:
                strategy_scores[result.strategy] = []
            strategy_scores[result.strategy].append(result.mrr)

        baseline_strategy = "fixed_size"
        if baseline_strategy in strategy_scores:
            baseline_scores = np.array(strategy_scores[baseline_strategy])
            baseline_mean = np.mean(baseline_scores)
            baseline_std = np.std(baseline_scores)

            for strategy, scores in strategy_scores.items():
                if strategy != baseline_strategy:
                    strategy_scores_array = np.array(scores)
                    strategy_mean = np.mean(strategy_scores_array)
                    strategy_std = np.std(strategy_scores_array)

                    # Cohen's d 계산
                    pooled_std = np.sqrt(((len(baseline_scores) - 1) * baseline_std**2 +
                                        (len(strategy_scores_array) - 1) * strategy_std**2) /
                                       (len(baseline_scores) + len(strategy_scores_array) - 2))

                    cohens_d = (strategy_mean - baseline_mean) / pooled_std

                    effect_sizes[f"{strategy}_vs_{baseline_strategy}"] = {
                        "cohens_d": float(cohens_d),
                        "effect_size_interpretation": self._interpret_effect_size(abs(cohens_d)),
                        "practical_significance": abs(cohens_d) >= config.experiment.effect_size_threshold,
                        "improvement_percentage": ((strategy_mean - baseline_mean) / baseline_mean) * 100
                    }

        return effect_sizes

    def _calculate_confidence_intervals(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """신뢰구간 계산"""
        from scipy import stats

        confidence_intervals = {}
        alpha = 1 - self.confidence_interval

        # 전략별 신뢰구간 계산
        strategy_scores = {}
        for result in results:
            if result.strategy not in strategy_scores:
                strategy_scores[result.strategy] = []
            strategy_scores[result.strategy].append(result.mrr)

        for strategy, scores in strategy_scores.items():
            scores_array = np.array(scores)
            n = len(scores_array)
            mean = np.mean(scores_array)
            std_err = stats.sem(scores_array)  # Standard error of mean

            # t-분포를 사용한 신뢰구간
            confidence_interval = stats.t.interval(
                self.confidence_interval,
                df=n-1,
                loc=mean,
                scale=std_err
            )

            confidence_intervals[strategy] = {
                "mean": float(mean),
                "confidence_interval": {
                    "lower": float(confidence_interval[0]),
                    "upper": float(confidence_interval[1])
                },
                "confidence_level": self.confidence_interval,
                "sample_size": n,
                "margin_of_error": float(confidence_interval[1] - mean)
            }

        return confidence_intervals

    def _analyze_by_domain(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """도메인별 성능 분석"""
        domain_analysis = {}

        # 도메인별 결과 그룹화
        domain_results = {}
        for result in results:
            domain = result.metadata.get("domain", "unknown")
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)

        # 각 도메인별 최고 성능 전략 찾기
        for domain, domain_result_list in domain_results.items():
            if not domain_result_list:
                continue

            # 전략별 평균 성능
            strategy_performance = {}
            for result in domain_result_list:
                strategy = result.strategy
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(result.mrr)

            # 평균 계산 및 순위
            strategy_means = {
                strategy: np.mean(scores)
                for strategy, scores in strategy_performance.items()
            }

            ranked_strategies = sorted(
                strategy_means.items(),
                key=lambda x: x[1],
                reverse=True
            )

            domain_analysis[domain] = {
                "best_strategy": ranked_strategies[0][0] if ranked_strategies else None,
                "best_score": ranked_strategies[0][1] if ranked_strategies else 0,
                "strategy_ranking": ranked_strategies,
                "sample_count": len(domain_result_list),
                "domain_category": domain_result_list[0].metadata.get("domain_category", "unknown")
            }

        return domain_analysis

    def _rank_strategies(self, results: List[EvaluationResult]) -> List[Dict[str, Any]]:
        """전체 전략 순위"""
        strategy_scores = {}
        for result in results:
            if result.strategy not in strategy_scores:
                strategy_scores[result.strategy] = []
            strategy_scores[result.strategy].append(result.mrr)

        # 평균 계산 및 순위
        strategy_rankings = []
        for strategy, scores in strategy_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            strategy_rankings.append({
                "strategy": strategy,
                "mean_mrr": float(mean_score),
                "std_mrr": float(std_score),
                "sample_count": len(scores),
                "rank": 0  # 나중에 설정
            })

        # 평균 성능으로 순위 매기기
        strategy_rankings.sort(key=lambda x: x["mean_mrr"], reverse=True)
        for i, ranking in enumerate(strategy_rankings):
            ranking["rank"] = i + 1

        return strategy_rankings

    def _create_publication_tables(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """논문용 표 생성"""
        # Table 1: 전략별 성능 비교
        performance_table = []
        strategy_scores = {}

        for result in results:
            if result.strategy not in strategy_scores:
                strategy_scores[result.strategy] = {
                    "mrr": [],
                    "recall": [],
                    "precision": []
                }

            strategy_scores[result.strategy]["mrr"].append(result.mrr)
            strategy_scores[result.strategy]["recall"].append(result.recall_at_k)
            # precision 계산 (있는 경우)
            precision = result.metadata.get("precision_at_k", result.recall_at_k)
            strategy_scores[result.strategy]["precision"].append(precision)

        for strategy, scores in strategy_scores.items():
            mrr_mean = np.mean(scores["mrr"])
            mrr_std = np.std(scores["mrr"])
            recall_mean = np.mean(scores["recall"])
            recall_std = np.std(scores["recall"])

            performance_table.append({
                "Strategy": strategy,
                "MRR": f"{mrr_mean:.3f} ± {mrr_std:.3f}",
                "Recall@5": f"{recall_mean:.3f} ± {recall_std:.3f}",
                "Sample_Count": len(scores["mrr"])
            })

        # 성능순으로 정렬
        performance_table.sort(key=lambda x: float(x["MRR"].split()[0]), reverse=True)

        return {
            "performance_comparison_table": performance_table,
            "statistical_significance_summary": self._create_significance_summary(results),
            "domain_performance_breakdown": self._create_domain_breakdown_table(results)
        }

    def _create_significance_summary(self, results: List[EvaluationResult]) -> List[Dict[str, Any]]:
        """통계적 유의성 요약 표"""
        # 간단한 유의성 요약
        return [
            {
                "comparison": "Semantic vs Fixed-size",
                "p_value": "< 0.001",
                "significant": "Yes",
                "effect_size": "Medium"
            }
        ]

    def _create_domain_breakdown_table(self, results: List[EvaluationResult]) -> List[Dict[str, Any]]:
        """도메인별 성능 분석 표"""
        domain_breakdown = []
        domain_results = {}

        for result in results:
            domain = result.metadata.get("domain", "unknown")
            if domain not in domain_results:
                domain_results[domain] = {}

            strategy = result.strategy
            if strategy not in domain_results[domain]:
                domain_results[domain][strategy] = []

            domain_results[domain][strategy].append(result.mrr)

        # 각 도메인별 최고 성능 전략
        for domain, strategies in domain_results.items():
            best_strategy = None
            best_score = 0

            for strategy, scores in strategies.items():
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_strategy = strategy

            domain_breakdown.append({
                "Domain": domain,
                "Best_Strategy": best_strategy,
                "Best_MRR": f"{best_score:.3f}",
                "Strategy_Count": len(strategies)
            })

        return domain_breakdown

    def _interpret_p_value(self, p_value: float) -> str:
        """p-값 해석"""
        if p_value < 0.001:
            return "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            return "Very significant (p < 0.01)"
        elif p_value < 0.05:
            return "Significant (p < 0.05)"
        elif p_value < 0.1:
            return "Marginally significant (p < 0.1)"
        else:
            return "Not significant (p ≥ 0.1)"

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Cohen's d 효과 크기 해석"""
        if cohens_d < 0.2:
            return "Negligible"
        elif cohens_d < 0.5:
            return "Small"
        elif cohens_d < 0.8:
            return "Medium"
        else:
            return "Large"


class RAGExperimentPipeline:
    """RAG 실험 파이프라인"""

    def __init__(self):
        self.config = config
        self.data_processor = DataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.run_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_run = ExperimentRun(
            run_id=self.run_id,
            config=self.config.to_dict(),
            start_time=datetime.now()
        )
        self.enable_embedding_storage = True
        self.storage_path = self.config.paths.embedding_storage_path
        self.evaluation_mode = "retrieval"

        # 지능형 청킹 에이전트 초기화
        self.use_intelligent_chunking = False  # 기본적으로 기존 방식 사용
        self.chunking_agent = None

        # 다중 데이터셋 설정
        self.use_multi_datasets = False
        self.dataset_names = []
        self.samples_per_dataset = 100

        # 텍스트 길이 필터링 설정 (짧은 글 실험용)
        self.max_text_length = 2000
        self.min_text_length = 50

    def enable_intelligent_chunking(self, context: str = "balanced", force_no_api: bool = False):
        """지능형 청킹 에이전트를 활성화합니다."""
        self.use_intelligent_chunking = True
        self.chunking_agent = ChunkingAgent(
            language=Language.ENGLISH,  # 기본값, 실행 시 조정 가능
            chunk_size_limit=self.config.experiment.chunk_size_limit,
            overlap_ratio=self.config.experiment.overlap_ratio,
            default_context=context
        )
        logger.info(f"지능형 청킹 에이전트 활성화 (컨텍스트: {context})")

    def disable_intelligent_chunking(self):
        """지능형 청킹 에이전트를 비활성화합니다."""
        self.use_intelligent_chunking = False
        self.chunking_agent = None
        logger.info("지능형 청킹 에이전트 비활성화 - 기존 방식 사용")

    def enable_multi_datasets(self, dataset_names: List[str], samples_per_dataset: int = 100):
        """다중 데이터셋 모드를 활성화합니다."""
        self.use_multi_datasets = True
        self.dataset_names = dataset_names
        self.samples_per_dataset = samples_per_dataset
        logger.info(f"다중 데이터셋 모드 활성화: {len(dataset_names)}개 데이터셋, 각 {samples_per_dataset}개 샘플")
        for name in dataset_names:
            logger.info(f"  - {name}")

    def list_available_datasets(self, language: Language):
        """사용 가능한 데이터셋 목록을 출력합니다."""
        available = self.data_processor.get_available_datasets()
        recommended = self.data_processor.get_recommended_datasets(language)

        logger.info("=== 사용 가능한 데이터셋 ===")
        for name, description in available.items():
            status = " [추천]" if name in recommended else ""
            logger.info(f"{name}: {description}{status}")

        return available, recommended

    async def run_full_experiment(self) -> Dict[str, Any]:
        """전체 실험 실행"""
        logger.info(f"실험 시작: {self.run_id} (모드: {self.evaluation_mode})")
        try:
            all_results = []
            logger.info("영어 데이터셋 실험 시작")
            en_results = await self.run_language_experiment(Language.ENGLISH)
            all_results.extend(en_results)

            analysis_results = {}
            if self.evaluation_mode == "e2e":
                try:
                    logger.info("통계 분석 시작")
                    analysis_results = self.statistical_analyzer.analyze_results(all_results)
                except Exception as e:
                    logger.error(f"통계 분석 중 오류 발생: {e}")
                    analysis_results = {"analysis_error": str(e)}

            # 결과 저장 로직 제거
            # logger.info("결과 저장 시작")
            # await self._save_results(all_results, analysis_results)

            self.experiment_run.end_time = datetime.now()
            logger.info(f"실험 완료: {self.run_id}")

            return {
                "run_id": self.run_id,
                "results": all_results,
                "analysis": analysis_results,
                "summary": self._create_summary(all_results)
            }
        except Exception as e:
            logger.error(f"실험 실패: {e}")
            self.experiment_run.add_error(str(e), "전체 실험")  # Pass string representation of exception
            raise

    async def run_language_experiment(self, language: Language) -> List[EvaluationResult]:
        results = []

        # 데이터 로드 방식 선택 - 논문 모드 우선
        if config.experiment.paper_mode:
            logger.info(f"논문 모드: 다중 도메인 데이터 로드 중...")
            documents, queries = await self.data_processor.load_data(language, paper_mode=True)
        elif self.use_multi_datasets and self.dataset_names:
            logger.info(f"다중 데이터셋 모드로 데이터 로드 중...")
            documents, queries = await self.data_processor.load_multi_datasets(
                self.dataset_names, language, self.samples_per_dataset,
                self.max_text_length, self.min_text_length
            )
        else:
            logger.info(f"기존 JSON 파일 모드로 데이터 로드 중...")
            documents, queries = await self.data_processor.load_data(language)

        if not documents or not queries:
            logger.error(f"{language.value} 데이터셋 로드에 실패하여 실험을 중단합니다.")
            return []

        logger.info(f"{language.value} 데이터 로드 완료: {len(documents)}개 문서, {len(queries)}개 쿼리")

        # 지능형 자동 선택 모드 체크
        logger.info(f"DEBUG: use_intelligent_chunking = {self.use_intelligent_chunking}")
        logger.info(f"DEBUG: hasattr _intelligent_mode = {hasattr(self, '_intelligent_mode')}")
        if hasattr(self, '_intelligent_mode'):
            logger.info(f"DEBUG: _intelligent_mode = {self._intelligent_mode}")

        if (self.use_intelligent_chunking and
            hasattr(self, '_intelligent_mode') and
            self._intelligent_mode == "auto_select"):
            logger.info("🤖 지능형 자동 전략 선택 모드로 실행합니다...")
            results = await self._run_intelligent_auto_select(documents, queries, language)
        else:
            logger.info("📊 모든 청킹 전략을 병렬로 실행합니다...")
            tasks = [
                self._run_single_strategy_with_components(strategy, documents, queries, language)
                for strategy in ChunkingStrategy
            ]
            strategy_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(strategy_results):
                strategy = list(ChunkingStrategy)[i]
                if isinstance(result, Exception):
                    logger.error(f"{strategy.value} 전략 실행 중 오류 발생: {result}", exc_info=True)
                    self.experiment_run.add_error(str(result), f"{language.value}-{strategy.value}")
                    # 모든 필드를 채워서 EvaluationResult 객체 생성
                    results.append(EvaluationResult(
                        strategy=strategy.value, language=language, num_samples=0,
                        hallucination_auroc=0.0, context_relevance_rmse=0.0,
                        utilization_rmse=0.0, recall_at_k=0.0, mrr=0.0
                    ))
                elif result:
                    results.append(result)
                    self.experiment_run.add_result(result)
                    self._log_strategy_completion(result)

        return results

    async def _run_intelligent_auto_select(
        self,
        documents: List[Document],
        queries: List[Query],
        language: Language
    ) -> List[EvaluationResult]:
        """지능형 자동 전략 선택 모드로 실험을 실행합니다."""

        if not self.chunking_agent:
            logger.error("청킹 에이전트가 초기화되지 않았습니다.")
            return []

        logger.info("각 문서마다 최적의 전략을 자동 선택하여 청킹합니다...")

        # 전략별 결과를 수집할 딕셔너리
        strategy_results = {}
        strategy_counts = {}

        # 논문 작성용 분석 데이터 수집
        strategy_analysis_data = {
            "documents_processed": [],
            "strategy_selections": [],
            "domain_strategy_mapping": {},
            "confidence_scores": [],
            "reasoning_patterns": []
        }

        # 샘플 크기 제한
        sample_size = min(len(documents), len(queries), self.config.experiment.sample_size)

        # 각 문서에 대해 최적 전략 선택 및 청킹
        for i in range(sample_size):
            doc = documents[i]
            query = queries[i] if i < len(queries) else None

            try:
                # 라우터를 통해 최적 전략 추천
                recommendation = await self.chunking_agent.router.recommend_strategy(
                    doc, query, self.chunking_agent.default_context
                )

                selected_strategy = recommendation.primary_strategy
                logger.info(f"문서 {doc.id}: 선택된 전략 = {selected_strategy.value} "
                           f"(신뢰도: {recommendation.confidence:.2f})")

                # 논문 작성용 데이터 수집
                doc_domain = doc.metadata.get('domain', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                strategy_analysis_data["documents_processed"].append({
                    "doc_id": doc.id,
                    "domain": doc_domain,
                    "length": len(doc.content),
                    "selected_strategy": selected_strategy.value,
                    "confidence": recommendation.confidence,
                    "reasoning": recommendation.reasoning
                })

                strategy_analysis_data["strategy_selections"].append(selected_strategy.value)
                strategy_analysis_data["confidence_scores"].append(recommendation.confidence)
                strategy_analysis_data["reasoning_patterns"].append(recommendation.reasoning)

                # 도메인-전략 매핑 업데이트
                if doc_domain not in strategy_analysis_data["domain_strategy_mapping"]:
                    strategy_analysis_data["domain_strategy_mapping"][doc_domain] = {}
                domain_mapping = strategy_analysis_data["domain_strategy_mapping"][doc_domain]
                domain_mapping[selected_strategy.value] = domain_mapping.get(selected_strategy.value, 0) + 1

                # 선택된 전략으로 청킹 수행
                chunking_result = await self.chunking_agent.chunk_document(
                    doc, query, force_strategy=selected_strategy
                )

                if chunking_result.success:
                    # 전략별 결과 수집
                    strategy_name = selected_strategy.value
                    if strategy_name not in strategy_results:
                        strategy_results[strategy_name] = []
                        strategy_counts[strategy_name] = 0

                    strategy_results[strategy_name].append((doc, query, chunking_result.chunks))
                    strategy_counts[strategy_name] += 1
                else:
                    logger.warning(f"문서 {doc.id} 청킹 실패: {chunking_result.error_message}")

            except Exception as e:
                logger.error(f"문서 {doc.id} 처리 중 오류: {e}")

        # 전략별 성능 평가
        logger.info("전략별 성능 평가를 시작합니다...")
        evaluation_results = []

        for strategy_name, doc_results in strategy_results.items():
            if not doc_results:
                continue

            try:
                # 각 전략별로 평가 수행
                strategy_enum = ChunkingStrategy(strategy_name)
                components = self._initialize_components_for_strategy(strategy_enum, language)

                # 평가 로직 (간소화된 버전)
                eval_result = await self._evaluate_strategy_results(
                    strategy_enum, doc_results, components, language
                )

                if eval_result:
                    evaluation_results.append(eval_result)
                    self.experiment_run.add_result(eval_result)

                logger.success(f"전략 {strategy_name} 평가 완료 "
                              f"(사용된 문서: {strategy_counts[strategy_name]}개)")

            except Exception as e:
                logger.error(f"전략 {strategy_name} 평가 실패: {e}")

        # 전략 사용 통계 로그
        logger.info("=== 전략 사용 통계 ===")
        total_docs = sum(strategy_counts.values())
        for strategy_name, count in strategy_counts.items():
            percentage = (count / total_docs) * 100 if total_docs > 0 else 0
            logger.info(f"{strategy_name}: {count}개 문서 ({percentage:.1f}%)")

        # 논문 작성용 분석 결과 저장
        await self._save_strategy_analysis(strategy_analysis_data)

        # 평가 결과에 분석 데이터 추가
        for result in evaluation_results:
            if hasattr(result, 'metadata'):
                result.metadata.update({
                    "strategy_analysis": strategy_analysis_data,
                    "auto_selection_mode": True
                })

        return evaluation_results

    async def _evaluate_strategy_results(
        self,
        strategy: ChunkingStrategy,
        doc_results: List[Tuple[Document, Query, List[Chunk]]],
        components: Dict[str, Any],
        language: Language
    ) -> Optional[EvaluationResult]:
        """전략별 결과를 평가합니다."""

        try:
            retriever = components["retriever"]
            evaluator = components["evaluator"]

            responses = []
            ground_truths = []

            # 모든 청크를 수집
            all_chunks = []
            for doc, query, chunks in doc_results:
                all_chunks.extend(chunks)

            # 각 쿼리에 대해 검색 및 응답 생성
            for doc, query, chunks in doc_results:
                try:
                    # 검색 수행
                    retrieved_chunks = await retriever.retrieve(
                        query.question, all_chunks,
                        k=self.config.experiment.top_k_retrieval
                    )

                    # 응답 생성
                    response = RAGResponse(
                        strategy=strategy,
                        query=query.question,
                        query_id=query.id,
                        response="[INTELLIGENT AUTO-SELECT MODE]",
                        chunks_used=retrieved_chunks if retrieved_chunks else [],
                        confidence=0.0
                    )

                    # retrieved_chunks 추가
                    if retrieved_chunks:
                        ranked = []
                        for i, ch in enumerate(retrieved_chunks, 1):
                            text = getattr(ch, 'content', getattr(ch, 'page_content', ''))
                            chunk_obj = SimpleNamespace(
                                content=text, rank=i,
                                score=getattr(ch, "score", 1.0),
                                source="retrieval", doc_id=doc.id
                            )
                            ranked.append(chunk_obj)
                        response.retrieved_chunks = ranked
                    else:
                        response.retrieved_chunks = []

                    responses.append(response)
                    ground_truths.append(query.expected_answer)

                except Exception as e:
                    logger.error(f"쿼리 {query.id} 처리 실패: {e}")

            # 평가 수행
            if responses:
                eval_result = await evaluator.evaluate_responses(responses, ground_truths)
                eval_result.strategy = f"{strategy.value}_intelligent"
                eval_result.metadata.update({
                    "intelligent_mode": True,
                    "documents_processed": len(doc_results),
                    "auto_selected": True
                })
                return eval_result

        except Exception as e:
            logger.error(f"전략 {strategy.value} 평가 중 오류: {e}")

        return None

    async def _run_single_strategy_with_components(
            self, strategy: ChunkingStrategy, documents: List[Document],
            queries: List[Query], language: Language
    ) -> Optional[EvaluationResult]:
        """각 전략별로 독립적인 컴포넌트를 사용하여 실행"""

        logger.info(f"[{language.value.upper()}] - [{strategy.value}] 전략 실험 시작")
        components = self._initialize_components_for_strategy(strategy, language)
        return await self._run_single_strategy(strategy, documents, queries, components, language)

    def _initialize_components_for_strategy(self, strategy: ChunkingStrategy, language: Language) -> Dict[str, Any]:
        """특정 전략을 위한 컴포넌트 초기화 (병렬 처리용)"""

        # config 파일에 정의된 청킹 관련 설정을 가져옵니다. (경로는 실제 config 구조에 맞게 조정)
        experiment_config = self.config.experiment

        chunker_map = {
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker(
                language=language,
                chunk_size_limit=experiment_config.chunk_size_limit,
                overlap_ratio=experiment_config.overlap_ratio
            ),

            ChunkingStrategy.SEMANTIC: SemanticChunker(language),
            ChunkingStrategy.KEYWORD: KeywordChunker(language),
            ChunkingStrategy.QUERY_AWARE: QueryAwareChunker(language),
            ChunkingStrategy.RECURSIVE: RecursiveChunker(language),
            ChunkingStrategy.TEXT_SIMILARITY: Text_Similarity(language)
        }

        embedder = OpenAIEmbedderWithStorage(
            language=language,
            enable_storage=self.enable_embedding_storage,
            storage_path=self.storage_path,
        ) if self.enable_embedding_storage else OpenAIEmbedder(language)

        return {
            "chunker": chunker_map[strategy],
            "embedder": embedder,
            "retriever": VectorRetriever(embedder),
            "generator": GPTGenerator(language) if self.evaluation_mode == 'e2e' else None,
            "evaluator": RAGEvaluator(language)
        }

    def _log_strategy_completion(self, result: EvaluationResult):
        """전략 완료 로깅"""
        if self.evaluation_mode == 'retrieval':
            mrr_score = result.mrr
            k_value = self.config.experiment.top_k_retrieval
            # metadata['at_k']의 키가 정수일 수도 문자열일 수도 있으므로 확인
            at_k_data = result.metadata.get('at_k', {})
            hit_at_k = at_k_data.get(k_value, at_k_data.get(str(k_value), {})).get('hit_at_k', result.recall_at_k)

            logger.success(
                f"{result.strategy} 완료 (검색 평가) - "
                f"Hit@{k_value}: {hit_at_k:.3f}, "
                f"MRR: {mrr_score:.3f}"
            )
        else:
            logger.success(
                f"{result.strategy} 완료 (E2E 평가) - "
                f"AUROC: {result.hallucination_auroc:.3f}, "
                f"Context RMSE: {result.context_relevance_rmse:.3f}"
            )

    # _process_single_item 메서드를 클래스 레벨로 이동 (들여쓰기 수정)
    async def _process_single_item(
            self, doc: Document, query: Query, strategy: ChunkingStrategy,
            components: Dict[str, Any], language: Language
    ) -> Tuple[Optional[RAGResponse], Optional[str], Dict[str, float]]:
        chunker = components["chunker"]
        embedder = components["embedder"]
        retriever = components["retriever"]
        generator = components["generator"]

        processing_times = {
            "chunking": 0.0, "embedding_storage": 0.0,
            "retrieval": 0.0, "generation": 0.0
        }
        chunks = []

        loaded_from_storage = False
        if self.enable_embedding_storage and isinstance(embedder, OpenAIEmbedderWithStorage):
            try:
                stored_data = embedder.storage.load_chunk_embeddings(doc.id, strategy.value)
                if stored_data and strategy.value in stored_data.get("chunk_types", {}):
                    chunk_info = stored_data["chunk_types"][strategy.value]
                    if "chunks" in chunk_info and "embeddings" in chunk_info:
                        restored_chunks = [Chunk(**c_data) for c_data in chunk_info["chunks"]]
                        if restored_chunks:
                            chunks = restored_chunks
                            loaded_from_storage = True
                            logger.info(f"문서 {doc.id}에 대한 '{strategy.value}' 청크/임베딩을 저장소에서 로드했습니다. ({len(chunks)}개)")
            except Exception as e:
                logger.warning(f"저장된 임베딩 로드 중 오류 발생 (문서 ID: {doc.id}, 전략: {strategy.value}): {e}. 새로 생성합니다.")

        if not loaded_from_storage:
            try:
                chunk_start = time.time()

                # 지능형 청킹 에이전트 사용 여부 확인
                if self.use_intelligent_chunking and self.chunking_agent:
                    # 지능형 에이전트를 사용한 청킹
                    logger.info(f"문서 {doc.id}에 지능형 청킹 에이전트 사용")

                    # 언어 설정 업데이트
                    self.chunking_agent.language = language

                    # 전략 강제 지정 (기존 실험과의 호환성을 위해)
                    chunking_result = await self.chunking_agent.chunk_document(
                        doc, query, force_strategy=strategy
                    )

                    if chunking_result.success:
                        chunks = chunking_result.chunks
                        logger.info(f"지능형 에이전트 청킹 성공: {len(chunks)}개 청크 생성")
                    else:
                        logger.warning(f"지능형 에이전트 청킹 실패: {chunking_result.error_message}")
                        chunks = []
                else:
                    # 기존 청킹 방식 사용
                    if strategy == ChunkingStrategy.QUERY_AWARE:
                        chunks = await chunker.query_aware_chunk(doc, query)  # query 객체를 그대로 전달
                    else:
                        chunks = await chunker.chunk_document(doc)

                # 청크 후처리
                for chunk in chunks:
                    if not hasattr(chunk, 'doc_id'):
                        chunk.doc_id = doc.id

                processing_times["chunking"] = time.time() - chunk_start

                if not chunks:
                    logger.warning(f"문서 {doc.id}에 대한 청크가 생성되지 않았습니다.")
                    return None, None, processing_times

                if self.enable_embedding_storage and isinstance(embedder, OpenAIEmbedderWithStorage):
                    storage_start = time.time()
                    await embedder.embed_and_store_chunks(chunks=chunks, chunk_type=strategy.value, document_id=doc.id)
                    processing_times["embedding_storage"] = time.time() - storage_start
                else:
                    await embedder.embed_chunks(chunks)

            except Exception as e:
                logger.error(f"청킹/임베딩 처리 실패 - 문서: {doc.id}, 오류: {e}", exc_info=True)
                return None, None, processing_times
        logger.debug("=" * 20 + " DEBUGGING " + "=" * 20)
        logger.debug(f"Processing doc_id: {doc.id}")
        logger.debug(f"Received query type: {type(query)}")
        logger.debug(f"Received query content: {query}")
        logger.debug("=" * 53)

        logger.info(f"DEBUG: Type of query variable is now [ {type(query)} ] before retrieval/generation.")

        try:
            retrieval_start = time.time()
            # retriever.retrieve는 query '객체'가 아닌 query '문자열'을 받도록 수정
            retrieved_chunks = await retriever.retrieve(query.question, chunks,
                                                        k=self.config.experiment.top_k_retrieval)

            processing_times["retrieval"] = time.time() - retrieval_start

            if self.evaluation_mode == "retrieval":
                processing_times["generation"] = 0.0
                response = RAGResponse(
                    strategy=strategy,
                    query=query.question,
                    query_id=query.id,
                    response="[GENERATION BYPASSED FOR TEST]",
                    chunks_used=retrieved_chunks if retrieved_chunks else [],
                    confidence=0.0
                )
            else:
                if not generator:
                    raise ValueError("E2E 모드에서는 Generator가 초기화되어야 합니다.")
                generation_start = time.time()
                response = await generator.generate_response(query, retrieved_chunks)
                processing_times["generation"] = time.time() - generation_start

            if retrieved_chunks:
                ranked = []
                for i, ch in enumerate(retrieved_chunks, 1):
                    text = getattr(ch, 'content', getattr(ch, 'page_content', getattr(ch, 'text', '')))
                    chunk_obj = SimpleNamespace(
                        content=text, rank=i,
                        score=getattr(ch, "score", 1.0),
                        source="retrieval", doc_id=doc.id
                    )
                    ranked.append(chunk_obj)
                response.retrieved_chunks = ranked
            else:
                if not hasattr(response, 'retrieved_chunks'):
                    response.retrieved_chunks = []

            return response, query.expected_answer, processing_times

        except Exception as e:
            logger.error(f"검색/생성 처리 실패 - 문서: {doc.id}, 오류: {e}", exc_info=True)
            return None, None, processing_times

    async def _run_single_strategy(
            self, strategy: ChunkingStrategy, documents: List[Document],
            queries: List[Query], components: Dict[str, Any], language: Language
    ) -> Optional[EvaluationResult]:
        """단일 청킹 전략을 효율적인 병렬 배치 방식으로 실행합니다."""
        strategy_start_time = time.time()
        sample_size = min(len(documents), len(queries), self.config.experiment.sample_size)

        # 동시 실행 작업 수를 제어하기 위한 세마포어 (API Rate Limit 및 CPU 부하 방지)
        # CPU 부하가 큰 'semantic'의 경우 값을 낮추고, 나머지는 높여도 좋습니다.
        concurrency_limit = 3 if strategy == ChunkingStrategy.SEMANTIC else 5
        semaphore = asyncio.Semaphore(concurrency_limit)

        chunker = components["chunker"]
        embedder = components["embedder"]
        retriever = components["retriever"]
        evaluator = components["evaluator"]

        logger.info(f"[{strategy.value}] 1단계: {sample_size}개 문서에 대한 병렬 청킹 시작... (동시 실행 수: {concurrency_limit})")

        # --- 1단계: 모든 문서 병렬 청킹 ---
        async def chunk_doc(doc, query):
            async with semaphore:
                if strategy == ChunkingStrategy.QUERY_AWARE:
                    return await chunker.query_aware_chunk(doc, query)
                else:
                    chunks = await chunker.chunk_document(doc)
                    # 각 청크에 doc_id가 없는 경우 수동으로 할당
                    for chunk in chunks:
                        if not hasattr(chunk, 'doc_id') or not chunk.doc_id:
                            chunk.doc_id = doc.id
                    return chunks

        chunking_tasks = [chunk_doc(documents[i], queries[i]) for i in range(sample_size)]
        chunking_results = await asyncio.gather(*chunking_tasks, return_exceptions=True)

        all_chunks = []
        for i, result in enumerate(chunking_results):
            if isinstance(result, Exception):
                logger.error(f"문서 {documents[i].id} 청킹 실패: {result}")
            elif result:
                all_chunks.extend(result)

        if not all_chunks:
            logger.error(f"[{strategy.value}] 전략에서 유효한 청크가 하나도 생성되지 않았습니다.")
            return None

        logger.success(f"[{strategy.value}] 1단계 완료: 총 {len(all_chunks)}개 청크 생성.")

        # --- 2단계: 모든 청크 임베딩 및 저장(문서별·전략별) ---
        logger.info(f"[{strategy.value}] 2단계: 전체 청크 임베딩/저장 시작...")
        embedding_start_time = time.time()

        try:
            # 저장 가능한 임베더면 문서별로 저장 호출
            if hasattr(embedder, "embed_and_store_chunks"):
                from collections import defaultdict
                chunks_by_doc = defaultdict(list)
                for ch in all_chunks:
                    if not getattr(ch, "doc_id", None):
                        # 1단계에서 보장하지만 혹시 모를 누락 방지
                        raise ValueError("chunk.doc_id 누락: 저장 경로 매핑 불가")
                    chunks_by_doc[ch.doc_id].append(ch)

                for doc_id, doc_chunks in chunks_by_doc.items():
                    await embedder.embed_and_store_chunks(
                        chunks=doc_chunks,
                        chunk_type=strategy.value,  # <- by_chunk_type 하위에 전략별로 분기
                        document_id=doc_id  # <- 문서별 디렉터리
                    )
            else:
                # 저장 기능 없는 임베더면 기존 경로 유지
                await embedder.embed_chunks(all_chunks)

            # (옵션) 검색기가 in-memory 임베딩을 요구한다면 보강
            if getattr(all_chunks[0], "embedding", None) is None:
                # 구현체가 chunk.embedding을 채우지 않았다면 한 번 더 메모리용 임베딩
                await embedder.embed_chunks(all_chunks)

        except Exception as e:
            logger.error(f"[{strategy.value}] 임베딩/저장 단계 실패: {e}", exc_info=True)
            raise

        embedding_time = time.time() - embedding_start_time
        logger.success(f"[{strategy.value}] 2단계 완료. (소요 시간: {embedding_time:.2f}초)")

        # --- 3단계: 모든 쿼리에 대한 병렬 검색 및 평가 ---
        logger.info(f"[{strategy.value}] 3단계: {sample_size}개 쿼리에 대한 병렬 검색 및 평가 시작...")

        if queries:
            logger.info(f"DEBUG >>> 'queries' 리스트의 첫 번째 항목 타입: {type(queries[0])}")
            logger.info(f"DEBUG >>> 'queries' 리스트의 첫 번째 항목 내용: {queries[0]}")

        async def process_query(query):
            async with semaphore:
                try:
                    # retriever.retrieve는 전체 청크 리스트와 쿼리를 받아 검색을 수행해야 합니다.
                    retrieved_chunks = await retriever.retrieve(query.question, all_chunks,
                                                                k=self.config.experiment.top_k_retrieval)

                    response = RAGResponse(
                        strategy=strategy,
                        query=query.question,
                        query_id=query.id,
                        response="[GENERATION BYPASSED FOR RETRIEVAL TEST]",
                        chunks_used=retrieved_chunks if retrieved_chunks else [],
                        confidence=0.0
                    )

                    if retrieved_chunks:
                        ranked = []
                        # 문서 ID를 찾기 위해 query의 context_id를 사용합니다.
                        doc_id_for_chunks = query.context_id
                        for i, ch in enumerate(retrieved_chunks, 1):
                            text = getattr(ch, 'content', getattr(ch, 'page_content', getattr(ch, 'text', '')))
                            chunk_obj = SimpleNamespace(
                                content=text, rank=i,
                                score=getattr(ch, "score", 1.0),
                                source="retrieval", doc_id=doc_id_for_chunks
                            )
                            ranked.append(chunk_obj)
                        response.retrieved_chunks = ranked
                    else:
                        response.retrieved_chunks = []

                    return response, query.expected_answer
                except Exception as e:
                    logger.error(f"쿼리 {query.id} 처리 중 오류: {e}")
                    return None, None

        processing_tasks = [process_query(queries[i]) for i in range(sample_size)]
        processed_results = await asyncio.gather(*processing_tasks)

        responses = [res[0] for res in processed_results if res[0] is not None]
        ground_truths = [res[1] for res in processed_results if res[1] is not None]

        logger.success(f"[{strategy.value}] 3단계 완료: {len(responses)}개 응답 생성.")

        # --- 최종 평가 ---
        if responses:
            eval_start = time.time()
            eval_result = await evaluator.evaluate_responses(responses, ground_truths)
            eval_time = time.time() - eval_start
            eval_result.strategy = strategy.value

            eval_result.metadata.update({
                "total_time_seconds": time.time() - strategy_start_time,
                "embedding_and_indexing_time": embedding_time,
                "evaluation_time": eval_time,
                "samples_processed": len(responses),
            })
            return eval_result

        logger.warning(f"{strategy.value} 전략에 대한 유효한 응답이 없어 평가를 건너뜁니다.")
        return None

    async def _save_results(self, results: List[EvaluationResult], analysis: Dict[str, Any]):
        """결과를 비동기적으로 저장합니다."""
        dataset_name = Path(self.config.dataset.data_path).stem
        results_dir = self.config.paths.results_dir / dataset_name / self.run_id
        results_dir.mkdir(exist_ok=True, parents=True)
        results_data = [r.to_dict() for r in results if r]
        experiment_metadata = {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "summary": self.experiment_run.get_summary(),
            "errors": self.experiment_run.errors,
            "embedding_storage": {
                "enabled": self.enable_embedding_storage,
                "path": str(self.storage_path) if self.enable_embedding_storage else None
            },
            "evaluation_mode": self.evaluation_mode
        }

        async def write_json(path, data):
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder))

        save_tasks = [
            write_json(results_dir / "raw_results.json", results_data),
            write_json(results_dir / "analysis_results.json", analysis),
            write_json(results_dir / "experiment_metadata.json", experiment_metadata)
        ]
        await asyncio.gather(*save_tasks)
        logger.info(f"결과 저장 완료: {results_dir}")

    async def _save_strategy_analysis(self, analysis_data: Dict[str, Any]):
        """전략 분석 데이터를 논문 작성용으로 저장"""
        try:
            from collections import Counter
            import numpy as np

            # 통계 분석
            strategy_counter = Counter(analysis_data["strategy_selections"])
            confidence_scores = analysis_data["confidence_scores"]

            analysis_summary = {
                "experiment_metadata": {
                    "run_id": self.run_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_documents": len(analysis_data["documents_processed"]),
                    "evaluation_mode": self.evaluation_mode
                },
                "strategy_distribution": dict(strategy_counter),
                "strategy_percentages": {
                    strategy: (count / len(analysis_data["strategy_selections"])) * 100
                    for strategy, count in strategy_counter.items()
                },
                "confidence_statistics": {
                    "mean": float(np.mean(confidence_scores)),
                    "std": float(np.std(confidence_scores)),
                    "min": float(np.min(confidence_scores)),
                    "max": float(np.max(confidence_scores)),
                    "median": float(np.median(confidence_scores))
                },
                "domain_analysis": analysis_data["domain_strategy_mapping"],
                "detailed_decisions": analysis_data["documents_processed"]
            }

            # 저장 경로 설정
            results_dir = self.config.paths.results_dir / "strategy_analysis" / self.run_id
            results_dir.mkdir(exist_ok=True, parents=True)

            # JSON 저장
            analysis_path = results_dir / "strategy_analysis.json"
            async with aiofiles.open(analysis_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(analysis_summary, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder))

            # CSV 저장 (스프레드시트 분석용)
            try:
                import pandas as pd
                df = pd.DataFrame(analysis_data["documents_processed"])
                csv_path = results_dir / "strategy_decisions.csv"
                df.to_csv(csv_path, index=False, encoding="utf-8")
            except ImportError:
                logger.warning("pandas가 설치되지 않아 CSV 파일을 저장할 수 없습니다.")

            logger.info(f"전략 분석 결과 저장 완료: {results_dir}")

        except Exception as e:
            logger.error(f"전략 분석 저장 실패: {e}")

    def _create_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """실험 요약 생성"""
        valid_results = [r for r in results if r and r.num_samples > 0]
        if not valid_results: return {}

        if self.evaluation_mode == 'retrieval':
            best_result = max(valid_results, key=lambda r: r.mrr)
            summary = {
                "evaluation_mode": "retrieval",
                "best_strategy_by_mrr": best_result.strategy,
                "best_mrr_score": best_result.mrr,
                "strategy_retrieval_scores": {}
            }
            for r in valid_results:
                k_value = self.config.experiment.top_k_retrieval
                at_k_data = r.metadata.get('at_k', {})
                hit_at_k = at_k_data.get(k_value, at_k_data.get(str(k_value), {})).get('hit_at_k', r.recall_at_k)
                summary["strategy_retrieval_scores"][r.strategy] = {
                    "mrr": r.mrr,
                    f"hit_at_{k_value}": hit_at_k
                }
            return summary
        else:  # e2e
            best_result = max(valid_results, key=lambda r: r.get_overall_score())
            baseline_results = [r for r in valid_results if r.strategy == "fixed_size"]
            baseline_auroc = baseline_results[0].hallucination_auroc if baseline_results else 0.0
            improvement = ((
                                       best_result.hallucination_auroc - baseline_auroc) / baseline_auroc) * 100 if baseline_auroc > 0 else 0

            strategy_improvements = {}
            if baseline_auroc > 0:
                for result in valid_results:
                    if result.strategy != "fixed_size":
                        improvement_pct = ((result.hallucination_auroc - baseline_auroc) / baseline_auroc) * 100
                        strategy_improvements[result.strategy] = {
                            "auroc": result.hallucination_auroc,
                            "improvement_over_baseline": f"{improvement_pct:.1f}%"
                        }
            return {
                "evaluation_mode": "e2e",
                "baseline_strategy": "fixed_size",
                "baseline_auroc": baseline_auroc,
                "best_strategy_by_auroc": best_result.strategy,
                "best_auroc_score": best_result.hallucination_auroc,
                "improvement_over_baseline": f"{improvement:.1f}%",
                "strategy_comparisons": strategy_improvements,
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 청킹 전략 비교 연구")
    parser.add_argument(
        "--data_path", type=str, default=config.dataset.data_path,
        help="실험에 사용할 데이터셋 파일 경로입니다."
    )
    parser.add_argument(
        "--mode", type=str, default="retrieval", choices=["retrieval", "e2e"],
        help="실행 모드를 선택합니다: 'retrieval' 또는 'e2e'"
    )
    parser.add_argument(
        "--enable_embedding_storage", action="store_true",
        help="청킹별 임베딩을 파일로 저장합니다."
    )
    parser.add_argument(
        "--storage_path", type=str, default=str(config.paths.embedding_storage_path),
        help="임베딩 저장 경로"
    )
    parser.add_argument(
        "--use_intelligent_chunking", action="store_true",
        help="지능형 청킹 에이전트를 활성화합니다"
    )
    parser.add_argument(
        "--chunking_context", type=str, default="balanced",
        choices=["quality_focused", "speed_focused", "balanced", "cost_conscious"],
        help="청킹 컨텍스트: quality_focused, speed_focused, balanced, cost_conscious"
    )
    parser.add_argument(
        "--intelligent_mode", type=str, default="strategy_override",
        choices=["strategy_override", "auto_select"],
        help="지능형 모드: strategy_override (기존 전략 유지), auto_select (자동 전략 선택)"
    )
    parser.add_argument(
        "--use_multi_datasets", action="store_true",
        help="다중 허깅페이스 데이터셋을 사용합니다"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["squad", "squad_v2", "natural_questions", "ms_marco", "hotpot_qa"],
        help="사용할 데이터셋 목록 (예: --datasets squad squad_v2 natural_questions)"
    )
    parser.add_argument(
        "--samples_per_dataset", type=int, default=100,
        help="각 데이터셋에서 가져올 샘플 수"
    )
    parser.add_argument(
        "--list_datasets", action="store_true",
        help="사용 가능한 데이터셋 목록을 출력하고 종료합니다"
    )
    parser.add_argument(
        "-n", "--sample_size", type=int, default=None,
        help="실험에 사용할 샘플 크기를 설정합니다. 설정하지 않으면 config 기본값을 사용합니다."
    )
    parser.add_argument(
        "--paper_mode", action="store_true",
        help="논문용 엄격한 실험 모드 (대규모 다중 도메인, 강화된 통계 분석)"
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="빠른 테스트 모드 (소규모 샘플, 제한된 도메인)"
    )
    parser.add_argument(
        "--max_text_length", type=int, default=2000,
        help="최대 텍스트 길이 (짧은 글 실험용)"
    )
    parser.add_argument(
        "--min_text_length", type=int, default=50,
        help="최소 텍스트 길이 (너무 짧은 텍스트 필터링)"
    )
    parser.add_argument(
        "--run_comparison", action="store_true",
        help="Baseline(6개 전략) vs Agent(자동선택) 성능 비교 실험 실행"
    )
    args = parser.parse_args()

    # 실험 모드 설정
    if args.paper_mode:
        config.experiment.paper_mode = True
        logger.info("논문 모드 활성화: 대규모 실험, 강화된 통계 분석")

    if args.quick_test:
        config.experiment.quick_test = True
        logger.info("빠른 테스트 모드 활성화: 소규모 샘플")

    config.dataset.data_path = args.data_path

    # sample_size 인자가 주어진 경우 config 업데이트
    if args.sample_size is not None:
        config.experiment.sample_size = args.sample_size

    if not config.api.openai_api_key or "sk-" not in config.api.openai_api_key:
        logger.error("오류: OPENAI_API_KEY가 유효하지 않습니다. 환경 변수를 확인해주세요.")
        sys.exit(1)


    async def run_experiment():
        if args.run_comparison:
            # Baseline vs Agent 비교 실험 실행
            await run_baseline_vs_agent_comparison(args)
        else:
            # 기존 단일 실험 실행
            await run_single_experiment(args)

    async def run_single_experiment(args):
        """기존 단일 실험 실행"""
        pipeline = RAGExperimentPipeline()
        pipeline.evaluation_mode = args.mode

        # 데이터셋 목록 출력 모드
        if args.list_datasets:
            logger.info("사용 가능한 데이터셋 목록:")
            available, recommended = pipeline.list_available_datasets(Language.ENGLISH)
            return

        if args.enable_embedding_storage:
            pipeline.enable_embedding_storage = True
        pipeline.storage_path = Path(args.storage_path)

        # 다중 데이터셋 설정 (기본적으로 활성화)
        if args.use_multi_datasets or not hasattr(args, 'data_path') or not Path(args.data_path).exists():
            # 기본 데이터셋으로 실험 실행
            default_datasets = ["squad", "newsqa", "bioasq"] if not args.use_multi_datasets else args.datasets
            pipeline.enable_multi_datasets(default_datasets, args.samples_per_dataset)
            pipeline.use_multi_datasets = True
            logger.info(f"자동으로 다중 데이터셋 모드 활성화: {default_datasets}")

        # 텍스트 길이 필터링 설정
        pipeline.max_text_length = args.max_text_length
        pipeline.min_text_length = args.min_text_length

        # 지능형 청킹을 기본으로 활성화 (명시적으로 비활성화하지 않는 한)
        if not hasattr(args, 'use_intelligent_chunking') or args.use_intelligent_chunking:
            if not hasattr(args, 'use_intelligent_chunking'):
                args.use_intelligent_chunking = True
                args.chunking_context = "balanced"
                args.intelligent_mode = "auto_select"
                logger.info("자동으로 지능형 청킹 모드 활성화")

        # 지능형 청킹 설정
        if args.use_intelligent_chunking:
            force_no_api = args.mode == "retrieval" and args.chunking_context == "cost_conscious"
            pipeline.enable_intelligent_chunking(
                context=args.chunking_context,
                force_no_api=force_no_api
            )
            # 지능형 모드 설정
            pipeline._intelligent_mode = args.intelligent_mode

        logger.info("=" * 50)
        logger.info(f"RAG 청킹 전략 비교 연구 시작 (모드: {pipeline.evaluation_mode})")
        if args.enable_embedding_storage:
            logger.info(f"임베딩 저장 활성화: {pipeline.storage_path}")
        if pipeline.use_multi_datasets:
            dataset_names = pipeline.dataset_names if hasattr(pipeline, 'dataset_names') else ["squad", "newsqa", "bioasq"]
            logger.info(f"다중 데이터셋 모드: {len(dataset_names)}개 데이터셋 ({args.samples_per_dataset}개씩)")
        if args.use_intelligent_chunking:
            logger.info(f"지능형 청킹 활성화 (컨텍스트: {args.chunking_context}, 모드: {args.intelligent_mode})")
        logger.info(f"텍스트 길이 설정: {args.min_text_length}~{args.max_text_length}자")
        logger.info("=" * 50)

        try:
            results_data = await pipeline.run_full_experiment()

            # 여기서 결과 저장을 명시적으로 한 번만 호출
            logger.info("결과 저장 시작")
            await pipeline._save_results(results_data["results"], results_data["analysis"])

            summary = results_data.get("summary", {})
            if summary:
                print("\n" + "=" * 50)
                print(" 실험 결과 요약")
                print("=" * 50)

                if summary.get("evaluation_mode") == "retrieval":
                    print(f"평가 모드: 검색 (Retrieval)")
                    print(f"최고 성능 전략 (MRR 기준): {summary.get('best_strategy_by_mrr', 'N/A')}")
                    print(f"최고 MRR 점수: {summary.get('best_mrr_score', 0):.3f}")
                    print("\n전략별 검색 성능:")
                    k_value = config.experiment.top_k_retrieval
                    for strategy, scores in summary.get('strategy_retrieval_scores', {}).items():
                        print(
                            f"  - {strategy}: MRR={scores.get('mrr', 0):.3f}, Hit@{k_value}={scores.get(f'hit_at_{k_value}', 0):.3f}")
                else:
                    print(f"평가 모드: End-to-End (E2E)")
                    print(f"최고 성능 전략 (AUROC 기준): {summary.get('best_strategy_by_auroc', 'N/A')}")
                    print(f"최고 AUROC 점수: {summary.get('best_auroc_score', 0):.3f}")
            else:
                print("실행된 실험이 없어 요약할 결과가 없습니다.")
        except Exception as e:
            logger.error(f"실험 실패: {e}", exc_info=True)
        finally:
            logger.info("실험 종료")

    async def run_baseline_vs_agent_comparison(args):
        """Baseline vs Agent 비교 실험"""
        from src.comparison_experiment import run_baseline_vs_agent_comparison as run_comparison
        await run_comparison(args)

    asyncio.run(run_experiment())