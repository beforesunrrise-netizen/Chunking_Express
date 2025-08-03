

import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger
from config import (
    config, Language, ChunkingStrategy, EnsembleMethod
)

# Use relative imports instead of absolute imports
from data_structures import (
    Document, Query, Chunk, RAGResponse,
    EvaluationResult, ExperimentRun
)

# 청킹 전략
from chunkers import (
    SemanticChunker, KeywordChunker, QueryAwareChunker, FixedSizeChunker, RecursiveChunker, EmbeddingSemanticChunker
)

# 임베딩 및 검색
from embedders import OpenAIEmbedder
from retrievers import VectorRetriever

# 생성 및 평가
from generators import GPTGenerator
from evaluators import RAGEvaluator

# 앙상블
from ensembles import (
    VotingEnsemble, RerankingEnsemble, FusionEnsemble
)

# 데이터 처리 및 통계
from data_processor import DataProcessor
from statistical_analyzer import StatisticalAnalyzer
from embedders.openai_embedder import OpenAIEmbedderWithStorage
from storage.chunk_storage import ChunkEmbeddingStorage


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


class RAGExperimentPipeline:
    """RAG 실험 파이프라인"""

    def __init__(self):
        self.config = config
        self.data_processor = DataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        # self.visualizer = ResultsVisualizer() <- 시각화 객체 생성 제거

        # 실험 실행 정보
        self.run_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_run = ExperimentRun(
            run_id=self.run_id,
            config=self.config.to_dict(),
            start_time=datetime.now()
        )

        # 청킹 임베딩 저장소 초기화 (새로 추가)
        self.enable_embedding_storage = True  # 기본값은 False
        self.storage_path = self.config.paths.embedding_storage_path

    async def run_full_experiment(self) -> Dict[str, Any]:
        """전체 실험 실행"""
        logger.info(f"실험 시작: {self.run_id}")

        try:
            # 1. 언어별 실험 실행
            all_results = []

            # 실험 (영어 데이터셋으로 고정)
            logger.info("영어 데이터셋 실험 시작")
            en_results = await self.run_language_experiment(Language.ENGLISH)
            all_results.extend(en_results)

            analysis_results = {}  # 기본값으로 초기화
            try:
                # 2. 통계 분석
                logger.info("통계 분석 시작")
                analysis_results = self.statistical_analyzer.analyze_results(all_results)

                # 3. 결과 저장
                logger.info("결과 저장 시작")
                self._save_results(all_results, analysis_results)

            except Exception as e:
                logger.error(f"통계 분석 또는 결과 저장 중 치명적인 오류 발생: {e}")
                logger.error("오류가 발생했지만, 현재까지의 원시 결과라도 저장 시도합니다.")
                # 에러가 나더라도 원시(raw) 결과는 저장하도록 처리
                self._save_results(all_results, {"analysis_error": str(e)})

            # 5. 실험 완료
            self.experiment_run.end_time = datetime.now()

            logger.info(f"실험 완료: {self.run_id}")

            return {
                "run_id": self.run_id,
                "results": all_results,
                "analysis": analysis_results,
                # "visualizations": visualizations, <- 반환값에서 제거
                "summary": self._create_summary(all_results, analysis_results)
            }

        except Exception as e:
            logger.error(f"실험 실패: {e}")
            self.experiment_run.add_error(e, "전체 실험")
            raise

    async def run_language_experiment(self, language: Language) -> List[EvaluationResult]:
        """특정 언어에 대한 실험 실행"""
        results = []

        # 1. 데이터 로드
        documents, queries = await self.data_processor.load_data(language)
        logger.info(f"{language.value} 데이터 로드 완료: {len(documents)}개 문서, {len(queries)}개 쿼리")

        # 2. 컴포넌트 초기화
        components = await self._initialize_components(language)

        # 3. 개별 청킹 전략 실험 (Semantic, Keyword, Query-Aware)
        for strategy in ChunkingStrategy:
            logger.info(f"{language.value} - {strategy.value} 전략 실험 시작")
            try:
                result = await self._run_single_strategy(
                    strategy, documents, queries, components, language
                )
                results.append(result)
                self.experiment_run.add_result(result)
                logger.info(
                    f"{strategy.value} 완료 - "
                    f"AUROC: {result.hallucination_auroc:.3f}, "
                    f"Context RMSE: {result.context_relevance_rmse:.3f}"
                )
            except Exception as e:
                logger.error(f"{strategy.value} 전략 실패: {e}")
                self.experiment_run.add_error(e, f"{language.value}-{strategy.value}")

        # --- 수정된 부분: 앙상블 전략 실험을 주석 처리하여 비활성화 ---
        # # 4. 앙상블 전략 실험
        # for ensemble_method in EnsembleMethod:
        #     logger.info(f"{language.value} - {ensemble_method.value} 앙상블 실험 시작")
        #     try:
        #         result = await self._run_ensemble_strategy(
        #             ensemble_method, documents, queries, components, language
        #         )
        #         results.append(result)
        #         self.experiment_run.add_result(result)
        #         logger.info(
        #             f"{ensemble_method.value} 앙상블 완료 - "
        #             f"AUROC: {result.hallucination_auroc:.3f}"
        #         )
        #     except Exception as e:
        #         logger.error(f"{ensemble_method.value} 앙상블 실패: {e}")
        #         self.experiment_run.add_error(e, f"{language.value}-ensemble-{ensemble_method.value}")
        # --- 수정 완료 ---

        return results

    async def _initialize_components(self, language: Language) -> Dict[str, Any]:
        """언어별 컴포넌트 초기화"""
        # 청킹 전략
        chunkers = {
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker(language),
            ChunkingStrategy.SEMANTIC: SemanticChunker(language),
            ChunkingStrategy.KEYWORD: KeywordChunker(language),
            ChunkingStrategy.QUERY_AWARE: QueryAwareChunker(language),
            ChunkingStrategy.RECURSIVE : RecursiveChunker(language),
            ChunkingStrategy.LANGCHAIN_SEMANTIC : EmbeddingSemanticChunker(language)
        }

        # 임베더 선택 - 저장 기능 활성화 여부에 따라
        if self.enable_embedding_storage:
            embedder = OpenAIEmbedderWithStorage(
                language=language,
                enable_storage=True,
                storage_path=self.storage_path
            )
        else:
            embedder = OpenAIEmbedder(language)

        retriever = VectorRetriever(embedder)
        generator = GPTGenerator(language)
        evaluator = RAGEvaluator(language)
        ensembles = {
            EnsembleMethod.VOTING: VotingEnsemble("weighted"),
            EnsembleMethod.RERANKING: RerankingEnsemble("combined"),
            EnsembleMethod.FUSION: FusionEnsemble(language)
        }
        return {
            "chunkers": chunkers, "embedder": embedder, "retriever": retriever,
            "generator": generator, "evaluator": evaluator, "ensembles": ensembles
        }

    async def _run_single_strategy(
            self,
            strategy: ChunkingStrategy,
            documents: List[Document],
            queries: List[Query],
            components: Dict[str, Any],
            language: Language
    ) -> EvaluationResult:
        """단일 청킹 전략 실행 (시간 측정 및 임베딩 저장 포함)"""
        import time
        strategy_start_time = time.time()

        chunker = components["chunkers"][strategy]
        embedder = components["embedder"]
        retriever = components["retriever"]
        generator = components["generator"]
        evaluator = components["evaluator"]

        responses = []
        ground_truths = []
        sample_size = min(len(documents), len(queries), self.config.experiment.sample_size)

        # 개별 처리 시간 기록
        processing_times = {
            "chunking": [],
            "retrieval": [],
            "generation": [],
            "embedding_storage": []  # 새로 추가
        }

        for i in range(sample_size):
            doc, query = documents[i], queries[i]
            try:
                # 청킹 시간 측정
                chunk_start = time.time()
                if strategy == ChunkingStrategy.QUERY_AWARE:
                    chunks = await chunker.query_aware_chunk(doc, query)
                else:
                    chunks = await chunker.chunk_document(doc)

                # 청크에 doc_id 속성 추가 (새로 추가된 부분)
                for chunk in chunks:
                    if not hasattr(chunk, 'doc_id'):
                        chunk.doc_id = doc.id

                processing_times["chunking"].append(time.time() - chunk_start)

                if not chunks:
                    continue

                # 임베딩 저장 (저장 기능이 활성화된 경우)
                if self.enable_embedding_storage and isinstance(embedder, OpenAIEmbedderWithStorage):
                    storage_start = time.time()

                    logger.info(f"임베딩 저장 시작 - strategy: {strategy.value}, chunks: {len(chunks)}")
                    logger.info(f"저장 메타데이터: query_id={query.id if strategy == ChunkingStrategy.QUERY_AWARE else None}")

                    # 임베딩 생성 및 저장
                    embeddings = await embedder.embed_and_store_chunks(
                        chunks=chunks,
                        chunk_type=strategy.value,
                        document_id=doc.id,
                        additional_metadata={
                            "run_id": self.run_id,
                            "language": language.value,
                            "query_id": query.id if strategy == ChunkingStrategy.QUERY_AWARE else None
                        },
                        force_recompute=True  # 강제로 재계산 및 저장
                    )
                    logger.info(f"임베딩 저장 완료 - embeddings: {len(embeddings) if embeddings else 0}")

                    processing_times["embedding_storage"].append(time.time() - storage_start)

                # 검색 시간 측정
                retrieval_start = time.time()
                retrieved_chunks = await retriever.retrieve(query, chunks, k=self.config.experiment.top_k_retrieval)
                processing_times["retrieval"].append(time.time() - retrieval_start)

                # 생성 시간 측정
                generation_start = time.time()
                response = await generator.generate_response(query, retrieved_chunks)
                processing_times["generation"].append(time.time() - generation_start)

                responses.append(response)
                ground_truths.append(query.expected_answer)

            except Exception as e:
                logger.exception(f"처리 실패 - 문서: {doc.id}, 오류 발생")

        if responses:
            eval_start = time.time()
            eval_result = await evaluator.evaluate_responses(responses, ground_truths)
            eval_time = time.time() - eval_start

            eval_result.strategy = strategy.value

            # 시간 정보를 metadata에 추가
            eval_result.metadata.update({
                "total_time_seconds": time.time() - strategy_start_time,
                "avg_chunking_time": np.mean(processing_times["chunking"]) if processing_times["chunking"] else 0,
                "avg_retrieval_time": np.mean(processing_times["retrieval"]) if processing_times["retrieval"] else 0,
                "avg_generation_time": np.mean(processing_times["generation"]) if processing_times["generation"] else 0,
                "avg_embedding_storage_time": np.mean(processing_times["embedding_storage"]) if processing_times[
                    "embedding_storage"] else 0,
                "evaluation_time": eval_time,
                "samples_processed": len(responses),
                "embedding_storage_enabled": self.enable_embedding_storage
            })

            # 상세 로그 출력
            logger.info(
                f"{strategy.value} 완료 - "
                f"총 시간: {eval_result.metadata['total_time_seconds']:.2f}초, "
                f"처리 샘플: {len(responses)}개, "
                f"평균 처리 시간: {eval_result.metadata['total_time_seconds'] / len(responses):.2f}초/샘플"
            )

            if self.enable_embedding_storage:
                logger.info(
                    f"임베딩 저장 시간: {eval_result.metadata['avg_embedding_storage_time']:.2f}초/샘플"
                )

            return eval_result

        return EvaluationResult(strategy=strategy.value, language=language, num_samples=0)

    async def _run_ensemble_strategy(
            self,
            ensemble_method: EnsembleMethod,
            documents: List[Document],
            queries: List[Query],
            components: Dict[str, Any],
            language: Language
    ) -> EvaluationResult:
        """앙상블 전략 실행"""
        ensemble = components["ensembles"][ensemble_method]
        evaluator = components["evaluator"]

        ensemble_responses = []
        ground_truths = []
        sample_size = min(len(documents), len(queries), self.config.experiment.sample_size)

        for i in range(sample_size):
            doc, query = documents[i], queries[i]
            try:
                strategy_responses = []
                for strategy in ChunkingStrategy:
                    chunker = components["chunkers"][strategy]
                    retriever = components["retriever"]
                    generator = components["generator"]
                    if strategy == ChunkingStrategy.QUERY_AWARE:
                        chunks = await chunker.query_aware_chunk(doc, query)
                    else:
                        chunks = await chunker.chunk_document(doc)
                    if chunks:
                        retrieved_chunks = await retriever.retrieve(query, chunks,
                                                                    k=self.config.experiment.top_k_retrieval)
                        response = await generator.generate_response(query, retrieved_chunks)
                        strategy_responses.append(response)
                if strategy_responses:
                    ensemble_response = await ensemble.combine_responses(strategy_responses)
                    ensemble_responses.append(ensemble_response)
                    ground_truths.append(query.expected_answer)
            except Exception as e:
                logger.error(f"앙상블 처리 실패 - 문서: {doc.id}, 오류: {e}")

        if ensemble_responses:
            eval_result = await evaluator.evaluate_responses(ensemble_responses, ground_truths)
            eval_result.strategy = f"ensemble_{ensemble_method.value}"
            eval_result.ensemble_method = ensemble_method
            return eval_result
        return EvaluationResult(strategy=f"ensemble_{ensemble_method.value}", language=language, num_samples=0,
                                ensemble_method=ensemble_method)

    def _save_results(
            self,
            results: List[EvaluationResult],
            analysis: Dict[str, Any]
    ):
        """결과 저장 (임베딩 저장소 통계 포함)"""
        # 데이터셋 파일 경로에서 이름만 추출 (예: "squad_train_100_random")
        dataset_name = Path(self.config.dataset.data_path).stem

        # 데이터셋 이름으로 된 폴더를 포함한 결과 저장 경로 생성
        results_dir = self.config.paths.results_dir / dataset_name / self.run_id
        results_dir.mkdir(exist_ok=True, parents=True)

        # 1. 원시 결과 저장
        results_data = [r.to_dict() for r in results]
        with open(results_dir / "raw_results.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

        # 2. 분석 결과 저장
        with open(results_dir / "analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

        # 3. 임베딩 저장소 통계 저장 (새로 추가)
        if self.enable_embedding_storage:
            from src.storage.chunk_storage import ChunkEmbeddingStorage
            storage = ChunkEmbeddingStorage(self.storage_path)
            storage_stats = storage.get_statistics()
            with open(results_dir / "embedding_storage_stats.json", "w", encoding="utf-8") as f:
                json.dump(storage_stats, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

            logger.info(f"임베딩 저장소 통계: {storage_stats['total_chunks']}개 청크, "
                        f"{storage_stats['storage_info']['total_size_mb']:.2f} MB")

        # 4. 실험 메타데이터 저장
        experiment_metadata = {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "summary": self.experiment_run.get_summary(),
            "errors": self.experiment_run.errors,
            "embedding_storage": {
                "enabled": self.enable_embedding_storage,
                "path": self.storage_path if self.enable_embedding_storage else None
            }
        }
        with open(results_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
            json.dump(experiment_metadata, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

        logger.info(f"결과 저장 완료: {results_dir}")

    def _create_summary(
            self,
            results: List[EvaluationResult],
            analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """실험 요약 생성"""
        if not results: return {}

        best_result = max(results, key=lambda r: r.get_overall_score())
        en_results = [r for r in results if r.language == Language.ENGLISH]
        en_avg_auroc = np.mean([r.hallucination_auroc for r in en_results]) if en_results else 0

        # Baseline (fixed_size) 성능 찾기
        baseline_results = [r for r in results if r.strategy == "fixed_size"]
        baseline_auroc = baseline_results[0].hallucination_auroc if baseline_results else 0.57  # 기본값 유지

        # 최고 성능과 baseline 비교
        improvement = ((
                               best_result.hallucination_auroc - baseline_auroc) / baseline_auroc) * 100 if baseline_auroc else 0

        # 각 전략별 baseline 대비 개선율 계산
        strategy_improvements = {}
        for result in results:
            if result.strategy != "fixed_size" and baseline_auroc > 0:
                improvement_pct = ((result.hallucination_auroc - baseline_auroc) / baseline_auroc) * 100
                strategy_improvements[result.strategy] = {
                    "auroc": result.hallucination_auroc,
                    "improvement_over_baseline": f"{improvement_pct:.1f}%"
                }

        summary = {
            "baseline_strategy": "fixed_size",
            "baseline_auroc": baseline_auroc,
            "best_strategy": best_result.strategy,
            "best_language": best_result.language.value,
            "best_auroc": best_result.hallucination_auroc,
            "improvement_over_baseline": f"{improvement:.1f}%",
            "strategy_comparisons": strategy_improvements,  # 새로 추가
            "language_performance": {
                "english": {"avg_auroc": en_avg_auroc, "num_experiments": len(en_results)},
                "korean": {"avg_auroc": 0, "num_experiments": 0}  # 현재 실험에서 제외
            },
            "statistical_significance": analysis.get("statistical_tests", {}),
            "total_experiments": len(results),
            "total_duration": self.experiment_run.get_duration()
        }
        return summary

class DataProcessor:
    """데이터 처리 클래스"""

    def __init__(self):
        self.logger = logger

    async def load_data(self, language: Language) -> Tuple[List[Document], List[Query]]:
        """데이터 로드"""
        data_path = config.paths.data_dir / config.dataset.data_path
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            documents, queries = [], []
            for i, item in enumerate(data[:config.experiment.sample_size]):
                doc_id = str(i)
                documents.append(Document(id=doc_id, content=item["context"], language=language))
                queries.append(Query(id=doc_id, question=item["question"], language=language,
                                     expected_answer=item.get("answer", ""), context_id=doc_id))
            return documents, queries
        except Exception as e:
            self.logger.error(f"데이터 로드 실패: {e}")
            return [], []


class StatisticalAnalyzer:
    """통계 분석 클래스"""

    def analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """결과 통계 분석"""
        from scipy import stats
        return {
            "descriptive_stats": self._calculate_descriptive_stats(results),
            "statistical_tests": self._perform_statistical_tests(results)
        }

    def _calculate_descriptive_stats(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """기술 통계 계산"""
        stats_by_strategy = {}
        for result in results:
            strategy = result.strategy
            if strategy not in stats_by_strategy:
                stats_by_strategy[strategy] = {"auroc_scores": [], "context_rmse_scores": [],
                                               "utilization_rmse_scores": []}
            stats_by_strategy[strategy]["auroc_scores"].append(result.hallucination_auroc)
            stats_by_strategy[strategy]["context_rmse_scores"].append(result.context_relevance_rmse)
            stats_by_strategy[strategy]["utilization_rmse_scores"].append(result.utilization_rmse)

        descriptive_stats = {}
        for strategy, scores in stats_by_strategy.items():
            descriptive_stats[strategy] = {
                "auroc": {"mean": np.mean(scores["auroc_scores"]), "std": np.std(scores["auroc_scores"])},
                "context_rmse": {"mean": np.mean(scores["context_rmse_scores"]),
                                 "std": np.std(scores["context_rmse_scores"])},
                "utilization_rmse": {"mean": np.mean(scores["utilization_rmse_scores"]),
                                     "std": np.std(scores["utilization_rmse_scores"])}
            }
        return descriptive_stats

    def _perform_statistical_tests(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """통계적 검정"""
        from scipy import stats
        tests = {}
        baseline_auroc = 0.57
        proposed_auroc = [r.hallucination_auroc for r in results if "ensemble" in r.strategy]
        if proposed_auroc:
            t_stat, p_value = stats.ttest_1samp(proposed_auroc, baseline_auroc)
            tests["baseline_comparison"] = {
                "test": "one_sample_t_test", "baseline": baseline_auroc, "proposed_mean": np.mean(proposed_auroc),
                "t_statistic": t_stat, "p_value": p_value, "significant": p_value < 0.05,
                "improvement": ((np.mean(
                    proposed_auroc) - baseline_auroc) / baseline_auroc) * 100 if baseline_auroc else 0
            }
        return tests


async def main():
    """메인 실행 함수"""
    logger.info("=" * 50)
    logger.info("RAG 청킹 전략 비교 연구 시작 (Baseline 포함)")
    logger.info("=" * 50)

    pipeline = RAGExperimentPipeline()
    try:
        results = await pipeline.run_full_experiment()
        print("\n" + "=" * 50)
        print("실험 결과 요약")
        print("=" * 50)
        summary = results["summary"]
        if summary:
            print(f"Baseline 전략: {summary['baseline_strategy']}")
            print(f"Baseline AUROC: {summary['baseline_auroc']:.3f}")
            print(f"\n최고 성능 전략: {summary['best_strategy']} ({summary['best_language']})")
            print(f"최고 AUROC: {summary['best_auroc']:.3f}")
            print(f"Baseline 대비 개선율: {summary['improvement_over_baseline']}")

            # 각 전략별 성능 비교 출력
            if summary.get('strategy_comparisons'):
                print(f"\n전략별 Baseline 대비 성능:")
                for strategy, perf in summary['strategy_comparisons'].items():
                    print(f"  - {strategy}: {perf['auroc']:.3f} ({perf['improvement_over_baseline']})")

            print("\n언어별 평균 성능:")
            print(f"  - 영어: {summary['language_performance']['english']['avg_auroc']:.3f}")
            print(f"\n총 실험 수: {summary['total_experiments']}")
            print(f"실행 시간: {summary['total_duration']:.1f}초")
            print("\n결과 파일이 다음 위치에 저장되었습니다:")
            print(f"  {config.paths.results_dir / results['run_id']}")
        else:
            print("실행된 실험이 없어 요약할 결과가 없습니다.")
    except Exception as e:
        logger.error(f"실험 실패: {e}")
        raise
    finally:
        logger.info("실험 종료")


async def search_embeddings_demo():
    """저장된 임베딩에서 유사 청크 검색 데모"""
    storage = ChunkEmbeddingStorage("./src/data")
    embedder = OpenAIEmbedderWithStorage(
        language=Language.ENGLISH,
        storage_path="./src/data"
    )

    # 검색 쿼리
    query = "What is the main concept discussed in this document?"

    logger.info(f"\n=== 임베딩 검색 데모 ===")
    logger.info(f"검색 쿼리: {query}")

    # 유사 청크 검색
    similar_chunks = await embedder.search_similar_chunks(
        query=query,
        top_k=5,
        threshold=0.7
    )

    # 결과 출력
    logger.info(f"\n검색 결과 (상위 5개):")
    for i, result in enumerate(similar_chunks, 1):
        logger.info(f"\n{i}. 문서: {result['document_id']}, 청킹 타입: {result['chunk_type']}")
        logger.info(f"   유사도: {result['similarity']:.3f}")
        logger.info(f"   내용: {result['chunk']['content'][:200]}...")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="RAG 청킹 전략 비교 연구")
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.dataset.data_path,
        help=f"실험에 사용할 데이터셋 파일 경로입니다. (기본값: {config.dataset.data_path})"
    )

    # 새로운 옵션 추가
    parser.add_argument(
        "--enable_embedding_storage",
        action="store_true",
        help="청킹별 임베딩을 파일로 저장합니다."
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default=config.paths.embedding_storage_path,
        help=f"임베딩 저장 경로 (기본값: {config.paths.embedding_storage_path})"
    )

    args = parser.parse_args()

    config.dataset.data_path = args.data_path
    logger.info(f"실험에 사용될 데이터셋: {config.dataset.data_path}")

    if not config.api.openai_api_key:
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        exit(1)


    async def run_experiment():
        # ◀◀ [수정] 파이프라인 설정을 실행 함수 내부로 이동
        pipeline = RAGExperimentPipeline()
        pipeline.enable_embedding_storage = args.enable_embedding_storage
        pipeline.storage_path = Path(args.storage_path)  # 경로를 Path 객체로 변환

        logger.info("=" * 50)
        logger.info("RAG 청킹 전략 비교 연구 시작")
        if args.enable_embedding_storage:
            logger.info(f"임베딩 저장 활성화: {pipeline.storage_path}")
        logger.info("=" * 50)

        try:
            results = await pipeline.run_full_experiment()
            summary = results.get("summary", {})
            if summary:
                print("\n" + "=" * 50)
                print(" 실험 결과 요약")
                print("=" * 50)
                # ... (결과 요약 출력) ...
                print(f"\n결과 파일이 다음 위치에 저장되었습니다:")
                print(f"  {config.paths.results_dir / results['run_id']}")
        except Exception as e:
            logger.error(f"실험 실패: {e}", exc_info=True)
        finally:
            logger.info("실험 종료")


    asyncio.run(run_experiment())