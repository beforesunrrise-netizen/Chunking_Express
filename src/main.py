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
from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger
import aiofiles
from types import SimpleNamespace
import argparse

# Assuming config and other modules are available in the path
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
        self.run_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_run = ExperimentRun(
            run_id=self.run_id,
            config=self.config.to_dict(),
            start_time=datetime.now()
        )
        self.enable_embedding_storage = True
        self.storage_path = self.config.paths.embedding_storage_path
        self.evaluation_mode = "retrieval"

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

            logger.info("결과 저장 시작")
            await self._save_results(all_results, analysis_results)

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
            self.experiment_run.add_error(e, "전체 실험")
            raise

    async def run_language_experiment(self, language: Language) -> List[EvaluationResult]:
        """특정 언어에 대한 실험 실행"""
        results = []
        documents, queries = await self.data_processor.load_data(language)
        if not documents or not queries:
            logger.error(f"{language.value} 데이터셋 로드에 실패하여 실험을 중단합니다.")
            return []

        logger.info(f"{language.value} 데이터 로드 완료: {len(documents)}개 문서, {len(queries)}개 쿼리")

        components = self._initialize_components(language)

        for strategy in ChunkingStrategy:
            logger.info(f"[{language.value.upper()}] - [{strategy.value}] 전략 실험 시작")
            try:
                result = await self._run_single_strategy(
                    strategy, documents, queries, components, language
                )
                if result:
                    results.append(result)
                    self.experiment_run.add_result(result)
                    if self.evaluation_mode == 'retrieval':
                        mrr_score = result.mrr
                        k_value = self.config.experiment.top_k_retrieval
                        hit_at_k = result.metadata.get('at_k', {}).get(str(k_value), {}).get('hit_at_k',
                                                                                             result.recall_at_k)
                        logger.success(
                            f"{strategy.value} 완료 (검색 평가) - "
                            f"Hit@{k_value}: {hit_at_k:.3f}, "
                            f"MRR: {mrr_score:.3f}"
                        )
                    else:
                        logger.success(
                            f"{strategy.value} 완료 (E2E 평가) - "
                            f"AUROC: {result.hallucination_auroc:.3f}, "
                            f"Context RMSE: {result.context_relevance_rmse:.3f}"
                        )
            except Exception as e:
                logger.error(f"{strategy.value} 전략 실행 중 최상위 오류 발생: {e}", exc_info=True)
                self.experiment_run.add_error(e, f"{language.value}-{strategy.value}")
                results.append(EvaluationResult(strategy=strategy.value, language=language, num_samples=0))

        return results

    def _initialize_components(self, language: Language) -> Dict[str, Any]:
        """언어별 컴포넌트 초기화"""
        chunkers = {
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker(language),
            ChunkingStrategy.SEMANTIC: SemanticChunker(language),
            ChunkingStrategy.KEYWORD: KeywordChunker(language),
            ChunkingStrategy.QUERY_AWARE: QueryAwareChunker(language),
            ChunkingStrategy.RECURSIVE: RecursiveChunker(language),
            ChunkingStrategy.LANGCHAIN_SEMANTIC: EmbeddingSemanticChunker(language)
        }
        if self.enable_embedding_storage:
            embedder = OpenAIEmbedderWithStorage(
                language=language, enable_storage=True, storage_path=self.storage_path
            )
        else:
            embedder = OpenAIEmbedder(language)

        retriever = VectorRetriever(embedder)
        generator = GPTGenerator(language) if self.evaluation_mode == 'e2e' else None
        evaluator = RAGEvaluator(language)

        return {
            "chunkers": chunkers, "embedder": embedder, "retriever": retriever,
            "generator": generator, "evaluator": evaluator
        }

    async def _process_single_item(
            self, doc: Document, query: Query, strategy: ChunkingStrategy,
            components: Dict[str, Any], language: Language
    ) -> Tuple[RAGResponse, str, Dict[str, float]]:
        """단일 아이템(문서-쿼리 쌍)을 처리하고 결과와 처리 시간을 반환합니다."""
        chunker = components["chunkers"][strategy]
        embedder = components["embedder"]
        retriever = components["retriever"]
        generator = components["generator"]

        processing_times = {
            "chunking": 0.0, "embedding_storage": 0.0,
            "retrieval": 0.0, "generation": 0.0
        }

        try:
            # 1. 청킹
            chunk_start = time.time()
            if strategy == ChunkingStrategy.QUERY_AWARE:
                chunks = await chunker.query_aware_chunk(doc, query)
            else:
                chunks = await chunker.chunk_document(doc)
            for chunk in chunks:
                if not hasattr(chunk, 'doc_id'):
                    chunk.doc_id = doc.id
            processing_times["chunking"] = time.time() - chunk_start

            if not chunks:
                logger.warning(f"문서 {doc.id}에 대한 청크가 생성되지 않았습니다.")
                return None, None, processing_times

            # 2. 임베딩 및 저장
            if self.enable_embedding_storage and isinstance(embedder, OpenAIEmbedderWithStorage):
                storage_start = time.time()
                await embedder.embed_and_store_chunks(chunks=chunks, chunk_type=strategy.value, document_id=doc.id)
                processing_times["embedding_storage"] = time.time() - storage_start

            # 3. 검색
            retrieval_start = time.time()
            retrieved_chunks = await retriever.retrieve(query, chunks, k=self.config.experiment.top_k_retrieval)
            processing_times["retrieval"] = time.time() - retrieval_start

            if self.evaluation_mode == "retrieval":
                processing_times["generation"] = 0.0

                # [수정] 오류 메시지에 나온 모든 필수 인자를 채워서 RAGResponse 객체 생성
                response = RAGResponse(
                    strategy=strategy,
                    query=query.question,
                    query_id=query.id,
                    response="",  # 생성된 답변이 없으므로 빈 문자열
                    chunks_used=retrieved_chunks if retrieved_chunks else [],
                    confidence=0.0  # 생성된 답변이 없으므로 0.0
                )
            else:  # e2e mode
                if not generator:
                    raise ValueError("E2E 모드에서는 Generator가 초기화되어야 합니다.")
                generation_start = time.time()
                response = await generator.generate_response(query, retrieved_chunks)
                processing_times["generation"] = time.time() - generation_start

            # 평가용 raw Top-K 주입
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

                # RAGResponse 객체에 retrieved_chunks 속성을 할당
                # (클래스 정의에 따라 init=False 필드일 수 있음)
                response.retrieved_chunks = ranked
            else:
                if not hasattr(response, 'retrieved_chunks'):
                    response.retrieved_chunks = []

            return response, query.expected_answer, processing_times

        except Exception as e:
            logger.error(f"아이템 처리 실패 - 문서: {doc.id}, 오류: {e}", exc_info=True)
            return None, None, processing_times

    async def _run_single_strategy(
            self, strategy: ChunkingStrategy, documents: List[Document],
            queries: List[Query], components: Dict[str, Any], language: Language
    ) -> EvaluationResult:
        """단일 청킹 전략을 모든 샘플에 대해 동시에 실행합니다."""
        strategy_start_time = time.time()
        sample_size = min(len(documents), len(queries), self.config.experiment.sample_size)
        tasks = [
            self._process_single_item(documents[i], queries[i], strategy, components, language)
            for i in range(sample_size)
        ]
        results = await asyncio.gather(*tasks)

        responses, ground_truths, total_processing_times = [], [], {
            "chunking": [], "retrieval": [], "generation": [], "embedding_storage": []
        }
        for response, ground_truth, p_times in results:
            if response and ground_truth is not None:
                responses.append(response)
                ground_truths.append(ground_truth)
                for key, value in p_times.items():
                    if value is not None:
                        total_processing_times[key].append(value)

        if responses:
            eval_start = time.time()
            eval_result = await components["evaluator"].evaluate_responses(responses, ground_truths)
            eval_time = time.time() - eval_start
            eval_result.strategy = strategy.value

            if eval_result.metadata.get('at_k'):
                for k_str in eval_result.metadata['at_k']:
                    if 'k' not in eval_result.metadata['at_k'][k_str]:
                        eval_result.metadata['at_k'][k_str]['k'] = int(k_str)

            eval_result.metadata.update({
                "total_time_seconds": time.time() - strategy_start_time,
                "avg_chunking_time": np.mean(total_processing_times["chunking"]) if total_processing_times[
                    "chunking"] else 0,
                "avg_retrieval_time": np.mean(total_processing_times["retrieval"]) if total_processing_times[
                    "retrieval"] else 0,
                "avg_generation_time": np.mean(total_processing_times["generation"]) if total_processing_times[
                    "generation"] else 0,
                "avg_embedding_storage_time": np.mean(total_processing_times["embedding_storage"]) if
                total_processing_times["embedding_storage"] else 0,
                "evaluation_time": eval_time,
                "samples_processed": len(responses),
                "embedding_storage_enabled": self.enable_embedding_storage
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
            write_json(results_dir / "100_raw_results.json", results_data),
            write_json(results_dir / "analysis_results.json", analysis),
            write_json(results_dir / "experiment_metadata.json", experiment_metadata)
        ]
        await asyncio.gather(*save_tasks)
        logger.info(f"결과 저장 완료: {results_dir}")

    def _create_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """실험 요약 생성"""
        valid_results = [r for r in results if r]
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
                hit_at_k = r.metadata.get('at_k', {}).get(str(k_value), {}).get('hit_at_k', r.recall_at_k)
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


class DataProcessor:
    async def load_data(self, language: Language) -> Tuple[List[Document], List[Query]]:
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


class StatisticalAnalyzer:
    def analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        return {"descriptive_stats": {}, "statistical_tests": {}}


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
    args = parser.parse_args()

    config.dataset.data_path = args.data_path

    if not config.api.openai_api_key or "sk-" not in config.api.openai_api_key:
        logger.error("오류: OPENAI_API_KEY가 유효하지 않습니다. 환경 변수를 확인해주세요.")
        sys.exit(1)


    async def run_experiment():
        pipeline = RAGExperimentPipeline()
        pipeline.evaluation_mode = args.mode

        if args.enable_embedding_storage:
            pipeline.enable_embedding_storage = True
        pipeline.storage_path = Path(args.storage_path)

        logger.info("=" * 50)
        logger.info(f"RAG 청킹 전략 비교 연구 시작 (모드: {pipeline.evaluation_mode})")
        if args.enable_embedding_storage:
            logger.info(f"임베딩 저장 활성화: {pipeline.storage_path}")
        logger.info("=" * 50)

        try:
            results_data = await pipeline.run_full_experiment()
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


    asyncio.run(run_experiment())