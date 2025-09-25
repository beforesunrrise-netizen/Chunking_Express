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


class DataProcessor:
    def __init__(self):
        self.multi_loader = MultiDatasetLoader()

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


class BaselineExperimentPipeline:
    """Baseline 실험 파이프라인 - 모든 전략을 병렬로 실행"""

    def __init__(self):
        self.config = config
        self.data_processor = DataProcessor()
        self.run_id = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_run = ExperimentRun(
            run_id=self.run_id,
            config=self.config.to_dict(),
            start_time=datetime.now()
        )
        self.enable_embedding_storage = True
        self.storage_path = self.config.paths.embedding_storage_path
        self.evaluation_mode = "retrieval"

        # 다중 데이터셋 설정
        self.dataset_names = []
        self.samples_per_dataset = 100

        # 텍스트 길이 필터링 설정 (짧은 글 실험용)
        self.max_text_length = 2000
        self.min_text_length = 50

    def enable_multi_datasets(self, dataset_names: List[str], samples_per_dataset: int = 100):
        """다중 데이터셋 모드를 활성화합니다."""
        self.dataset_names = dataset_names
        self.samples_per_dataset = samples_per_dataset
        logger.info(f"Baseline 다중 데이터셋 모드: {len(dataset_names)}개 데이터셋, 각 {samples_per_dataset}개 샘플")

    async def run_baseline_experiment(self) -> Dict[str, Any]:
        """Baseline 실험 실행 - 모든 6개 전략을 병렬로 실행"""
        logger.info(f"📊 Baseline 실험 시작: {self.run_id} (모드: {self.evaluation_mode})")
        logger.info("모든 청킹 전략을 병렬로 실행합니다...")

        try:
            all_results = []

            # 데이터 로드
            logger.info("영어 데이터셋 로드 중...")
            documents, queries = await self.data_processor.load_multi_datasets(
                self.dataset_names, Language.ENGLISH, self.samples_per_dataset,
                self.max_text_length, self.min_text_length
            )

            if not documents or not queries:
                logger.error("데이터셋 로드에 실패하여 실험을 중단합니다.")
                return {}

            logger.info(f"데이터 로드 완료: {len(documents)}개 문서, {len(queries)}개 쿼리")

            # 모든 전략을 병렬로 실행
            tasks = [
                self._run_single_strategy_with_components(strategy, documents, queries, Language.ENGLISH)
                for strategy in ChunkingStrategy
            ]
            strategy_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(strategy_results):
                strategy = list(ChunkingStrategy)[i]
                if isinstance(result, Exception):
                    logger.error(f"{strategy.value} 전략 실행 중 오류 발생: {result}", exc_info=True)
                    self.experiment_run.add_error(str(result), f"baseline-{strategy.value}")
                elif result:
                    all_results.append(result)
                    self.experiment_run.add_result(result)
                    logger.success(f"{strategy.value} 전략 완료 - MRR: {result.mrr:.3f}")

            self.experiment_run.end_time = datetime.now()
            logger.info(f"📊 Baseline 실험 완료: {self.run_id}")

            # 결과 저장
            await self._save_baseline_results(all_results)

            return {
                "run_id": self.run_id,
                "results": all_results,
                "experiment_summary": self._create_summary(all_results)
            }
        except Exception as e:
            logger.error(f"Baseline 실험 실패: {e}")
            self.experiment_run.add_error(str(e), "baseline_experiment")
            raise

    def _initialize_components_for_strategy(self, strategy: ChunkingStrategy, language: Language) -> Dict[str, Any]:
        """특정 전략을 위한 컴포넌트 초기화 (병렬 처리용)"""

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

    async def _run_single_strategy_with_components(
            self, strategy: ChunkingStrategy, documents: List[Document],
            queries: List[Query], language: Language
    ) -> Optional[EvaluationResult]:
        """각 전략별로 독립적인 컴포넌트를 사용하여 실행"""

        logger.info(f"[BASELINE] [{strategy.value}] 전략 실험 시작")
        components = self._initialize_components_for_strategy(strategy, language)
        return await self._run_single_strategy(strategy, documents, queries, components, language)

    async def _run_single_strategy(
            self, strategy: ChunkingStrategy, documents: List[Document],
            queries: List[Query], components: Dict[str, Any], language: Language
    ) -> Optional[EvaluationResult]:
        """단일 청킹 전략을 효율적인 병렬 배치 방식으로 실행합니다."""
        strategy_start_time = time.time()
        sample_size = min(len(documents), len(queries), self.config.experiment.sample_size)

        concurrency_limit = 3 if strategy == ChunkingStrategy.SEMANTIC else 5
        semaphore = asyncio.Semaphore(concurrency_limit)

        chunker = components["chunker"]
        embedder = components["embedder"]
        retriever = components["retriever"]
        evaluator = components["evaluator"]

        logger.info(f"[{strategy.value}] {sample_size}개 문서에 대한 병렬 청킹 시작...")

        # 모든 문서 병렬 청킹
        async def chunk_doc(doc, query):
            async with semaphore:
                if strategy == ChunkingStrategy.QUERY_AWARE:
                    return await chunker.query_aware_chunk(doc, query)
                else:
                    chunks = await chunker.chunk_document(doc)
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

        logger.success(f"[{strategy.value}] 청킹 완료: 총 {len(all_chunks)}개 청크 생성.")

        # 모든 청크 임베딩
        logger.info(f"[{strategy.value}] 전체 청크 임베딩 시작...")
        try:
            if hasattr(embedder, "embed_and_store_chunks"):
                from collections import defaultdict
                chunks_by_doc = defaultdict(list)
                for ch in all_chunks:
                    if not getattr(ch, "doc_id", None):
                        raise ValueError("chunk.doc_id 누락: 저장 경로 매핑 불가")
                    chunks_by_doc[ch.doc_id].append(ch)

                for doc_id, doc_chunks in chunks_by_doc.items():
                    await embedder.embed_and_store_chunks(
                        chunks=doc_chunks,
                        chunk_type=strategy.value,
                        document_id=doc_id
                    )
            else:
                await embedder.embed_chunks(all_chunks)

            if getattr(all_chunks[0], "embedding", None) is None:
                await embedder.embed_chunks(all_chunks)
        except Exception as e:
            logger.error(f"[{strategy.value}] 임베딩 단계 실패: {e}", exc_info=True)
            raise

        # 모든 쿼리에 대한 병렬 검색
        logger.info(f"[{strategy.value}] {sample_size}개 쿼리에 대한 병렬 검색 시작...")

        async def process_query(query):
            async with semaphore:
                try:
                    retrieved_chunks = await retriever.retrieve(query.question, all_chunks,
                                                                k=self.config.experiment.top_k_retrieval)

                    response = RAGResponse(
                        strategy=strategy,
                        query=query.question,
                        query_id=query.id,
                        response="[BASELINE MODE - GENERATION BYPASSED]",
                        chunks_used=retrieved_chunks if retrieved_chunks else [],
                        confidence=0.0
                    )

                    if retrieved_chunks:
                        ranked = []
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

        logger.success(f"[{strategy.value}] 검색 완료: {len(responses)}개 응답 생성.")

        # 최종 평가
        if responses:
            eval_result = await evaluator.evaluate_responses(responses, ground_truths)
            eval_result.strategy = strategy.value
            eval_result.metadata.update({
                "total_time_seconds": time.time() - strategy_start_time,
                "samples_processed": len(responses),
                "mode": "baseline"
            })
            return eval_result

        logger.warning(f"{strategy.value} 전략에 대한 유효한 응답이 없어 평가를 건너뜁니다.")
        return None

    async def _save_baseline_results(self, results: List[EvaluationResult]):
        """Baseline 결과 저장"""
        results_dir = Path("baseline_result")
        results_dir.mkdir(exist_ok=True, parents=True)

        results_data = [r.to_dict() for r in results if r]
        experiment_metadata = {
            "run_id": self.run_id,
            "mode": "baseline",
            "config": self.config.to_dict(),
            "summary": self.experiment_run.get_summary(),
            "errors": self.experiment_run.errors,
            "evaluation_mode": self.evaluation_mode
        }

        async def write_json(path, data):
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder))

        save_tasks = [
            write_json(results_dir / "baseline_raw_results.json", results_data),
            write_json(results_dir / "baseline_experiment_metadata.json", experiment_metadata),
            write_json(results_dir / "baseline_summary.json", self._create_summary(results))
        ]
        await asyncio.gather(*save_tasks)
        logger.info(f"📊 Baseline 결과 저장 완료: {results_dir}")

    def _create_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Baseline 실험 요약 생성"""
        valid_results = [r for r in results if r and r.num_samples > 0]
        if not valid_results:
            return {}

        best_result = max(valid_results, key=lambda r: r.mrr)
        summary = {
            "evaluation_mode": "baseline_retrieval",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline RAG 청킹 전략 실험 (6개 전략 병렬)")
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
        "--datasets", type=str, nargs="+",
        default=["squad", "newsqa", "bioasq"],
        help="사용할 데이터셋 목록"
    )
    parser.add_argument(
        "--samples_per_dataset", type=int, default=100,
        help="각 데이터셋에서 가져올 샘플 수"
    )
    parser.add_argument(
        "--max_text_length", type=int, default=2000,
        help="최대 텍스트 길이 (짧은 글 실험용)"
    )
    parser.add_argument(
        "--min_text_length", type=int, default=50,
        help="최소 텍스트 길이 (너무 짧은 텍스트 필터링)"
    )

    args = parser.parse_args()

    if not config.api.openai_api_key or "sk-" not in config.api.openai_api_key:
        logger.error("오류: OPENAI_API_KEY가 유효하지 않습니다. 환경 변수를 확인해주세요.")
        sys.exit(1)

    async def main():
        logger.info("📊 Baseline 실험 파이프라인 시작")
        logger.info("모든 청킹 전략(6개)을 병렬로 실행합니다")

        pipeline = BaselineExperimentPipeline()
        pipeline.evaluation_mode = args.mode

        if args.enable_embedding_storage:
            pipeline.enable_embedding_storage = True
        pipeline.storage_path = Path(args.storage_path)

        # 텍스트 길이 설정
        pipeline.max_text_length = args.max_text_length
        pipeline.min_text_length = args.min_text_length

        # 데이터셋 설정
        pipeline.enable_multi_datasets(args.datasets, args.samples_per_dataset)

        logger.info("=" * 50)
        logger.info(f"Baseline RAG 실험 시작 (모드: {args.mode})")
        logger.info(f"데이터셋: {args.datasets} ({args.samples_per_dataset}개씩)")
        logger.info(f"텍스트 길이: {args.min_text_length}~{args.max_text_length}자")
        logger.info("=" * 50)

        try:
            results_data = await pipeline.run_baseline_experiment()
            summary = results_data.get("experiment_summary", {})

            print("\n" + "=" * 60)
            print("📊 BASELINE 실험 결과 요약")
            print("=" * 60)

            if summary:
                print(f"최고 성능 전략 (MRR 기준): {summary.get('best_strategy_by_mrr', 'N/A')}")
                print(f"최고 MRR 점수: {summary.get('best_mrr_score', 0):.3f}")
                print("\n전략별 성능:")
                for strategy, scores in summary.get('strategy_retrieval_scores', {}).items():
                    print(f"  - {strategy}: MRR={scores.get('mrr', 0):.3f}")
            else:
                print("실행된 실험이 없어 요약할 결과가 없습니다.")

            print("=" * 60)

        except Exception as e:
            logger.error(f"Baseline 실험 실패: {e}", exc_info=True)
        finally:
            logger.info("📊 Baseline 실험 종료")

    asyncio.run(main())