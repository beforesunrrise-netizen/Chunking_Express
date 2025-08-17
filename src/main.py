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
        # 실제 분석 로직이 필요하다면 여기에 구현
        return {"descriptive_stats": {}, "statistical_tests": {}}


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
        documents, queries = await self.data_processor.load_data(language)
        if not documents or not queries:
            logger.error(f"{language.value} 데이터셋 로드에 실패하여 실험을 중단합니다.")
            return []

        logger.info(f"{language.value} 데이터 로드 완료: {len(documents)}개 문서, {len(queries)}개 쿼리")

        logger.info("모든 청킹 전략을 병렬로 실행합니다...")

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
                if strategy == ChunkingStrategy.QUERY_AWARE:
                    chunks = await chunker.query_aware_chunk(doc, query)  # query 객체를 그대로 전달
                else:
                    chunks = await chunker.chunk_document(doc)

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
        concurrency_limit = 10 if strategy == ChunkingStrategy.SEMANTIC else 50
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
            if not getattr(all_chunks[0], "embedding", None):
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
            write_json(results_dir / "100_raw_results.json", results_data),
            write_json(results_dir / "analysis_results.json", analysis),
            write_json(results_dir / "experiment_metadata.json", experiment_metadata)
        ]
        await asyncio.gather(*save_tasks)
        logger.info(f"결과 저장 완료: {results_dir}")

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


    asyncio.run(run_experiment())