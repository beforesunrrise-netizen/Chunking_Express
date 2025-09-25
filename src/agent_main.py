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


class AgentExperimentPipeline:
    """Agent 실험 파이프라인 - 지능형 자동 전략 선택"""

    def __init__(self):
        self.config = config
        self.data_processor = DataProcessor()
        self.run_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_run = ExperimentRun(
            run_id=self.run_id,
            config=self.config.to_dict(),
            start_time=datetime.now()
        )
        self.enable_embedding_storage = True
        self.storage_path = self.config.paths.embedding_storage_path
        self.evaluation_mode = "retrieval"

        # 지능형 청킹 에이전트
        self.chunking_agent = None

        # 다중 데이터셋 설정
        self.dataset_names = []
        self.samples_per_dataset = 100

        # 텍스트 길이 필터링 설정 (짧은 글 실험용)
        self.max_text_length = 2000
        self.min_text_length = 50

    def enable_intelligent_chunking(self, context: str = "balanced"):
        """지능형 청킹 에이전트를 활성화합니다."""
        self.chunking_agent = ChunkingAgent(
            language=Language.ENGLISH,
            chunk_size_limit=self.config.experiment.chunk_size_limit,
            overlap_ratio=self.config.experiment.overlap_ratio,
            default_context=context
        )
        logger.info(f"🤖 지능형 청킹 에이전트 활성화 (컨텍스트: {context})")

    def enable_multi_datasets(self, dataset_names: List[str], samples_per_dataset: int = 100):
        """다중 데이터셋 모드를 활성화합니다."""
        self.dataset_names = dataset_names
        self.samples_per_dataset = samples_per_dataset
        logger.info(f"Agent 다중 데이터셋 모드: {len(dataset_names)}개 데이터셋, 각 {samples_per_dataset}개 샘플")

    async def run_agent_experiment(self) -> Dict[str, Any]:
        """Agent 실험 실행 - 지능형 자동 전략 선택"""
        logger.info(f"🤖 Agent 실험 시작: {self.run_id} (모드: {self.evaluation_mode})")
        logger.info("도메인별 최적 전략을 자동 선택하여 실행합니다...")

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

            # 지능형 자동 전략 선택 실행
            results = await self._run_intelligent_auto_select(documents, queries, Language.ENGLISH)
            all_results.extend(results)

            self.experiment_run.end_time = datetime.now()
            logger.info(f"🤖 Agent 실험 완료: {self.run_id}")

            # 결과 저장
            await self._save_agent_results(all_results)

            return {
                "run_id": self.run_id,
                "results": all_results,
                "experiment_summary": self._create_summary(all_results)
            }
        except Exception as e:
            logger.error(f"Agent 실험 실패: {e}")
            self.experiment_run.add_error(str(e), "agent_experiment")
            raise

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

        logger.info("🤖 각 문서마다 최적의 전략을 자동 선택하여 청킹합니다...")

        # 전략별 결과를 수집할 딕셔너리
        strategy_results = {}
        strategy_counts = {}

        # 분석 데이터 수집
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

                # 분석 데이터 수집
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
        logger.info("🤖 전략별 성능 평가를 시작합니다...")
        evaluation_results = []

        for strategy_name, doc_results in strategy_results.items():
            if not doc_results:
                continue

            try:
                # 각 전략별로 평가 수행
                strategy_enum = ChunkingStrategy(strategy_name)
                components = self._initialize_components_for_strategy(strategy_enum, language)

                # 평가 로직
                eval_result = await self._evaluate_strategy_results(
                    strategy_enum, doc_results, components, language
                )

                if eval_result:
                    evaluation_results.append(eval_result)
                    self.experiment_run.add_result(eval_result)

                logger.success(f"전략 {strategy_name} 평가 완료 "
                              f"(사용된 문서: {strategy_counts[strategy_name]}개, MRR: {eval_result.mrr:.3f})")

            except Exception as e:
                logger.error(f"전략 {strategy_name} 평가 실패: {e}")

        # 전략 사용 통계 로그
        logger.info("=== 🤖 Agent 전략 선택 통계 ===")
        total_docs = sum(strategy_counts.values())
        for strategy_name, count in strategy_counts.items():
            percentage = (count / total_docs) * 100 if total_docs > 0 else 0
            logger.info(f"{strategy_name}: {count}개 문서 ({percentage:.1f}%)")

        # 분석 결과 저장
        await self._save_strategy_analysis(strategy_analysis_data)

        return evaluation_results

    def _initialize_components_for_strategy(self, strategy: ChunkingStrategy, language: Language) -> Dict[str, Any]:
        """특정 전략을 위한 컴포넌트 초기화"""

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
                        response="[AGENT MODE - INTELLIGENT AUTO-SELECT]",
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
                    "auto_selected": True,
                    "mode": "agent"
                })
                return eval_result

        except Exception as e:
            logger.error(f"전략 {strategy.value} 평가 중 오류: {e}")

        return None

    async def _save_agent_results(self, results: List[EvaluationResult]):
        """Agent 결과 저장"""
        results_dir = Path("agent_result")
        results_dir.mkdir(exist_ok=True, parents=True)

        results_data = [r.to_dict() for r in results if r]
        experiment_metadata = {
            "run_id": self.run_id,
            "mode": "agent",
            "config": self.config.to_dict(),
            "summary": self.experiment_run.get_summary(),
            "errors": self.experiment_run.errors,
            "evaluation_mode": self.evaluation_mode
        }

        async def write_json(path, data):
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder))

        save_tasks = [
            write_json(results_dir / "agent_raw_results.json", results_data),
            write_json(results_dir / "agent_experiment_metadata.json", experiment_metadata),
            write_json(results_dir / "agent_summary.json", self._create_summary(results))
        ]
        await asyncio.gather(*save_tasks)
        logger.info(f"🤖 Agent 결과 저장 완료: {results_dir}")

    async def _save_strategy_analysis(self, analysis_data: Dict[str, Any]):
        """전략 분석 데이터를 저장"""
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
            results_dir = Path("agent_result") / "strategy_analysis"
            results_dir.mkdir(exist_ok=True, parents=True)

            # JSON 저장
            analysis_path = results_dir / "agent_strategy_analysis.json"
            async with aiofiles.open(analysis_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(analysis_summary, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder))

            logger.info(f"Agent 전략 분석 결과 저장 완료: {results_dir}")

        except Exception as e:
            logger.error(f"Agent 전략 분석 저장 실패: {e}")

    def _create_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Agent 실험 요약 생성"""
        valid_results = [r for r in results if r and r.num_samples > 0]
        if not valid_results:
            return {}

        best_result = max(valid_results, key=lambda r: r.mrr)
        summary = {
            "evaluation_mode": "agent_retrieval",
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
    parser = argparse.ArgumentParser(description="Agent RAG 청킹 전략 실험 (지능형 자동 선택)")
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
        "--chunking_context", type=str, default="balanced",
        choices=["quality_focused", "speed_focused", "balanced", "cost_conscious"],
        help="청킹 컨텍스트: quality_focused, speed_focused, balanced, cost_conscious"
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
        logger.info("🤖 Agent 실험 파이프라인 시작")
        logger.info("지능형 청킹 에이전트가 도메인별 최적 전략을 자동 선택합니다")

        pipeline = AgentExperimentPipeline()
        pipeline.evaluation_mode = args.mode

        if args.enable_embedding_storage:
            pipeline.enable_embedding_storage = True
        pipeline.storage_path = Path(args.storage_path)

        # 텍스트 길이 설정
        pipeline.max_text_length = args.max_text_length
        pipeline.min_text_length = args.min_text_length

        # 지능형 청킹 활성화
        pipeline.enable_intelligent_chunking(args.chunking_context)

        # 데이터셋 설정
        pipeline.enable_multi_datasets(args.datasets, args.samples_per_dataset)

        logger.info("=" * 50)
        logger.info(f"Agent RAG 실험 시작 (모드: {args.mode})")
        logger.info(f"청킹 컨텍스트: {args.chunking_context}")
        logger.info(f"데이터셋: {args.datasets} ({args.samples_per_dataset}개씩)")
        logger.info(f"텍스트 길이: {args.min_text_length}~{args.max_text_length}자")
        logger.info("=" * 50)

        try:
            results_data = await pipeline.run_agent_experiment()
            summary = results_data.get("experiment_summary", {})

            print("\n" + "=" * 60)
            print("🤖 AGENT 실험 결과 요약")
            print("=" * 60)

            if summary:
                print(f"최고 성능 전략 (MRR 기준): {summary.get('best_strategy_by_mrr', 'N/A')}")
                print(f"최고 MRR 점수: {summary.get('best_mrr_score', 0):.3f}")
                print("\n선택된 전략별 성능:")
                for strategy, scores in summary.get('strategy_retrieval_scores', {}).items():
                    print(f"  - {strategy}: MRR={scores.get('mrr', 0):.3f}")
            else:
                print("실행된 실험이 없어 요약할 결과가 없습니다.")

            print("=" * 60)

        except Exception as e:
            logger.error(f"Agent 실험 실패: {e}", exc_info=True)
        finally:
            logger.info("🤖 Agent 실험 종료")

    asyncio.run(main())