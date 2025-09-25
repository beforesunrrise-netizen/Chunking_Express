"""
Baseline vs Agent 성능 비교 실험 모듈
"""
import asyncio
from pathlib import Path
from loguru import logger

from src.main import RAGExperimentPipeline
from src.config import config


async def run_baseline_vs_agent_comparison(args):
    """Baseline(6개 전략) vs Agent(자동선택) 성능 비교 실험"""
    logger.info("🔍 Baseline vs Agent 성능 비교 실험을 시작합니다...")

    baseline_results = {}
    agent_results = {}

    # 공통 설정
    default_datasets = ["squad", "newsqa", "bioasq"]

    try:
        # === 1. Baseline 실험 (6개 전략 모두) ===
        logger.info("📊 [1단계] Baseline 실험 시작 - 모든 전략 실행...")

        baseline_pipeline = RAGExperimentPipeline()
        baseline_pipeline.evaluation_mode = args.mode
        baseline_pipeline.enable_multi_datasets(default_datasets, args.samples_per_dataset)
        baseline_pipeline.use_multi_datasets = True
        baseline_pipeline.max_text_length = args.max_text_length
        baseline_pipeline.min_text_length = args.min_text_length
        baseline_pipeline.enable_embedding_storage = args.enable_embedding_storage
        baseline_pipeline.storage_path = Path(args.storage_path)

        # 지능형 청킹 비활성화 (모든 전략 실행)
        baseline_pipeline.disable_intelligent_chunking()

        logger.info("Baseline: 6개 전략 모두 실행 중...")
        baseline_data = await baseline_pipeline.run_full_experiment()
        baseline_results = baseline_data.get("experiment_summary", {}) if baseline_data else {}

        # === 2. Agent 실험 (자동 전략 선택) ===
        logger.info("🤖 [2단계] Agent 실험 시작 - 자동 전략 선택...")

        agent_pipeline = RAGExperimentPipeline()
        agent_pipeline.evaluation_mode = args.mode
        agent_pipeline.enable_multi_datasets(default_datasets, args.samples_per_dataset)
        agent_pipeline.use_multi_datasets = True
        agent_pipeline.max_text_length = args.max_text_length
        agent_pipeline.min_text_length = args.min_text_length
        agent_pipeline.enable_embedding_storage = args.enable_embedding_storage
        agent_pipeline.storage_path = Path(args.storage_path)

        # 지능형 청킹 활성화 (자동 전략 선택)
        agent_pipeline.enable_intelligent_chunking(context="balanced", force_no_api=False)
        agent_pipeline._intelligent_mode = "auto_select"

        logger.info("Agent: 도메인별 최적 전략 자동 선택 중...")
        agent_data = await agent_pipeline.run_full_experiment()
        agent_results = agent_data.get("experiment_summary", {}) if agent_data else {}

        # === 3. 결과 비교 분석 ===
        print_comparison_results(baseline_results, agent_results, args.mode)

    except Exception as e:
        logger.error(f"비교 실험 실행 중 오류 발생: {str(e)}")
        import traceback
        logger.error(f"상세 오류:\n{traceback.format_exc()}")


def print_comparison_results(baseline_results, agent_results, mode):
    """Baseline vs Agent 결과 비교 출력"""
    print(f"\n{'=' * 80}")
    print(f" 🔍 BASELINE vs AGENT 성능 비교 결과")
    print(f"{'=' * 80}")

    if mode == 'retrieval':
        # 검색 성능 비교
        baseline_best = baseline_results.get('best_mrr_score', 0)
        agent_best = agent_results.get('best_mrr_score', 0)

        print(f"📊 검색 성능 (MRR 기준)")
        print(f"┌─────────────────────┬─────────────┬─────────────────┐")
        print(f"│ 방법                │ 최고 MRR    │ 최고 전략       │")
        print(f"├─────────────────────┼─────────────┼─────────────────┤")
        print(f"│ Baseline (6개 전략) │ {baseline_best:.3f}       │ {baseline_results.get('best_strategy_by_mrr', 'N/A')[:15]} │")
        print(f"│ Agent (자동 선택)   │ {agent_best:.3f}       │ {agent_results.get('best_strategy_by_mrr', 'N/A')[:15]} │")
        print(f"└─────────────────────┴─────────────┴─────────────────┘")

        # 성능 개선/감소 계산
        if baseline_best > 0:
            improvement = ((agent_best - baseline_best) / baseline_best) * 100
            print(f"\n📈 성능 변화: {improvement:+.1f}% ({'개선' if improvement > 0 else '감소'})")

        # 전략별 상세 비교
        print(f"\n📋 전략별 상세 성능:")

        baseline_strategies = baseline_results.get('strategy_retrieval_scores', {})
        agent_strategies = agent_results.get('strategy_retrieval_scores', {})

        print(f"┌─────────────────────┬───────────────────┬───────────────────┐")
        print(f"│ 전략                │ Baseline MRR      │ Agent MRR         │")
        print(f"├─────────────────────┼───────────────────┼───────────────────┤")

        all_strategies = set(baseline_strategies.keys()) | set(agent_strategies.keys())
        for strategy in sorted(all_strategies):
            baseline_mrr = baseline_strategies.get(strategy, {}).get('mrr', 0)
            agent_mrr = agent_strategies.get(strategy, {}).get('mrr', 0)
            print(f"│ {strategy[:19]:<19} │ {baseline_mrr:.3f}             │ {agent_mrr:.3f}             │")
        print(f"└─────────────────────┴───────────────────┴───────────────────┘")

    elif mode == 'e2e':
        # E2E 성능 비교
        baseline_best = baseline_results.get('best_overall_score', 0)
        agent_best = agent_results.get('best_overall_score', 0)

        print(f"📊 종합 성능 (Overall 기준)")
        print(f"┌─────────────────────┬─────────────┬─────────────────┐")
        print(f"│ 방법                │ 최고 Overall│ 최고 전략       │")
        print(f"├─────────────────────┼─────────────┼─────────────────┤")
        print(f"│ Baseline (6개 전략) │ {baseline_best:.3f}       │ {baseline_results.get('best_strategy_by_overall', 'N/A')[:15]} │")
        print(f"│ Agent (자동 선택)   │ {agent_best:.3f}       │ {agent_results.get('best_strategy_by_overall', 'N/A')[:15]} │")
        print(f"└─────────────────────┴─────────────┴─────────────────┘")

    print(f"\n💡 결론: {'Agent가 더 효율적' if agent_best >= baseline_best else 'Baseline이 더 안정적'}")
    print(f"{'=' * 80}\n")