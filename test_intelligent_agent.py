#!/usr/bin/env python3
"""
Agent 지능형 전략 선택 테스트 스크립트

이 스크립트는 Agent가 각 문서를 분석하여 최적의 청킹 전략을
자동으로 선택하는 기능을 테스트합니다.
"""

import subprocess
import sys
from pathlib import Path

def run_intelligent_agent_test():
    """Agent 지능형 전략 선택 테스트 실행"""

    print("=" * 70)
    print("🤖 Agent 지능형 전략 선택 테스트 시작")
    print("=" * 70)

    # 실행할 명령어 설정 (확실하게 auto_select 모드로)
    command = [
        sys.executable,
        "src/main.py",
        "--use_intelligent_chunking",      # 지능형 청킹 활성화 ✅
        "--intelligent_mode", "auto_select",  # 자동 전략 선택 모드 ✅
        "--chunking_context", "balanced",   # 균형잡힌 컨텍스트
        "--mode", "retrieval",             # 검색 모드
        "--use_multi_datasets",            # 다중 데이터셋 사용
        "--datasets", "squad", "newsqa",   # 2개 데이터셋만 사용
        "--samples_per_dataset", "50",     # 테스트용으로 50개씩만
        "--enable_embedding_storage",      # 임베딩 저장 활성화
        "-n", "50"                        # 전체 샘플 수 제한
    ]

    print("🚀 실행 명령어:")
    print(" ".join(command))
    print("\n📋 실험 설정:")
    print("- 지능형 청킹: ✅ 활성화")
    print("- 모드: auto_select (Agent가 각 문서마다 전략 자동 선택)")
    print("- 데이터셋: squad, newsqa")
    print("- 각 데이터셋별 샘플 수: 50개")
    print("- 컨텍스트: balanced")
    print("- 평가 모드: retrieval")
    print("\n🔍 기대 결과:")
    print("- Agent가 각 문서를 분석")
    print("- 문서 특성에 따라 최적 전략 자동 선택")
    print("- 전략 선택 분석 결과 저장 (results/strategy_analysis/)")
    print("\n실험 시작...")

    try:
        # 실험 실행
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )

        print("\n" + "=" * 70)
        print("📊 실험 결과")
        print("=" * 70)

        if result.returncode == 0:
            print("✅ Agent 지능형 전략 선택 테스트가 성공적으로 완료되었습니다!")
            print("\n📤 표준 출력:")
            print(result.stdout)

            # 결과 파일 확인
            results_dir = Path("results")
            if results_dir.exists():
                print(f"\n📁 결과 파일 저장 위치: {results_dir.absolute()}")

                # 전략 분석 결과 확인
                strategy_analysis_dir = results_dir / "strategy_analysis"
                if strategy_analysis_dir.exists():
                    print(f"\n🧠 Agent 전략 분석 결과: {strategy_analysis_dir.absolute()}")
                    for analysis_file in strategy_analysis_dir.rglob("*.json"):
                        print(f"   📄 {analysis_file.name}")
                    for analysis_file in strategy_analysis_dir.rglob("*.csv"):
                        print(f"   📄 {analysis_file.name}")
                else:
                    print("⚠️  전략 분석 결과 디렉토리가 없습니다. auto_select 모드가 작동하지 않았을 수 있습니다.")

                # 일반 결과 확인
                for result_file in results_dir.rglob("*.json"):
                    if "strategy_analysis" not in str(result_file):
                        print(f"   📄 {result_file.relative_to(results_dir)}")

        else:
            print("❌ 실험 실행 중 오류가 발생했습니다.")
            print(f"\n🚨 반환 코드: {result.returncode}")
            print("\n📤 표준 출력:")
            print(result.stdout)
            print("\n📥 표준 에러:")
            print(result.stderr)

    except Exception as e:
        print(f"❌ 실험 실행 중 예외가 발생했습니다: {e}")

    print("\n" + "=" * 70)
    print("🏁 Agent 지능형 전략 선택 테스트 완료")
    print("=" * 70)

if __name__ == "__main__":
    run_intelligent_agent_test()