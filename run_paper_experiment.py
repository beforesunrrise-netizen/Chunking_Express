#!/usr/bin/env python3
"""
논문 작성을 위한 청킹 전략 자동 분류 실험 실행 스크립트

이 스크립트는 다음과 같은 작업을 수행합니다:
1. 다양한 도메인의 데이터셋에서 각각 100개 샘플 수집
2. Router Agent가 각 문서를 읽고 최적의 청킹 전략을 자동 선택
3. 실험 결과와 전략 분석 데이터를 논문 작성용으로 저장
"""

import subprocess
import sys
from pathlib import Path

def run_experiment():
    """논문 작성을 위한 실험 실행"""

    print("=" * 60)
    print("청킹 전략 자동 분류 실험 시작")
    print("=" * 60)

    # 실행할 명령어 설정
    command = [
        sys.executable,  # Python 실행 파일 경로
        "src/main.py",
        "--use_multi_datasets",  # 다중 데이터셋 사용
        "--datasets", "squad", "newsqa",  # 사용할 데이터셋 (논문용으로 2개만)
        "--samples_per_dataset", "300",  # 각 데이터셋에서 100개씩
        "--mode", "retrieval",  # 검색 모드
        "--use_intelligent_chunking",  # 지능형 청킹 활성화
        "--intelligent_mode", "auto_select",  # 자동 전략 선택 모드
        "--chunking_context", "balanced",  # 균형잡힌 컨텍스트
        "--enable_embedding_storage"  # 임베딩 저장 활성화
    ]

    print("🚀 실행 명령어:")
    print(" ".join(command))
    print("\n📋 실험 설정:")
    print("- 다중 데이터셋: squad, newsqa")
    print("- 각 데이터셋별 샘플 수: 300")
    print("- 평가 모드: retrieval")
    print("- 🤖 지능형 청킹: auto_select (Agent 자동 전략 선택)")
    print("- 컨텍스트: balanced")
    print("\n🔍 중요: Agent가 각 문서를 분석하여 최적 전략을 자동 선택합니다!")
    print("📊 결과는 results/strategy_analysis/ 에 저장됩니다.")
    print("\n실험 시작...")

    try:
        # 실험 실행
        result = subprocess.run(command, capture_output=True, text=True, cwd=Path(__file__).parent)

        print("\n" + "=" * 60)
        print("실험 결과")
        print("=" * 60)

        if result.returncode == 0:
            print("✅ 실험이 성공적으로 완료되었습니다!")
            print("\n표준 출력:")
            print(result.stdout)

            # 결과 파일 확인
            results_dir = Path("results")
            if results_dir.exists():
                print(f"\n📁 결과 파일이 저장되었습니다: {results_dir.absolute()}")

                # 전략 분석 결과 확인
                strategy_analysis_dir = results_dir / "strategy_analysis"
                if strategy_analysis_dir.exists():
                    print(f"📊 전략 분석 결과: {strategy_analysis_dir.absolute()}")
                    for analysis_file in strategy_analysis_dir.rglob("*.json"):
                        print(f"   - {analysis_file.name}")
                    for analysis_file in strategy_analysis_dir.rglob("*.csv"):
                        print(f"   - {analysis_file.name}")
        else:
            print("❌ 실험 실행 중 오류가 발생했습니다.")
            print("\n표준 에러:")
            print(result.stderr)
            print("\n표준 출력:")
            print(result.stdout)

    except Exception as e:
        print(f"❌ 실험 실행 중 예외가 발생했습니다: {e}")

    print("\n" + "=" * 60)
    print("실험 완료")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()