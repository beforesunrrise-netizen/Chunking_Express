#!/usr/bin/env python3
"""
최소한의 Agent 테스트 - 직접 실행
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🚀 최소한의 Agent 테스트 시작")
    print("=" * 50)

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("API 키가 config 파일에 설정되어 있는지 확인합니다...")

    # 현재 디렉토리 확인
    print(f"현재 디렉토리: {os.getcwd()}")

    # src/main.py 직접 실행 (더 간단한 설정으로)
    command = [
        sys.executable,
        "src/main.py",
        "--use_intelligent_chunking",
        "--intelligent_mode", "auto_select",
        "--chunking_context", "balanced",
        "--mode", "retrieval",
        "-n", "10",  # 매우 작은 샘플 크기
        "--datasets", "squad",  # 하나의 데이터셋만
        "--samples_per_dataset", "10",
        "--use_multi_datasets"
    ]

    print("실행 명령어:")
    print(" ".join(command))
    print()

    try:
        # 실시간으로 출력을 보기 위해 subprocess.run 대신 Popen 사용
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        print("실험 실행 중...")
        print("-" * 50)

        # 실시간 출력
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        # 에러 출력 확인
        stderr = process.stderr.read()
        if stderr:
            print("에러 출력:")
            print(stderr)

        return_code = process.poll()
        print(f"\n프로세스 종료. 반환 코드: {return_code}")

        if return_code == 0:
            print("✅ 성공!")

            # 결과 파일 확인
            results_dir = Path("results")
            if results_dir.exists():
                print(f"\n📁 결과 디렉토리: {results_dir}")

                # strategy_analysis 디렉토리 확인
                strategy_analysis_dir = results_dir / "strategy_analysis"
                if strategy_analysis_dir.exists():
                    print(f"🧠 Agent 분석 결과 발견!")
                    for file in strategy_analysis_dir.rglob("*"):
                        if file.is_file():
                            print(f"   📄 {file.relative_to(results_dir)}")
                else:
                    print("⚠️  strategy_analysis 디렉토리가 없습니다.")
        else:
            print("❌ 실패")

    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")

    print("\n테스트 완료")

if __name__ == "__main__":
    main()