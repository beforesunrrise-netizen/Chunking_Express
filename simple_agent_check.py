#!/usr/bin/env python3
"""
간단한 Agent 작동 확인 스크립트
의존성 없이 Agent 관련 파일들이 존재하는지만 확인
"""

import os
from pathlib import Path

def check_agent_files():
    """Agent 관련 파일들 존재 여부 확인"""

    print("🔍 Agent 파일 구조 확인")
    print("=" * 50)

    base_dir = Path(__file__).parent
    required_files = [
        "src/agents/__init__.py",
        "src/agents/chunking_agent.py",
        "src/agents/chunking_router.py",
        "src/agents/text_analyzer.py",
        "src/config.py",
        "src/data_structures.py",
        "src/main.py"
    ]

    missing_files = []
    existing_files = []

    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")

    print(f"\n📊 결과: {len(existing_files)}/{len(required_files)} 파일 존재")

    if missing_files:
        print(f"⚠️  누락된 파일들:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    print("✅ 모든 Agent 관련 파일이 존재합니다!")
    return True

def check_environment():
    """환경 설정 확인"""

    print("\n🌍 환경 설정 확인")
    print("=" * 50)

    api_key = os.getenv("OPENAI_API_KEY", "")

    if api_key and api_key.startswith("sk-"):
        print(f"✅ OPENAI_API_KEY 설정됨: {api_key[:10]}...")
        return True
    else:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   환경변수를 설정해주세요:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False

def check_main_py_auto_select():
    """main.py에서 auto_select 로직이 있는지 확인"""

    print("\n🤖 Auto-select 로직 확인")
    print("=" * 50)

    main_py_path = Path(__file__).parent / "src/main.py"

    if not main_py_path.exists():
        print("❌ src/main.py 파일이 없습니다.")
        return False

    with open(main_py_path, "r", encoding="utf-8") as f:
        content = f.read()

    auto_select_indicators = [
        "_run_intelligent_auto_select",
        'intelligent_mode == "auto_select"',
        "지능형 자동 전략 선택"
    ]

    found_indicators = []
    for indicator in auto_select_indicators:
        if indicator in content:
            found_indicators.append(indicator)
            print(f"✅ '{indicator}' 발견")
        else:
            print(f"❌ '{indicator}' 없음")

    if len(found_indicators) >= 2:
        print("✅ Auto-select 로직이 구현되어 있습니다!")
        return True
    else:
        print("⚠️  Auto-select 로직이 불완전할 수 있습니다.")
        return False

def main():
    print("🚀 간단한 Agent 작동 확인 시작")
    print()

    files_ok = check_agent_files()
    env_ok = check_environment()
    logic_ok = check_main_py_auto_select()

    print("\n" + "=" * 50)
    print("📋 종합 결과")
    print("=" * 50)

    if files_ok and env_ok and logic_ok:
        print("🎉 모든 확인 완료! Agent 테스트 준비가 되었습니다.")
        print("\n다음 단계:")
        print("1. pip install -r requirements.txt")
        print("2. python quick_agent_test.py")
        print("3. python test_intelligent_agent.py")
    elif files_ok and logic_ok:
        print("⚠️  파일과 로직은 준비되었지만 API 키가 없습니다.")
        print("OPENAI_API_KEY 환경변수를 설정한 후 테스트하세요.")
    else:
        print("❌ 몇 가지 문제가 있습니다. 위의 결과를 확인해주세요.")

if __name__ == "__main__":
    main()