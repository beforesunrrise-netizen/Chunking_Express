#!/usr/bin/env python3
"""
빠른 Agent 지능형 전략 선택 테스트

소규모 샘플로 Agent가 제대로 작동하는지 확인하는 스크립트입니다.
"""

import asyncio
import sys
from pathlib import Path

# 모듈 경로 설정
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from src.config import config, Language, ChunkingStrategy
    from src.data_structures import Document, Query
    from src.agents import ChunkingAgent
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("현재 디렉토리에서 실행해주세요: cd /Users/jaeyoung/PycharmProjects/pythonProject/Chunking_Express")
    sys.exit(1)

async def test_agent_strategy_selection():
    """Agent의 전략 선택 기능을 직접 테스트"""

    print("🤖 Agent 전략 선택 직접 테스트 시작")
    print("=" * 50)

    # 테스트용 문서 생성
    test_documents = [
        Document(
            id="doc_1",
            content="""
            Machine learning is a subset of artificial intelligence (AI) that provides systems
            the ability to automatically learn and improve from experience without being explicitly
            programmed. Machine learning focuses on the development of computer programs that can
            access data and use it to learn for themselves.
            """,
            language=Language.ENGLISH
        ),
        Document(
            id="doc_2",
            content="""
            The COVID-19 pandemic has significantly impacted global health systems. Healthcare
            workers have been at the forefront of the response, working tirelessly to treat
            patients and prevent the spread of the virus. The development of vaccines has been
            crucial in controlling the pandemic.
            """,
            language=Language.ENGLISH
        ),
        Document(
            id="doc_3",
            content="""
            Climate change refers to long-term shifts and alterations in global or regional
            climate patterns. Since the mid-20th century, climate change has been largely
            attributed to the increased levels of atmospheric carbon dioxide produced by the
            use of fossil fuels.
            """,
            language=Language.ENGLISH
        )
    ]

    # 테스트용 쿼리 생성
    test_queries = [
        Query(
            id="query_1",
            question="What is machine learning and how does it work?",
            language=Language.ENGLISH,
            expected_answer="Machine learning is a subset of AI that learns from data.",
            context_id="doc_1"
        ),
        Query(
            id="query_2",
            question="How has COVID-19 affected healthcare workers?",
            language=Language.ENGLISH,
            expected_answer="Healthcare workers have been at the forefront fighting COVID-19.",
            context_id="doc_2"
        ),
        Query(
            id="query_3",
            question="What causes climate change?",
            language=Language.ENGLISH,
            expected_answer="Climate change is largely caused by increased CO2 from fossil fuels.",
            context_id="doc_3"
        )
    ]

    try:
        # Agent 초기화
        agent = ChunkingAgent(
            language=Language.ENGLISH,
            chunk_size_limit=512,
            overlap_ratio=0.1,
            default_context="balanced"
        )

        print(f"✅ Agent 초기화 완료")
        print(f"📊 테스트 문서 수: {len(test_documents)}개")
        print(f"❓ 테스트 쿼리 수: {len(test_queries)}개")
        print()

        # 각 문서에 대해 전략 선택 테스트
        strategy_selections = []

        for i, (doc, query) in enumerate(zip(test_documents, test_queries)):
            print(f"🔍 문서 {i+1} 분석 중...")
            print(f"   문서 ID: {doc.id}")
            print(f"   문서 길이: {len(doc.content)} 글자")
            print(f"   쿼리: {query.question[:50]}...")

            try:
                # Router를 통한 전략 추천
                recommendation = await agent.router.recommend_strategy(
                    document=doc,
                    query=query,
                    context="balanced"
                )

                selected_strategy = recommendation.primary_strategy
                confidence = recommendation.confidence
                reasoning = recommendation.reasoning

                print(f"   🎯 선택된 전략: {selected_strategy.value}")
                print(f"   📈 신뢰도: {confidence:.2f}")
                print(f"   🧠 선택 근거: {reasoning[:100]}...")

                strategy_selections.append({
                    "doc_id": doc.id,
                    "selected_strategy": selected_strategy.value,
                    "confidence": confidence,
                    "reasoning": reasoning
                })

                print("   ✅ 성공")

            except Exception as e:
                print(f"   ❌ 오류: {e}")
                strategy_selections.append({
                    "doc_id": doc.id,
                    "selected_strategy": "ERROR",
                    "confidence": 0.0,
                    "reasoning": str(e)
                })

            print()

        # 결과 요약
        print("=" * 50)
        print("📊 Agent 전략 선택 결과 요약")
        print("=" * 50)

        from collections import Counter
        strategy_counter = Counter([sel["selected_strategy"] for sel in strategy_selections if sel["selected_strategy"] != "ERROR"])

        print("전략 분포:")
        for strategy, count in strategy_counter.items():
            percentage = (count / len(strategy_selections)) * 100
            print(f"  - {strategy}: {count}회 ({percentage:.1f}%)")

        print(f"\n평균 신뢰도: {sum(sel['confidence'] for sel in strategy_selections) / len(strategy_selections):.2f}")

        success_count = len([sel for sel in strategy_selections if sel["selected_strategy"] != "ERROR"])
        print(f"성공률: {success_count}/{len(strategy_selections)} ({(success_count/len(strategy_selections)*100):.1f}%)")

        if success_count == len(strategy_selections):
            print("\n🎉 Agent가 정상적으로 작동합니다!")
            return True
        else:
            print(f"\n⚠️  {len(strategy_selections) - success_count}개 문서에서 오류 발생")
            return False

    except Exception as e:
        print(f"❌ Agent 테스트 실패: {e}")
        return False

async def main():
    print("🚀 빠른 Agent 테스트 시작")

    # API 키 확인
    if not config.api.openai_api_key or "sk-" not in config.api.openai_api_key:
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("환경 변수를 설정하거나 config 파일을 확인해주세요.")
        return

    success = await test_agent_strategy_selection()

    if success:
        print("\n✅ Agent가 정상 작동합니다!")
        print("이제 python test_intelligent_agent.py 를 실행하여 전체 실험을 진행하세요.")
    else:
        print("\n❌ Agent에 문제가 있습니다. 로그를 확인해주세요.")

if __name__ == "__main__":
    asyncio.run(main())