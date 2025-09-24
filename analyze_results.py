#!/usr/bin/env python3
"""
실험 결과 분석 스크립트 - 논문 작성용

이 스크립트는 실험 결과를 분석하고 논문에서 사용할 수 있는
통계와 시각화를 생성합니다.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import numpy as np

def analyze_strategy_results():
    """전략 분석 결과를 분석하고 시각화"""

    # 결과 디렉토리 찾기
    results_dir = Path("results/strategy_analysis")

    if not results_dir.exists():
        print("❌ 결과 디렉토리를 찾을 수 없습니다. 먼저 실험을 실행해주세요.")
        return

    # 가장 최신 실험 결과 찾기
    latest_result_dir = max(results_dir.glob("exp_*"), key=lambda x: x.name, default=None)

    if not latest_result_dir:
        print("❌ 실험 결과를 찾을 수 없습니다.")
        return

    print(f"📊 분석 중: {latest_result_dir}")

    # JSON 결과 로드
    analysis_file = latest_result_dir / "strategy_analysis.json"
    csv_file = latest_result_dir / "strategy_decisions.csv"

    if not analysis_file.exists():
        print("❌ strategy_analysis.json 파일을 찾을 수 없습니다.")
        return

    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)

    print("\n" + "=" * 60)
    print("실험 결과 분석")
    print("=" * 60)

    # 1. 기본 통계
    print("\n📈 기본 통계:")
    metadata = analysis_data["experiment_metadata"]
    print(f"- 총 문서 수: {metadata['total_documents']}")
    print(f"- 실험 ID: {metadata['run_id']}")
    print(f"- 평가 모드: {metadata['evaluation_mode']}")

    # 2. 전략 분포
    print("\n🎯 전략 선택 분포:")
    strategy_dist = analysis_data["strategy_distribution"]
    strategy_pct = analysis_data["strategy_percentages"]

    for strategy, count in strategy_dist.items():
        percentage = strategy_pct[strategy]
        print(f"- {strategy}: {count}개 ({percentage:.1f}%)")

    # 3. 신뢰도 통계
    print("\n🔍 신뢰도 통계:")
    conf_stats = analysis_data["confidence_statistics"]
    print(f"- 평균: {conf_stats['mean']:.3f}")
    print(f"- 표준편차: {conf_stats['std']:.3f}")
    print(f"- 최소값: {conf_stats['min']:.3f}")
    print(f"- 최대값: {conf_stats['max']:.3f}")
    print(f"- 중간값: {conf_stats['median']:.3f}")

    # 4. 도메인별 분석
    print("\n🏷️ 도메인별 전략 선택:")
    domain_analysis = analysis_data["domain_analysis"]
    for domain, strategies in domain_analysis.items():
        print(f"\n  {domain.upper()} 도메인:")
        total = sum(strategies.values())
        for strategy, count in strategies.items():
            percentage = (count / total) * 100
            print(f"    - {strategy}: {count}개 ({percentage:.1f}%)")

    # 5. CSV 데이터 분석 (있는 경우)
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print("\n📊 상세 분석 (CSV 데이터 기반):")

        # 도메인별 평균 신뢰도
        domain_confidence = df.groupby('domain')['confidence'].agg(['mean', 'std', 'count'])
        print("\n도메인별 평균 신뢰도:")
        for domain, stats in domain_confidence.iterrows():
            print(f"  {domain}: {stats['mean']:.3f} (±{stats['std']:.3f}, n={stats['count']})")

        # 문서 길이별 분석
        print("\n📏 문서 길이별 분석:")
        df['length_category'] = pd.cut(df['length'], bins=3, labels=['Short', 'Medium', 'Long'])
        length_analysis = df.groupby('length_category')['selected_strategy'].value_counts()
        print(length_analysis)

    # 6. 시각화 생성
    create_visualizations(analysis_data, csv_file if csv_file.exists() else None)

    print("\n" + "=" * 60)
    print("분석 완료! 📊")
    print("그래프는 'analysis_plots' 폴더에 저장되었습니다.")
    print("=" * 60)

def create_visualizations(analysis_data, csv_file):
    """분석 결과 시각화"""

    # 출력 디렉토리 생성
    output_dir = Path("analysis_plots")
    output_dir.mkdir(exist_ok=True)

    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 전략 분포 파이 차트
    fig, ax = plt.subplots(figsize=(10, 8))
    strategy_dist = analysis_data["strategy_distribution"]

    colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_dist)))
    wedges, texts, autotexts = ax.pie(strategy_dist.values(),
                                      labels=strategy_dist.keys(),
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)

    ax.set_title('Strategy Selection Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 도메인별 전략 선택 히트맵
    domain_data = analysis_data["domain_analysis"]
    if domain_data:
        # 데이터 준비
        domains = list(domain_data.keys())
        all_strategies = set()
        for strategies in domain_data.values():
            all_strategies.update(strategies.keys())
        all_strategies = sorted(list(all_strategies))

        # 히트맵 데이터 생성
        heatmap_data = []
        for domain in domains:
            row = []
            for strategy in all_strategies:
                count = domain_data[domain].get(strategy, 0)
                row.append(count)
            heatmap_data.append(row)

        # 히트맵 그리기
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data,
                    xticklabels=all_strategies,
                    yticklabels=domains,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=ax)

        ax.set_title('Strategy Selection by Domain', fontsize=16, fontweight='bold')
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Domain', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "domain_strategy_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. CSV 데이터 기반 시각화 (있는 경우)
    if csv_file and Path(csv_file).exists():
        df = pd.read_csv(csv_file)

        # 신뢰도 분포 히스토그램
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Confidence Score Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 도메인별 신뢰도 박스플롯
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column='confidence', by='domain', ax=ax)
        ax.set_title('Confidence Score by Domain', fontsize=16, fontweight='bold')
        ax.set_xlabel('Domain', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        plt.suptitle('')  # 기본 제목 제거
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_by_domain.png", dpi=300, bbox_inches='tight')
        plt.close()

def print_paper_summary():
    """논문용 요약 출력"""

    print("\n" + "=" * 60)
    print("논문 작성용 요약")
    print("=" * 60)

    print("""
📝 논문에서 활용할 수 있는 내용:

1. 연구 목적
   - 텍스트 특성에 따른 청킹 전략의 자동 선택
   - 도메인별 최적 전략 패턴 분석
   - Router Agent의 전략 선택 신뢰도 평가

2. 실험 설계
   - 다중 데이터셋 활용 (SQUAD, NewsQA 등)
   - 각 데이터셋에서 100개 샘플 수집
   - 지능형 라우터를 통한 자동 전략 선택

3. 주요 결과
   - 전략별 선택 빈도 및 분포
   - 도메인별 최적 전략 패턴
   - 자동 선택 시스템의 신뢰도 분석

4. 활용 가능한 데이터
   - strategy_analysis.json: 전체 통계 및 메타데이터
   - strategy_decisions.csv: 문서별 상세 결정 과정
   - 시각화 자료: 분포도, 히트맵, 박스플롯

5. 논문 기여도
   - 도메인 특성에 따른 청킹 전략 자동 선택 방법론
   - 실제 데이터셋을 통한 검증 결과
   - 향후 RAG 시스템 최적화 방향 제시
    """)

if __name__ == "__main__":
    analyze_strategy_results()
    print_paper_summary()