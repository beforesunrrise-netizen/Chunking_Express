# 청킹 전략 자동 분류 실험 가이드

이 실험은 Router Agent가 텍스트의 특성을 분석하여 각 문서에 최적의 청킹 전략을 자동으로 선택하는 시스템을 평가합니다.

## 🎯 실험 목적

1. **자동 전략 선택**: Router Agent가 문서의 도메인, 복잡도, 구조 등을 분석하여 최적의 청킹 전략을 자동 선택
2. **도메인별 패턴 분석**: 각 도메인(뉴스, 위키피디아, 서사, 기술, 의료 등)에서 선호되는 청킹 전략 패턴 발견
3. **성능 평가**: 자동 선택된 전략들의 검색 성능 비교 및 분석

## 📋 실험 설정

### 데이터셋
- **SQUAD**: 위키피디아 기반 질의응답 (100개 샘플)
- **NewsQA**: 뉴스 기사 기반 질의응답 (100개 샘플)
- 추가 가능: NarrativeQA, TechQA, COVID-QA

### 청킹 전략
1. **Semantic Chunking**: 의미 단위 기반 (고품질, 고비용)
2. **Keyword Chunking**: 키워드 기반 구조화 (중품질, 중비용)
3. **Query-Aware Chunking**: 질의 특화 (고품질, 고비용)
4. **Fixed-Size Chunking**: 고정 크기 (기준선, 무비용)
5. **Recursive Chunking**: 재귀적 계층화 (중품질, 저비용)
6. **Text Similarity**: 유사도 기반 (중품질, 중비용)

## 🚀 실험 실행

### 1. 기본 실험 실행
```bash
python run_paper_experiment.py
```

### 2. 수동 실험 실행
```bash
python src/main.py \
    --use_multi_datasets \
    --datasets squad newsqa \
    --samples_per_dataset 100 \
    --mode retrieval \
    --use_intelligent_chunking \
    --intelligent_mode auto_select \
    --chunking_context balanced \
    --enable_embedding_storage
```

### 3. 다양한 컨텍스트로 실험
```bash
# 품질 중심
python src/main.py --use_intelligent_chunking --chunking_context quality_focused --intelligent_mode auto_select

# 속도 중심
python src/main.py --use_intelligent_chunking --chunking_context speed_focused --intelligent_mode auto_select

# 비용 중심
python src/main.py --use_intelligent_chunking --chunking_context cost_conscious --intelligent_mode auto_select
```

## 📊 결과 분석

### 1. 자동 분석 실행
```bash
python analyze_results.py
```

### 2. 생성되는 결과 파일들

#### JSON 결과 (`results/strategy_analysis/exp_*/strategy_analysis.json`)
```json
{
    "experiment_metadata": {
        "run_id": "exp_20241224_143022",
        "total_documents": 200,
        "evaluation_mode": "retrieval"
    },
    "strategy_distribution": {
        "semantic": 85,
        "keyword": 45,
        "query_aware": 40,
        "fixed_size": 20,
        "recursive": 8,
        "text_similarity": 2
    },
    "strategy_percentages": {
        "semantic": 42.5,
        "keyword": 22.5,
        "query_aware": 20.0
    },
    "confidence_statistics": {
        "mean": 0.78,
        "std": 0.15,
        "min": 0.42,
        "max": 0.95
    },
    "domain_analysis": {
        "wikipedia": {
            "query_aware": 35,
            "keyword": 30,
            "semantic": 25
        },
        "news": {
            "keyword": 40,
            "semantic": 35,
            "recursive": 15
        }
    }
}
```

#### CSV 결과 (`results/strategy_analysis/exp_*/strategy_decisions.csv`)
| doc_id | domain | length | selected_strategy | confidence | reasoning |
|--------|--------|--------|-------------------|------------|-----------|
| squad_0 | wikipedia | 1250 | query_aware | 0.85 | 질의 최적화 필요, 구조화된 문서 |
| newsqa_0 | news | 980 | keyword | 0.78 | 키워드 기반 구조화, 뉴스 도메인 특성 |

### 3. 시각화 결과
- `analysis_plots/strategy_distribution.png`: 전략 선택 분포 파이 차트
- `analysis_plots/domain_strategy_heatmap.png`: 도메인별 전략 선택 히트맵
- `analysis_plots/confidence_distribution.png`: 신뢰도 점수 분포
- `analysis_plots/confidence_by_domain.png`: 도메인별 신뢰도 박스플롯

## 📈 예상 결과 및 분석 포인트

### 1. 도메인별 전략 선호도
- **Wikipedia (SQUAD)**: Query-Aware > Keyword > Semantic
- **News (NewsQA)**: Keyword > Semantic > Recursive

### 2. 신뢰도 분석
- 평균 신뢰도: 0.75-0.85 예상
- 도메인별 신뢰도 차이
- 문서 길이에 따른 신뢰도 변화

### 3. 성능 비교
- 자동 선택 vs 단일 전략 성능
- MRR (Mean Reciprocal Rank) 비교
- Hit@K 점수 분석

## 📝 논문 작성 활용 방안

### 1. 연구 기여도
- **방법론**: 텍스트 특성 기반 자동 청킹 전략 선택
- **실험**: 다중 도메인 데이터셋을 통한 검증
- **결과**: 도메인별 최적 전략 패턴 발견

### 2. 활용 가능한 통계
- 전략별 선택 빈도 및 성능
- 도메인별 최적화 패턴
- 자동 선택 시스템의 정확도

### 3. 시각화 자료
- 전략 분포도 → 논문 Figure 1
- 도메인-전략 히트맵 → 논문 Figure 2
- 성능 비교 차트 → 논문 Figure 3

### 4. 논문 구조 제안
1. **Introduction**: RAG 시스템에서 청킹 전략의 중요성
2. **Methodology**: Router Agent 아키텍처 및 전략 선택 알고리즘
3. **Experiments**: 다중 데이터셋 실험 설계
4. **Results**: 도메인별 전략 선호도 및 성능 분석
5. **Discussion**: 자동 선택의 효과성 및 한계점
6. **Conclusion**: 향후 RAG 시스템 최적화 방향

## 🔧 문제 해결

### 의존성 설치
```bash
pip install pandas matplotlib seaborn numpy datasets
```

### GPU 메모리 부족 시
```bash
# 샘플 수 줄이기
python src/main.py --samples_per_dataset 50

# 비용 절약 모드
python src/main.py --chunking_context cost_conscious
```

### API 비용 절약
```bash
# API 사용하지 않는 전략만 사용
python src/main.py --chunking_context cost_conscious --intelligent_mode auto_select
```

## 📞 지원

실험 관련 문제가 있으면 다음을 확인하세요:
1. OpenAI API 키 설정 확인
2. 필요한 Python 패키지 설치 확인
3. 실험 로그 파일 검토
4. 결과 디렉토리 권한 확인

이 실험을 통해 얻은 데이터는 청킹 전략 자동 선택에 관한 논문 작성에 직접 활용할 수 있습니다.