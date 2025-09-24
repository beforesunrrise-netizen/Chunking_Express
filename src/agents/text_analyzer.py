"""
텍스트 분석기 - 텍스트의 특성을 분석하여 최적의 청킹 전략 추천을 위한 데이터 제공
Text Analyzer - Provides text characteristics for optimal chunking strategy recommendation
"""

import re
import statistics
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

from src.config import Language
from src.data_structures import Document


@dataclass
class TextCharacteristics:
    """텍스트 특성 정보"""
    # 기본 정보
    length: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float

    # 구조 정보
    has_headers: bool
    has_lists: bool
    has_tables: bool
    structure_score: float  # 0.0-1.0 (구조화 정도)

    # 복잡도 정보
    complexity_score: float  # 0.0-1.0 (복잡도)
    vocabulary_diversity: float  # 0.0-1.0 (어휘 다양성)

    # 도메인 정보
    domain_indicators: Dict[str, float]  # 도메인별 가능성 점수

    # 언어 정보
    language: Language

    # 기타
    has_technical_terms: bool
    readability_score: float  # 0.0-1.0 (가독성)

    # Document 참조 (도메인 정보 접근용)
    document: Optional[Document] = None


class TextAnalyzer:
    """텍스트 분석기"""

    def __init__(self, language: Language = Language.ENGLISH):
        self.language = language

        # 도메인 키워드 사전
        self.domain_keywords = {
            "technical": [
                "algorithm", "api", "database", "server", "code", "software", "programming",
                "system", "network", "protocol", "implementation", "framework", "library",
                "알고리즘", "데이터베이스", "서버", "프로그래밍", "시스템", "네트워크", "구현"
            ],
            "scientific": [
                "research", "study", "analysis", "hypothesis", "methodology", "experiment",
                "data", "result", "conclusion", "theory", "model", "variable",
                "연구", "분석", "가설", "실험", "데이터", "결과", "결론", "이론", "모델"
            ],
            "academic": [
                "chapter", "section", "introduction", "conclusion", "bibliography", "reference",
                "논문", "연구", "학술", "참고문헌", "서론", "결론", "장", "절"
            ],
            "business": [
                "market", "revenue", "profit", "strategy", "customer", "business", "company",
                "management", "sales", "financial", "investment", "growth",
                "시장", "매출", "수익", "전략", "고객", "기업", "회사", "경영", "투자", "성장"
            ],
            "news": [
                "report", "breaking", "according", "statement", "official", "source",
                "보도", "발표", "관계자", "소식", "뉴스", "기사", "취재"
            ],
            "narrative": [
                "story", "character", "plot", "chapter", "dialogue", "scene",
                "이야기", "인물", "줄거리", "대화", "장면", "소설"
            ]
        }

        # 기술 용어 패턴
        self.technical_patterns = [
            r'\b[A-Z]{2,}\b',  # 대문자 약어
            r'\b\w+\(\)',       # 함수 호출 패턴
            r'\b\d+\.\d+\.\d+\b',  # 버전 번호
            r'<[^>]+>',         # HTML/XML 태그
            r'@\w+',            # 어노테이션
            r'#\w+',            # 해시태그
            r'\$\w+',           # 변수
        ]

    async def analyze_document(self, document: Document) -> TextCharacteristics:
        """문서의 텍스트 특성을 분석합니다."""
        text = document.content
        logger.info(f"문서 {document.id}의 텍스트 특성 분석 시작")

        # 기본 정보 분석
        basic_info = self._analyze_basic_info(text)

        # 구조 분석
        structure_info = self._analyze_structure(text)

        # 복잡도 분석
        complexity_info = self._analyze_complexity(text)

        # 도메인 분석
        domain_info = self._analyze_domain(text)

        # 기술 용어 검사
        has_technical = self._check_technical_terms(text)

        # 가독성 분석
        readability = self._analyze_readability(text)

        characteristics = TextCharacteristics(
            # 기본 정보
            length=basic_info["length"],
            sentence_count=basic_info["sentence_count"],
            paragraph_count=basic_info["paragraph_count"],
            avg_sentence_length=basic_info["avg_sentence_length"],

            # 구조 정보
            has_headers=structure_info["has_headers"],
            has_lists=structure_info["has_lists"],
            has_tables=structure_info["has_tables"],
            structure_score=structure_info["structure_score"],

            # 복잡도 정보
            complexity_score=complexity_info["complexity_score"],
            vocabulary_diversity=complexity_info["vocabulary_diversity"],

            # 도메인 정보
            domain_indicators=domain_info,

            # 언어 정보
            language=document.language,

            # 기타
            has_technical_terms=has_technical,
            readability_score=readability,

            # Document 참조 (도메인 정보 접근용)
            document=document
        )

        logger.info(f"문서 {document.id} 분석 완료: "
                   f"길이={characteristics.length}, "
                   f"구조점수={characteristics.structure_score:.2f}, "
                   f"복잡도={characteristics.complexity_score:.2f}")

        return characteristics

    def _analyze_basic_info(self, text: str) -> Dict[str, Any]:
        """기본 정보 분석"""
        # 문장 분할
        if self.language == Language.KOREAN:
            sentences = re.split(r'[.!?]+\s*', text)
        else:
            sentences = re.split(r'[.!?]+\s+', text)

        sentences = [s.strip() for s in sentences if s.strip()]

        # 단락 분할
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # 평균 문장 길이
        avg_sentence_length = (
            sum(len(s) for s in sentences) / len(sentences)
            if sentences else 0
        )

        return {
            "length": len(text),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "avg_sentence_length": avg_sentence_length
        }

    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """구조 분석"""
        # 헤더 검사 (마크다운 스타일, 번호 스타일 등)
        header_patterns = [
            r'^#{1,6}\s+.+$',     # 마크다운 헤더
            r'^\d+\.\s+.+$',      # 번호 헤더
            r'^[A-Z][^.!?]*:$',   # 제목 스타일
            r'^\s*=+\s*$',        # 구분선
            r'^\s*-{3,}\s*$',     # 구분선
        ]

        has_headers = any(
            re.search(pattern, text, re.MULTILINE)
            for pattern in header_patterns
        )

        # 리스트 검사
        list_patterns = [
            r'^\s*[-*+]\s+.+$',   # 불릿 리스트
            r'^\s*\d+\.\s+.+$',   # 번호 리스트
            r'^\s*[a-zA-Z]\.\s+.+$',  # 알파벳 리스트
        ]

        has_lists = any(
            re.search(pattern, text, re.MULTILINE)
            for pattern in list_patterns
        )

        # 테이블 검사
        table_patterns = [
            r'\|.*\|',            # 파이프 테이블
            r'\t.*\t',            # 탭 구분 테이블
            r'^\s*\+[-=+]+\+',    # ASCII 테이블
        ]

        has_tables = any(
            re.search(pattern, text, re.MULTILINE)
            for pattern in table_patterns
        )

        # 구조 점수 계산 (0.0-1.0)
        structure_indicators = [
            has_headers,
            has_lists,
            has_tables,
            len(text.split('\n\n')) > 3,  # 여러 단락
            bool(re.search(r'\n\s*\n', text)),  # 빈 줄로 구분
        ]

        structure_score = sum(structure_indicators) / len(structure_indicators)

        return {
            "has_headers": has_headers,
            "has_lists": has_lists,
            "has_tables": has_tables,
            "structure_score": structure_score
        }

    def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """복잡도 분석"""
        words = re.findall(r'\b\w+\b', text.lower())

        if not words:
            return {"complexity_score": 0.0, "vocabulary_diversity": 0.0}

        # 어휘 다양성 (고유 단어 비율)
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words)

        # 문장 복잡도 지표들
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {"complexity_score": 0.0, "vocabulary_diversity": vocabulary_diversity}

        # 평균 문장 길이 (단어 수)
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        avg_sentence_word_count = statistics.mean(sentence_lengths) if sentence_lengths else 0

        # 긴 단어 비율 (6글자 이상)
        long_words = [w for w in words if len(w) >= 6]
        long_word_ratio = len(long_words) / len(words) if words else 0

        # 복잡한 문장 패턴 (접속사, 관계사 등)
        complex_patterns = [
            r'\b(however|therefore|furthermore|moreover|nevertheless)\b',
            r'\b(because|although|unless|whereas|while)\b',
            r'\b(그러나|따라서|또한|그럼에도|비록)\b',
            r'\b(때문에|하지만|그런데|그러므로)\b',
        ]

        complex_indicators = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in complex_patterns
        )

        # 복잡도 점수 계산 (0.0-1.0)
        complexity_factors = [
            min(avg_sentence_word_count / 20, 1.0),  # 문장 길이 (20단어 기준)
            min(long_word_ratio * 2, 1.0),           # 긴 단어 비율
            min(complex_indicators / len(sentences), 1.0),  # 복잡한 패턴 비율
            min(vocabulary_diversity * 1.5, 1.0),    # 어휘 다양성
        ]

        complexity_score = statistics.mean(complexity_factors)

        return {
            "complexity_score": complexity_score,
            "vocabulary_diversity": vocabulary_diversity
        }

    def _analyze_domain(self, text: str) -> Dict[str, float]:
        """도메인 분석"""
        text_lower = text.lower()
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            # 각 도메인별 키워드 매칭 점수
            matches = sum(
                len(re.findall(rf'\b{re.escape(keyword.lower())}\b', text_lower))
                for keyword in keywords
            )

            # 정규화 (문서 길이 대비)
            word_count = len(re.findall(r'\b\w+\b', text))
            normalized_score = min(matches / max(word_count / 100, 1), 1.0)
            domain_scores[domain] = normalized_score

        return domain_scores

    def _check_technical_terms(self, text: str) -> bool:
        """기술 용어 검사"""
        for pattern in self.technical_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _analyze_readability(self, text: str) -> float:
        """가독성 분석 (간단한 지표)"""
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not words or not sentences:
            return 0.0

        # 평균 문장 당 단어 수
        avg_words_per_sentence = len(words) / len(sentences)

        # 짧은 단어 비율 (4글자 이하)
        short_words = [w for w in words if len(w) <= 4]
        short_word_ratio = len(short_words) / len(words)

        # 가독성 점수 (0.0-1.0, 높을수록 읽기 쉬움)
        readability_factors = [
            max(1.0 - (avg_words_per_sentence - 15) / 15, 0.0),  # 문장 길이
            short_word_ratio,  # 짧은 단어 비율
        ]

        return statistics.mean(readability_factors)

    def get_analysis_summary(self, characteristics: TextCharacteristics) -> str:
        """분석 결과 요약"""
        domain_max = max(characteristics.domain_indicators.items(), key=lambda x: x[1])

        return f"""
텍스트 분석 결과:
- 길이: {characteristics.length:,}자
- 문장 수: {characteristics.sentence_count}개
- 평균 문장 길이: {characteristics.avg_sentence_length:.1f}자
- 구조화 점수: {characteristics.structure_score:.2f}/1.0
- 복잡도 점수: {characteristics.complexity_score:.2f}/1.0
- 가독성 점수: {characteristics.readability_score:.2f}/1.0
- 주요 도메인: {domain_max[0]} ({domain_max[1]:.2f})
- 기술 용어 포함: {"예" if characteristics.has_technical_terms else "아니오"}
"""