"""
BM25 기반 키워드 청킹 전략
BM25-based keyword chunking strategy

추가예정...

"""

from typing import List, Dict, Set
import re
import math
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from loguru import logger

from src.data_structures import Chunk, Query
from .base_chunker import BaseChunker

# NLTK 데이터 다운로드 (최초 실행 시)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("NLTK 데이터 다운로드 중...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class BM25KeywordChunker(BaseChunker):
    """BM25 알고리즘을 사용한 키워드 기반 청킹"""

    def __init__(self,
                 k1: float = 1.5,
                 b: float = 0.75,
                 min_score_threshold: float = 0.1,
                 max_chunks: int = 10,
                 use_stemming: bool = True,
                 language: str = 'english'):
        """
        Args:
            k1: BM25 파라미터 (term frequency saturation)
            b: BM25 파라미터 (length normalization)
            min_score_threshold: 최소 BM25 점수 임계값
            max_chunks: 반환할 최대 청크 수
            use_stemming: 어간 추출 사용 여부
            language: 언어 ('english' 또는 'korean')
        """
        self.k1 = k1
        self.b = b
        self.min_score_threshold = min_score_threshold
        self.max_chunks = max_chunks
        self.use_stemming = use_stemming
        self.language = language

        # NLTK 설정
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if use_stemming else None

        # 한국어 불용어 추가
        if language == 'korean':
            korean_stopwords = {
                '그', '저', '이', '것', '수', '등', '들', '만', '더', '도', '를', '의', '가', '에',
                '는', '은', '을', '로', '으로', '하다', '있다', '되다', '같다', '없다', '아니다',
                '그렇다', '이렇다', '그런', '이런', '저런', '어떤', '무엇', '누구', '언제', '어디'
            }
            self.stop_words.update(korean_stopwords)

        # BM25 계산을 위한 전처리된 데이터
        self.doc_freq: Dict[str, int] = {}  # 문서 빈도
        self.avg_doc_length: float = 0.0
        self.doc_lengths: Dict[str, int] = {}  # 각 문서(청크)의 길이
        self.tokenized_docs: Dict[str, List[str]] = {}  # 토큰화된 문서들
        self.is_indexed: bool = False

    async def get_relevant_chunks(self, query: Query, chunks: List[Chunk]) -> List[Chunk]:
        """BM25 점수를 기반으로 관련 청크 검색"""
        try:
            # 처음 호출 시 인덱스 구축
            if not self.is_indexed:
                await self._build_bm25_index(chunks)

            # 쿼리 토큰화 및 키워드 추출
            query_keywords = self._extract_keywords(query.text)
            logger.info(f"쿼리 키워드: {query_keywords}")

            if not query_keywords:
                logger.warning("추출된 키워드가 없습니다.")
                return chunks[:self.max_chunks]

            # 각 청크에 대해 BM25 점수 계산
            chunk_scores = []
            for chunk in chunks:
                score = self._calculate_bm25_score(query_keywords, chunk.id)
                if score >= self.min_score_threshold:
                    chunk_scores.append((score, chunk))

            # 점수 기준 내림차순 정렬
            chunk_scores.sort(key=lambda x: x[0], reverse=True)

            # 상위 청크들 선택
            selected_chunks = []
            for score, chunk in chunk_scores[:self.max_chunks]:
                # 메타데이터에 BM25 점수 추가
                chunk.metadata["bm25_score"] = score
                chunk.metadata["relevance_score"] = score
                selected_chunks.append(chunk)

            logger.info(
                f"BM25로 {len(selected_chunks)}개 청크 선택 (평균 점수: {sum(s for s, _ in chunk_scores[:len(selected_chunks)]) / len(selected_chunks) if selected_chunks else 0:.3f})")

            return selected_chunks

        except Exception as e:
            logger.error(f"BM25 청크 검색 실패: {e}")
            # 폴백: 기본 키워드 매칭
            return await self._fallback_keyword_matching(query, chunks)

    async def _build_bm25_index(self, chunks: List[Chunk]):
        """BM25 인덱스 구축"""
        logger.info(f"BM25 인덱스 구축 시작: {len(chunks)}개 청크")

        # 각 청크 토큰화
        total_length = 0
        word_doc_count = defaultdict(int)

        for chunk in chunks:
            # 청크 내용 토큰화
            tokens = self._tokenize(chunk.content)
            self.tokenized_docs[chunk.id] = tokens
            self.doc_lengths[chunk.id] = len(tokens)
            total_length += len(tokens)

            # 단어별 문서 빈도 계산
            unique_words = set(tokens)
            for word in unique_words:
                word_doc_count[word] += 1

        # 평균 문서 길이 계산
        self.avg_doc_length = total_length / len(chunks) if chunks else 0

        # 문서 빈도 저장
        self.doc_freq = dict(word_doc_count)

        self.is_indexed = True
        logger.info(f"BM25 인덱스 구축 완료: 평균 문서 길이 {self.avg_doc_length:.1f}, 고유 단어 {len(self.doc_freq)}개")

    def _calculate_bm25_score(self, query_keywords: List[str], doc_id: str) -> float:
        """특정 문서에 대한 BM25 점수 계산"""
        if doc_id not in self.tokenized_docs:
            return 0.0

        doc_tokens = self.tokenized_docs[doc_id]
        doc_length = self.doc_lengths[doc_id]
        doc_word_count = Counter(doc_tokens)
        total_docs = len(self.tokenized_docs)

        score = 0.0

        for keyword in query_keywords:
            # 키워드가 문서에 없으면 건너뛰기
            if keyword not in doc_word_count:
                continue

            # TF (Term Frequency)
            tf = doc_word_count[keyword]

            # IDF (Inverse Document Frequency)
            doc_freq = self.doc_freq.get(keyword, 0)
            if doc_freq == 0:
                continue

            idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

            # BM25 점수 계산
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            term_score = idf * (numerator / denominator)
            score += term_score

        return score

    def _extract_keywords(self, text: str) -> List[str]:
        """NLTK를 사용한 고급 키워드 추출"""
        try:
            # NLTK 토큰화
            tokens = word_tokenize(text.lower())

            # 알파벳 문자만 포함하는 토큰 필터링
            tokens = [token for token in tokens if token.isalpha()]

            # 불용어 제거
            tokens = [token for token in tokens if token not in self.stop_words]

            # 길이 필터링 (2글자 이상)
            tokens = [token for token in tokens if len(token) >= 2]

            # 어간 추출 (옵션)
            if self.use_stemming and self.stemmer:
                tokens = [self.stemmer.stem(token) for token in tokens]

            # 중복 제거하되 순서 유지
            seen = set()
            keywords = []
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    keywords.append(token)

            return keywords

        except Exception as e:
            logger.warning(f"NLTK 키워드 추출 실패: {e}, 기본 방식 사용")
            return self._fallback_tokenize(text)

    def _fallback_tokenize(self, text: str) -> List[str]:
        """NLTK 실패 시 폴백 토큰화"""
        # 기본적인 토큰화
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.lower().split()

        # 기본 영어 불용어
        basic_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'this', 'that', 'these', 'those'
        }

        return [token for token in tokens
                if token not in basic_stopwords and len(token) >= 2]

    def _tokenize(self, text: str) -> List[str]:
        """NLTK를 사용한 토큰화"""
        try:
            # NLTK 토큰화
            tokens = word_tokenize(text.lower())

            # 알파벳 문자만 포함하는 토큰 필터링
            tokens = [token for token in tokens if token.isalpha() and len(token) >= 1]

            # 어간 추출 (옵션)
            if self.use_stemming and self.stemmer:
                tokens = [self.stemmer.stem(token) for token in tokens]

            return tokens

        except Exception as e:
            logger.warning(f"NLTK 토큰화 실패: {e}, 기본 방식 사용")
            # 폴백: 기본 토큰화
            text = re.sub(r'[^\w\s]', ' ', text)
            return [token for token in text.lower().split() if token.strip()]

    async def _fallback_keyword_matching(self, query: Query, chunks: List[Chunk]) -> List[Chunk]:
        """BM25 실패 시 폴백 키워드 매칭"""
        logger.info("BM25 폴백: 기본 키워드 매칭 사용")

        query_keywords = self._extract_keywords(query.text)
        if not query_keywords:
            return chunks[:self.max_chunks]

        chunk_scores = []
        for chunk in chunks:
            score = self._simple_keyword_score(query_keywords, chunk.content)
            if score > 0:
                chunk.metadata["keyword_score"] = score
                chunk.metadata["relevance_score"] = score
                chunk_scores.append((score, chunk))

        # 점수 기준 정렬
        chunk_scores.sort(key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in chunk_scores[:self.max_chunks]]

    def _simple_keyword_score(self, keywords: List[str], text: str) -> float:
        """간단한 키워드 매칭 점수"""
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return matches / len(keywords) if keywords else 0.0


class EnhancedBM25KeywordChunker(BM25KeywordChunker):
    """향상된 BM25 키워드 청커 (추가 기능들)"""

    def __init__(self,
                 k1: float = 1.5,
                 b: float = 0.75,
                 min_score_threshold: float = 0.1,
                 max_chunks: int = 10,
                 use_stemming: bool = True,
                 language: str = 'english',
                 use_phrase_matching: bool = True,
                 boost_exact_matches: float = 1.5,
                 use_ngrams: bool = True):
        super().__init__(k1, b, min_score_threshold, max_chunks, use_stemming, language)
        self.use_phrase_matching = use_phrase_matching
        self.boost_exact_matches = boost_exact_matches
        self.use_ngrams = use_ngrams

    def _calculate_bm25_score(self, query_keywords: List[str], doc_id: str) -> float:
        """향상된 BM25 점수 계산 (구문 매칭 및 정확한 매칭 부스트)"""
        base_score = super()._calculate_bm25_score(query_keywords, doc_id)

        if doc_id not in self.tokenized_docs:
            return base_score

        doc_content = ' '.join(self.tokenized_docs[doc_id])

        # 정확한 구문 매칭 보너스
        if self.use_phrase_matching and len(query_keywords) > 1:
            query_phrase = ' '.join(query_keywords[:3])  # 최대 3단어 구문
            if query_phrase.lower() in doc_content.lower():
                base_score *= self.boost_exact_matches
                logger.debug(f"정확한 구문 매칭 보너스 적용: {doc_id}")

        # 개별 키워드 정확 매칭 보너스
        exact_matches = 0
        for keyword in query_keywords:
            if keyword.lower() in doc_content.lower():
                exact_matches += 1

        if exact_matches > 0:
            exact_match_ratio = exact_matches / len(query_keywords)
            base_score += exact_match_ratio * 0.1  # 작은 보너스

        return base_score

    def _extract_keywords(self, text: str) -> List[str]:
        """향상된 키워드 추출 (N-gram 지원)"""
        # 기본 키워드 추출
        basic_keywords = super()._extract_keywords(text)

        if not self.use_ngrams or len(basic_keywords) < 2:
            return basic_keywords

        # 2-gram과 3-gram 추가 (중요한 구문 캡처)
        ngrams = []

        # Bigrams
        for i in range(len(basic_keywords) - 1):
            bigram = f"{basic_keywords[i]} {basic_keywords[i + 1]}"
            ngrams.append(bigram)

        # Trigrams (상위 키워드만)
        for i in range(min(len(basic_keywords) - 2, 3)):  # 최대 3개의 트라이그램
            trigram = f"{basic_keywords[i]} {basic_keywords[i + 1]} {basic_keywords[i + 2]}"
            ngrams.append(trigram)

        # 원본 키워드와 N-gram 결합 (N-gram은 제한적으로)
        return basic_keywords + ngrams[:5]  # 최대 5개의 N-gram만 추가