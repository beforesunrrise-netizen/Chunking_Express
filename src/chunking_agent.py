"""
LangGraph 기반 적응형 청킹 에이전트
문서 카테고리와 길이에 따라 최적의 청킹 전략을 동적으로 선택
"""

from typing import Dict, List, Any, Optional, TypedDict, Literal
from enum import Enum
from langgraph.graph import StateGraph, END

from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import json
import asyncio
from pathlib import Path

# 데이터 로더 임포트
from src.data.data_loader import StandardizedDocument, DocumentCategory


class ChunkingStrategy(str, Enum):
    """청킹 전략 열거형"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    QUERY_AWARE = "query_aware"
    RECURSIVE = "recursive"
    TEXT_SIMILARITY = "text_similarity"
    HYBRID = "hybrid"  # 여러 전략 조합


@dataclass
class ChunkingConfig:
    """청킹 설정"""
    strategy: ChunkingStrategy
    parameters: Dict[str, Any]
    reasoning: str = ""
    confidence_score: float = 0.0


class DocumentState(TypedDict):
    """LangGraph 상태 정의"""
    # 입력 데이터
    document: StandardizedDocument
    raw_data: Dict[str, Any]

    # 분석 결과
    document_analysis: Dict[str, Any]
    category: str
    content_length: int
    complexity_score: float

    # 청킹 결정
    selected_strategy: Optional[ChunkingStrategy]
    chunking_config: Optional[ChunkingConfig]
    backup_strategies: List[ChunkingStrategy]

    # 실행 결과
    chunks: Optional[List[Dict[str, Any]]]
    chunk_quality_scores: Dict[str, float]

    # 메타데이터
    processing_time: Dict[str, float]
    errors: List[str]
    warnings: List[str]

    # 학습 데이터
    performance_history: List[Dict[str, Any]]
    feedback: Optional[Dict[str, Any]]


class DocumentAnalyzer:
    """문서 분석 노드"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)

    async def analyze(self, state: DocumentState) -> DocumentState:
        """문서 심층 분석"""
        doc = state["document"]

        # 기본 통계
        content = doc.content
        words = content.split()
        sentences = content.split('.')

        # 복잡도 계산
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0

        # 구조 분석
        has_sections = bool('\n\n' in content or '\n#' in content)
        has_lists = bool('- ' in content or '* ' in content or '1. ' in content)
        has_code = bool('```' in content or 'def ' in content or 'function ' in content)

        # LLM 기반 심화 분석
        analysis_prompt = f"""
        Analyze this document for optimal chunking:

        Category: {doc.category.value}
        Length: {len(content)} characters
        Preview: {content[:500]}...

        Determine:
        1. Information density (low/medium/high)
        2. Structure type (linear/hierarchical/mixed)
        3. Topic coherence (single/multi/scattered)
        4. Optimal chunk size range
        5. Key semantic boundaries

        Return as JSON with scores 0-1 for each aspect.
        """

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a document structure analyst."),
                HumanMessage(content=analysis_prompt)
            ])

            # LLM 응답 파싱
            llm_analysis = self._parse_llm_response(response.content)
        except Exception as e:
            state["warnings"].append(f"LLM analysis failed: {e}")
            llm_analysis = {}

        # 복잡도 점수 계산 (0-1)
        complexity_score = min(1.0, (
                (avg_word_length / 10) * 0.2 +
                (avg_sentence_length / 30) * 0.3 +
                (1 if has_sections else 0) * 0.2 +
                (1 if has_code else 0) * 0.3
        ))

        state["document_analysis"] = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "has_sections": has_sections,
            "has_lists": has_lists,
            "has_code": has_code,
            "llm_analysis": llm_analysis
        }

        state["complexity_score"] = complexity_score
        state["category"] = doc.category.value
        state["content_length"] = len(content)

        return state

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        try:
            # JSON 블록 추출
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                json_str = json_match.group(0) if json_match else "{}"
            else:
                return {}

            return json.loads(json_str)
        except:
            return {}


class StrategySelector:
    """전략 선택 노드"""

    def __init__(self):
        # 카테고리별 전략 매핑 (기본값)
        self.strategy_rules = {
            DocumentCategory.SCIENCE.value: {
                "high_complexity": [ChunkingStrategy.SEMANTIC, ChunkingStrategy.RECURSIVE],
                "medium_complexity": [ChunkingStrategy.SEMANTIC],
                "low_complexity": [ChunkingStrategy.FIXED_SIZE]
            },
            DocumentCategory.TECHNOLOGY.value: {
                "high_complexity": [ChunkingStrategy.KEYWORD, ChunkingStrategy.SEMANTIC],
                "medium_complexity": [ChunkingStrategy.KEYWORD],
                "low_complexity": [ChunkingStrategy.FIXED_SIZE]
            },
            DocumentCategory.LITERATURE.value: {
                "high_complexity": [ChunkingStrategy.TEXT_SIMILARITY, ChunkingStrategy.SEMANTIC],
                "medium_complexity": [ChunkingStrategy.SEMANTIC],
                "low_complexity": [ChunkingStrategy.FIXED_SIZE]
            },
            DocumentCategory.MEDICAL.value: {
                "high_complexity": [ChunkingStrategy.SEMANTIC, ChunkingStrategy.KEYWORD],
                "medium_complexity": [ChunkingStrategy.SEMANTIC],
                "low_complexity": [ChunkingStrategy.KEYWORD]
            },
            DocumentCategory.NEWS.value: {
                "high_complexity": [ChunkingStrategy.SEMANTIC],
                "medium_complexity": [ChunkingStrategy.FIXED_SIZE],
                "low_complexity": [ChunkingStrategy.FIXED_SIZE]
            }
        }

        # 길이별 파라미터
        self.length_params = {
            "short": {"chunk_size": 256, "overlap": 50},
            "medium": {"chunk_size": 512, "overlap": 100},
            "long": {"chunk_size": 1024, "overlap": 200}
        }

    async def select(self, state: DocumentState) -> DocumentState:
        """최적 전략 선택"""
        category = state["category"]
        complexity = state["complexity_score"]
        length = state["content_length"]

        # 복잡도 레벨 결정
        if complexity > 0.7:
            complexity_level = "high_complexity"
        elif complexity > 0.4:
            complexity_level = "medium_complexity"
        else:
            complexity_level = "low_complexity"

        # 길이 카테고리
        if length < 5000:
            length_cat = "short"
        elif length < 20000:
            length_cat = "medium"
        else:
            length_cat = "long"

        # 기본 전략 선택
        category_rules = self.strategy_rules.get(
            category,
            self.strategy_rules[DocumentCategory.GENERAL.value]
        )

        strategies = category_rules.get(
            complexity_level,
            [ChunkingStrategy.FIXED_SIZE]
        )

        # 성능 히스토리 기반 조정
        if state["performance_history"]:
            strategies = self._adjust_by_history(
                strategies,
                state["performance_history"],
                category,
                complexity_level
            )

        # 주 전략과 백업 전략 설정
        primary_strategy = strategies[0]
        backup_strategies = strategies[1:] if len(strategies) > 1 else []

        # 파라미터 설정
        base_params = self.length_params[length_cat].copy()

        # 전략별 특수 파라미터
        strategy_specific_params = self._get_strategy_params(
            primary_strategy,
            state["document_analysis"]
        )

        params = {**base_params, **strategy_specific_params}

        # 설정 객체 생성
        config = ChunkingConfig(
            strategy=primary_strategy,
            parameters=params,
            reasoning=f"Selected {primary_strategy.value} for {category} document with {complexity_level} ({complexity:.2f}) and {length_cat} length",
            confidence_score=self._calculate_confidence(state, primary_strategy)
        )

        state["selected_strategy"] = primary_strategy
        state["chunking_config"] = config
        state["backup_strategies"] = backup_strategies

        return state

    def _adjust_by_history(
            self,
            strategies: List[ChunkingStrategy],
            history: List[Dict],
            category: str,
            complexity: str
    ) -> List[ChunkingStrategy]:
        """과거 성능 기반 전략 조정"""
        scores = {}

        for record in history:
            if (record.get("category") == category and
                    record.get("complexity_level") == complexity):

                strategy = record.get("strategy")
                if strategy:
                    score = record.get("performance_score", 0)
                    if strategy not in scores:
                        scores[strategy] = []
                    scores[strategy].append(score)

        # 평균 점수 계산
        avg_scores = {
            s: np.mean(scores.get(s, [0.5]))
            for s in strategies
        }

        # 점수 기준 정렬
        return sorted(strategies, key=lambda s: avg_scores.get(s, 0.5), reverse=True)

    def _get_strategy_params(
            self,
            strategy: ChunkingStrategy,
            analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """전략별 특수 파라미터"""
        params = {}

        if strategy == ChunkingStrategy.SEMANTIC:
            params["similarity_threshold"] = 0.75
            params["min_chunk_size"] = 100

        elif strategy == ChunkingStrategy.KEYWORD:
            params["max_keywords"] = 10
            params["keyword_overlap"] = 2

        elif strategy == ChunkingStrategy.RECURSIVE:
            params["depth_limit"] = 3
            params["min_split_size"] = 100

        elif strategy == ChunkingStrategy.TEXT_SIMILARITY:
            params["window_size"] = 5
            params["stride"] = 2

        return params

    def _calculate_confidence(
            self,
            state: DocumentState,
            strategy: ChunkingStrategy
    ) -> float:
        """전략 선택 신뢰도 계산"""
        confidence = 0.5  # 기본값

        # 카테고리 매칭
        if state["category"] in self.strategy_rules:
            confidence += 0.2

        # 복잡도 분석 완료
        if state["document_analysis"]:
            confidence += 0.15

        # LLM 분석 포함
        if state["document_analysis"].get("llm_analysis"):
            confidence += 0.15

        return min(1.0, confidence)


class ChunkingExecutor:
    """청킹 실행 노드"""

    def __init__(self):
        self.chunkers = {}  # 실제 chunker 인스턴스 캐시

    async def execute(self, state: DocumentState) -> DocumentState:
        """청킹 실행"""
        import time
        start_time = time.time()

        try:
            config = state["chunking_config"]
            strategy = config.strategy
            params = config.parameters

            # 실제 청킹 수행 (기존 chunker 클래스 사용)
            chunks = await self._perform_chunking(
                state["document"],
                strategy,
                params
            )

            # 청크 품질 평가
            quality_scores = self._evaluate_chunks(chunks)

            state["chunks"] = [self._chunk_to_dict(c) for c in chunks]
            state["chunk_quality_scores"] = quality_scores

            # 품질이 낮으면 백업 전략 시도
            if quality_scores.get("overall", 0) < 0.5 and state["backup_strategies"]:
                state["warnings"].append(f"Low quality score for {strategy.value}, trying backup")
                backup_strategy = state["backup_strategies"][0]

                # 백업 전략으로 재시도
                backup_config = ChunkingConfig(
                    strategy=backup_strategy,
                    parameters=params,
                    reasoning="Backup strategy due to low quality"
                )
                state["chunking_config"] = backup_config
                state["selected_strategy"] = backup_strategy

                # 재귀 호출
                return await self.execute(state)

        except Exception as e:
            state["errors"].append(f"Chunking failed: {e}")
            state["chunks"] = []
            state["chunk_quality_scores"] = {"overall": 0}

        state["processing_time"]["chunking"] = time.time() - start_time
        return state

    async def _perform_chunking(
            self,
            doc: StandardizedDocument,
            strategy: ChunkingStrategy,
            params: Dict[str, Any]
    ) -> List[Any]:
        """실제 청킹 수행"""
        # 여기서는 간단한 구현
        # 실제로는 기존 chunker 클래스들을 import해서 사용

        if strategy == ChunkingStrategy.FIXED_SIZE:
            chunk_size = params.get("chunk_size", 512)
            overlap = params.get("overlap", 50)
            return self._fixed_size_chunking(doc.content, chunk_size, overlap)

        elif strategy == ChunkingStrategy.SEMANTIC:
            # SemanticChunker 사용
            return self._semantic_chunking(doc.content, params)

        # 다른 전략들도 유사하게 구현
        else:
            return self._fixed_size_chunking(doc.content, 512, 50)

    def _fixed_size_chunking(
            self,
            text: str,
            chunk_size: int,
            overlap: int
    ) -> List[Dict]:
        """고정 크기 청킹"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "content": chunk_text,
                "start": start,
                "end": end,
                "index": len(chunks)
            })

            start = end - overlap

        return chunks

    def _semantic_chunking(
            self,
            text: str,
            params: Dict[str, Any]
    ) -> List[Dict]:
        """의미 기반 청킹 (간단 구현)"""
        # 실제로는 임베딩 기반 유사도 계산 필요
        sentences = text.split('. ')
        chunks = []
        current_chunk = []

        for sentence in sentences:
            current_chunk.append(sentence)

            # 간단한 크기 기반 분할
            if len(' '.join(current_chunk)) > params.get("chunk_size", 512):
                chunks.append({
                    "content": ' '.join(current_chunk),
                    "index": len(chunks)
                })
                current_chunk = []

        if current_chunk:
            chunks.append({
                "content": ' '.join(current_chunk),
                "index": len(chunks)
            })

        return chunks

    def _evaluate_chunks(self, chunks: List[Dict]) -> Dict[str, float]:
        """청크 품질 평가"""
        if not chunks:
            return {"overall": 0}

        sizes = [len(c.get("content", "")) for c in chunks]

        # 크기 일관성
        size_variance = np.var(sizes) if len(sizes) > 1 else 0
        size_consistency = 1.0 / (1.0 + size_variance / 10000)

        # 크기 적절성
        avg_size = np.mean(sizes)
        size_appropriateness = min(1.0, avg_size / 500) if avg_size < 1000 else max(0, 2000 - avg_size) / 1000

        # 전체 점수
        overall = (size_consistency + size_appropriateness) / 2

        return {
            "overall": overall,
            "size_consistency": size_consistency,
            "size_appropriateness": size_appropriateness,
            "num_chunks": len(chunks),
            "avg_size": avg_size
        }

    def _chunk_to_dict(self, chunk: Any) -> Dict[str, Any]:
        """청크를 딕셔너리로 변환"""
        if isinstance(chunk, dict):
            return chunk

        # Chunk 객체인 경우
        return {
            "content": getattr(chunk, "content", str(chunk)),
            "index": getattr(chunk, "index", 0),
            "metadata": getattr(chunk, "metadata", {})
        }


class QualityEvaluator:
    """품질 평가 노드"""

    async def evaluate(self, state: DocumentState) -> DocumentState:
        """최종 품질 평가 및 피드백 생성"""
        chunks = state.get("chunks", [])
        quality_scores = state.get("chunk_quality_scores", {})

        # 성능 점수 계산
        performance_score = quality_scores.get("overall", 0)

        # 히스토리에 기록
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "document_id": state["document"].id,
            "category": state["category"],
            "complexity_level": self._get_complexity_level(state["complexity_score"]),
            "strategy": state["selected_strategy"].value if state["selected_strategy"] else None,
            "performance_score": performance_score,
            "num_chunks": len(chunks),
            "processing_time": state["processing_time"].get("chunking", 0)
        }

        state["performance_history"].append(history_entry)

        # 피드백 생성
        state["feedback"] = {
            "success": performance_score > 0.6,
            "score": performance_score,
            "suggestions": self._generate_suggestions(state),
            "summary": f"Processed {len(chunks)} chunks with {performance_score:.2f} quality score"
        }

        return state

    def _get_complexity_level(self, score: float) -> str:
        if score > 0.7:
            return "high_complexity"
        elif score > 0.4:
            return "medium_complexity"
        else:
            return "low_complexity"

    def _generate_suggestions(self, state: DocumentState) -> List[str]:
        """개선 제안 생성"""
        suggestions = []

        quality_scores = state.get("chunk_quality_scores", {})

        if quality_scores.get("size_consistency", 1) < 0.5:
            suggestions.append("Consider using a more consistent chunking approach")

        if quality_scores.get("size_appropriateness", 1) < 0.5:
            suggestions.append("Adjust chunk size parameters for better content segmentation")

        if state.get("errors"):
            suggestions.append("Review and fix processing errors")

        return suggestions


class AdaptiveChunkingAgent:
    """메인 에이전트 클래스"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.analyzer = DocumentAnalyzer(self.llm)
        self.selector = StrategySelector()
        self.executor = ChunkingExecutor()
        self.evaluator = QualityEvaluator()

        # 성능 히스토리 (메모리 캐시)
        self.global_history = []

        # 워크플로우 구성
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(DocumentState)

        # 노드 추가
        workflow.add_node("analyze", self.analyzer.analyze)
        workflow.add_node("select_strategy", self.selector.select)
        workflow.add_node("execute_chunking", self.executor.execute)
        workflow.add_node("evaluate_quality", self.evaluator.evaluate)

        # 엣지 정의
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "select_strategy")
        workflow.add_edge("select_strategy", "execute_chunking")
        workflow.add_edge("execute_chunking", "evaluate_quality")
        workflow.add_edge("evaluate_quality", END)

        return workflow.compile()

    async def process_document(
            self,
            document: StandardizedDocument,
            questions: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """단일 문서 처리"""

        # 초기 상태 생성
        initial_state = DocumentState(
            document=document,
            raw_data=document.to_dict(),
            document_analysis={},
            category=document.category.value,
            content_length=document.content_length,
            complexity_score=0.0,
            selected_strategy=None,
            chunking_config=None,
            backup_strategies=[],
            chunks=None,
            chunk_quality_scores={},
            processing_time={},
            errors=[],
            warnings=[],
            performance_history=self.global_history.copy(),
            feedback=None
        )

        # 워크플로우 실행
        try:
            result = await self.workflow.ainvoke(initial_state)

            # 전역 히스토리 업데이트
            if result["performance_history"]:
                self.global_history.extend(result["performance_history"][-1:])

            return {
                "success": len(result.get("errors", [])) == 0,
                "document_id": document.id,
                "strategy": result["selected_strategy"].value if result["selected_strategy"] else None,
                "config": result["chunking_config"].__dict__ if result["chunking_config"] else None,
                "chunks": result.get("chunks", []),
                "quality_scores": result.get("chunk_quality_scores", {}),
                "feedback": result.get("feedback", {}),
                "warnings": result.get("warnings", []),
                "errors": result.get("errors", []),
                "processing_time": result.get("processing_time", {})
            }

        except Exception as e:
            return {
                "success": False,
                "document_id": document.id,
                "error": str(e),
                "chunks": [],
                "feedback": {"error": str(e)}
            }

    async def process_batch(
            self,
            documents: List[StandardizedDocument],
            max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """배치 처리"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(doc):
            async with semaphore:
                return await self.process_document(doc)

        tasks = [process_with_limit(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "document_id": documents[i].id,
                    "error": str(result)
                })
            else:
                processed_results.append(result)

        return processed_results

    def save_history(self, filepath: str):
        """성능 히스토리 저장"""
        with open(filepath, "w") as f:
            json.dump(self.global_history, f, indent=2)

    def load_history(self, filepath: str):
        """성능 히스토리 로드"""
        if Path(filepath).exists():
            with open(filepath, "r") as f:
                self.global_history = json.load(f)


# 사용 예시
async def main():
    from data_loader import iter_standardized_examples

    # 에이전트 생성
    agent = AdaptiveChunkingAgent()

    # 데이터 로드
    documents = list(iter_standardized_examples("squad", "train"))[:10]

    # 배치 처리
    results = await agent.process_batch(documents)

    # 결과 분석
    for result in results:
        print(f"Document: {result['document_id']}")
        print(f"Strategy: {result.get('strategy', 'N/A')}")
        print(f"Chunks: {len(result.get('chunks', []))}")
        print(f"Quality: {result.get('quality_scores', {}).get('overall', 0):.2f}")
        print(f"Success: {result['success']}")
        print("-" * 50)

    # 히스토리 저장
    agent.save_history("chunking_history.json")


if __name__ == "__main__":
    asyncio.run(main())