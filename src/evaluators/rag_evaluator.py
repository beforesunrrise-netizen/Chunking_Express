import json
from typing import List, Dict, Any, Tuple
import numpy as np
import openai
from loguru import logger
from sklearn.metrics import roc_auc_score
from rouge_score import rouge_scorer
import bert_score

from src.config import Language, config
from src.data_structures import RAGResponse, EvaluationResult
from .base_evaluator import BaseEvaluator
from src.config import APIConfig

class RAGEvaluator(BaseEvaluator):
    """RAG 시스템 평가기"""

    def __init__(self, language: Language):
        self.language = language
        config_data = APIConfig()
        self.client = openai.AsyncOpenAI(api_key=config_data.openai_api_key)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    async def evaluate_responses(
        self,
        responses: List[RAGResponse],
        ground_truths: List[str]
    ) -> EvaluationResult:
        """응답들을 평가"""
        if not responses:
            return self._create_empty_result()

        hallucination_scores = []
        context_relevance_scores = []
        utilization_scores = []
        retrieval_recall_scores = []

        # 각 응답 평가
        for i, response in enumerate(responses):
            if i < len(ground_truths):
                scores = await self.evaluate_single_response(response, ground_truths[i])

                hallucination_scores.append(scores["hallucination"])
                context_relevance_scores.append(scores["context_relevance"])
                utilization_scores.append(scores["utilization"])
                retrieval_recall_scores.append(scores["retrieval_recall"])

        # AUROC 및 RMSE 계산
        hallucination_auroc = self._calculate_auroc(hallucination_scores)
        context_relevance_rmse = self._calculate_rmse(context_relevance_scores)
        utilization_rmse = self._calculate_rmse(utilization_scores)

        # 전략 추출
        strategy = responses[0].strategy.value if responses and responses[0].strategy else "unknown"

        return EvaluationResult(
            strategy=strategy,
            language=self.language,
            hallucination_auroc=hallucination_auroc,
            context_relevance_rmse=context_relevance_rmse,
            utilization_rmse=utilization_rmse,
            num_samples=len(responses),
            metadata={
                "hallucination_scores": hallucination_scores,
                "context_relevance_scores": context_relevance_scores,
                "utilization_scores": utilization_scores,
                "retrieval_recall_scores": retrieval_recall_scores
            }
        )

    async def evaluate_single_response(
        self,
        response: RAGResponse,
        ground_truth: str
    ) -> Dict[str, float]:
        """단일 응답을 모든 지표에 대해 평가"""
        # 1. Hallucination 평가
        hallucination_score = await self._evaluate_hallucination(response, ground_truth)

        # 2. Context Relevance 평가
        context_relevance_score = await self._evaluate_context_relevance(response)

        # 3. Utilization 평가
        utilization_score = self._evaluate_utilization(response)

        # 4. Retrieval Recall 평가
        retrieval_recall_score = self._evaluate_retrieval_recall(response, ground_truth)

        return {
            "hallucination": hallucination_score,
            "context_relevance": context_relevance_score,
            "utilization": utilization_score,
            "retrieval_recall": retrieval_recall_score
        }

    async def _evaluate_hallucination(self, response: RAGResponse, ground_truth: str) -> float:
        """환각 현상 평가 (0: 환각 없음, 1: 환각 있음)"""
        # GPT-4 기반 평가
        gpt_score = await self._gpt_hallucination_check(response.response, ground_truth)

        # ROUGE 점수 기반 평가
        rouge_scores = self.rouge_scorer.score(ground_truth, response.response)
        rouge_l_score = rouge_scores['rougeL'].fmeasure

        # 종합 점수 (환각이 있으면 1에 가까움)
        hallucination_score = 1 - (0.7 * gpt_score + 0.3 * rouge_l_score)

        return max(0, min(1, hallucination_score))

    async def _evaluate_context_relevance(self, response: RAGResponse) -> float:
        """문맥 관련성 평가 (0: 완벽, 1: 관련 없음)"""
        if not response.chunks_used:
            return 1.0  # 청크가 없으면 최악의 점수

        relevance_scores = []
        for chunk in response.chunks_used:
            score = await self._gpt_relevance_check(response.response, chunk.content)
            relevance_scores.append(score)

        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
        return 1 - avg_relevance # RMSE를 위해 1에서 빼기 (0에 가까울수록 좋음)

    def _evaluate_utilization(self, response: RAGResponse) -> float:
        """청크 활용도 평가 (0: 완벽한 활용, 1: 활용 안 함)"""
        if not response.chunks_used:
            return 1.0

        response_length = len(response.response.split())
        num_chunks = len(response.chunks_used)
        ideal_ratio = response_length / (num_chunks * 75) # 이상적인 비율 (청크당 75 단어 가정)

        if ideal_ratio < 0.5:
            utilization_score = 1 - ideal_ratio * 2
        elif ideal_ratio > 2:
            utilization_score = min(1, ideal_ratio / 4)
        else:
            utilization_score = 0

        return utilization_score

    def _evaluate_retrieval_recall(self, response: RAGResponse, ground_truth: str) -> float:
        """Retrieval Recall을 평가합니다. (디버깅 로그 추가)"""
        logger.debug(f"=== Retrieval Recall 평가 시작 ===")
        logger.debug(f"Ground truth: {ground_truth[:100]}...")
        logger.debug(f"Chunks used: {len(response.chunks_used) if response.chunks_used else 0}")

        if not response.chunks_used or not ground_truth:
            logger.debug("청크가 없거나 ground_truth가 없어서 0.0 반환")
            return 0.0

        ground_truth_lower = ground_truth.lower().strip()
        relevant_chunks = []

        for i, chunk in enumerate(response.chunks_used):
            chunk_content_lower = chunk.content.lower().strip()
            logger.debug(f"\n청크 {i + 1} 평가:")
            logger.debug(f"청크 내용 (처음 100자): {chunk_content_lower[:100]}...")

            if ground_truth_lower in chunk_content_lower:
                logger.debug(f"  -> Exact match 발견!")
                relevant_chunks.append(chunk)
                continue

            if self.language == Language.KOREAN:
                stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '와', '과', '이다', '있다', '하다', '되다'}
            else:
                stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}

            ground_words = [w for w in ground_truth_lower.split() if w not in stopwords and len(w) > 1]
            logger.debug(f"  Ground truth 중요 단어: {ground_words[:10]}")

            if not ground_words:
                continue

            words_found = sum(1 for word in ground_words if word in chunk_content_lower)
            word_coverage = words_found / len(ground_words)
            logger.debug(f"  단어 coverage: {word_coverage:.2f} ({words_found}/{len(ground_words)})")

            if word_coverage >= 0.6:
                logger.debug(f"  -> Word coverage로 relevant 판정!")
                relevant_chunks.append(chunk)
                continue

            if hasattr(self, 'rouge_scorer'):
                rouge_scores = self.rouge_scorer.score(ground_truth_lower, chunk_content_lower)
                rouge_l_score = rouge_scores['rougeL'].fmeasure
                logger.debug(f"  ROUGE-L 점수: {rouge_l_score:.3f}")

                if rouge_l_score >= 0.5:
                    logger.debug(f"  -> ROUGE-L로 relevant 판정!")
                    relevant_chunks.append(chunk)

        recall = len(relevant_chunks) / len(response.chunks_used)
        logger.debug(f"\n최종 결과: {len(relevant_chunks)}/{len(response.chunks_used)} = {recall:.3f}")

        if len(relevant_chunks) > 0 and recall < 0.2:
            recall = 0.2
            logger.debug(f"보정 적용: {recall}")

        return min(1.0, recall)

    async def _gpt_hallucination_check(self, response: str, ground_truth: str) -> float:
        """GPT-4를 사용한 환각 체크"""
        prompt = self._create_hallucination_prompt(response, ground_truth)
        try:
            result = await self.client.chat.completions.create(
                model=config.model.gpt_model,
                messages=[
                    {"role": "system", "content": self._get_hallucination_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            evaluation = json.loads(result.choices[0].message.content)
            return evaluation.get("accuracy_score", 0.5)
        except Exception as e:
            logger.error(f"환각 평가 실패: {e}")
            return 0.5

    async def _gpt_relevance_check(self, response: str, context: str) -> float:
        """GPT-4를 사용한 관련성 체크"""
        prompt = self._create_relevance_prompt(response, context)
        try:
            result = await self.client.chat.completions.create(
                model=config.model.gpt_model,
                messages=[
                    {"role": "system", "content": self._get_relevance_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            evaluation = json.loads(result.choices[0].message.content)
            return evaluation.get("relevance_score", 0.5)
        except Exception as e:
            logger.error(f"관련성 평가 실패: {e}")
            return 0.5

    def _calculate_auroc(self, scores: List[float]) -> float:
        """AUROC 계산"""
        if len(scores) < 2: return 0.5
        threshold = 0.5
        y_true = [1 if score > threshold else 0 for score in scores]
        if len(set(y_true)) == 1: return 0.5
        try:
            return roc_auc_score(y_true, scores)
        except Exception as e:
            logger.error(f"AUROC 계산 실패: {e}")
            return 0.5

    def _calculate_rmse(self, scores: List[float]) -> float:
        """RMSE 계산"""
        if not scores: return 0.0
        ideal_scores = [0.0] * len(scores)
        mse = np.mean([(actual - ideal) ** 2 for actual, ideal in zip(scores, ideal_scores)])
        return np.sqrt(mse)

    def _get_hallucination_system_prompt(self) -> str:
        """환각 평가 시스템 프롬프트"""
        prompts = {
            Language.KOREAN: """당신은 AI 응답의 정확성을 평가하는 전문가입니다.
제공된 정답과 생성된 응답을 비교하여 정확성을 평가하세요.
환각(hallucination)이나 잘못된 정보가 있는지 확인하세요.""",
            Language.ENGLISH: """You are an expert at evaluating AI response accuracy.
Compare the provided ground truth with the generated response to evaluate accuracy.
Check for hallucinations or incorrect information."""
        }
        return prompts.get(self.language, prompts[Language.ENGLISH])

    def _get_relevance_system_prompt(self) -> str:
        """관련성 평가 시스템 프롬프트"""
        prompts = {
            Language.KOREAN: """당신은 텍스트 간의 관련성을 평가하는 전문가입니다.
응답이 제공된 문맥과 얼마나 관련이 있는지 평가하세요.""",
            Language.ENGLISH: """You are an expert at evaluating text relevance.
Evaluate how relevant the response is to the provided context."""
        }
        return prompts.get(self.language, prompts[Language.ENGLISH])

    def _create_hallucination_prompt(self, response: str, ground_truth: str) -> str:
        """환각 평가 프롬프트"""
        if self.language == Language.KOREAN:
            return f"""정답과 생성된 응답을 비교하여 정확성을 평가하세요.

    정답: {ground_truth}
    생성된 응답: {response}

    응답 형식 (JSON):
    {{
        "accuracy_score": 0.0-1.0,
        "has_hallucination": true/false,
        "incorrect_parts": ["잘못된 부분 1", "잘못된 부분 2"],
        "reasoning": "평가 이유"
    }}"""
        else:
            return f"""Compare the ground truth with the generated response to evaluate accuracy.

    Ground Truth: {ground_truth}
    Generated Response: {response}

    Response format (JSON):
    {{
        "accuracy_score": 0.0-1.0,
        "has_hallucination": true/false,
        "incorrect_parts": ["incorrect part 1", "incorrect part 2"],
        "reasoning": "evaluation reasoning"
    }}"""

    def _create_relevance_prompt(self, response: str, context: str) -> str:
        """관련성 평가 프롬프트"""
        if self.language == Language.KOREAN:
            return f"""응답이 제공된 문맥과 얼마나 관련이 있는지 평가하세요.

    문맥: {context}
    응답: {response}

    응답 형식 (JSON):
    {{
        "relevance_score": 0.0-1.0,
        "relevant_parts": ["관련 부분 1", "관련 부분 2"],
        "irrelevant_parts": ["무관한 부분 1"],
        "reasoning": "평가 이유"
    }}"""
        else:
            return f"""Evaluate how relevant the response is to the provided context.

    Context: {context}
    Response: {response}

    Response format (JSON):
    {{
        "relevance_score": 0.0-1.0,
        "relevant_parts": ["relevant part 1", "relevant part 2"],
        "irrelevant_parts": ["irrelevant part 1"],
        "reasoning": "evaluation reasoning"
    }}"""

    def _create_empty_result(self) -> EvaluationResult:
        """빈 평가 결과 생성"""
        return EvaluationResult(
            strategy="unknown",
            language=self.language,
            hallucination_auroc=0.5,
            context_relevance_rmse=1.0,
            utilization_rmse=1.0,
            num_samples=0
        )