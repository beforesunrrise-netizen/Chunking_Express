# rag_evaluator.py  — Minimal, paper-grade retrieval metrics only

import math
import re
import unicodedata
from typing import List, Dict, Optional

import numpy as np
from loguru import logger

from src.config import Language
from src.data_structures import RAGResponse, EvaluationResult
from .base_evaluator import BaseEvaluator


# ---------- Text normalization / matching ----------

def _num_normalize(t: str) -> str:
    # 1,000 -> 1000, 84% -> 84 percent (간단화)
    t = re.sub(r'(?<=\d),(?=\d{3}\b)', '', t)
    t = t.replace('%', ' percent')
    return t

def _normalize_text(t: Optional[str]) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = t.lower()
    t = _num_normalize(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _contains_answer(chunk_text: str, answer: str) -> bool:
    """단순/견고 substring 매칭 (필요 시 확장 가능)"""
    ct = _normalize_text(chunk_text)
    ans = _normalize_text(answer)
    if not ct or not ans:
        return False
    ans_re = re.escape(ans)
    return re.search(ans_re, ct) is not None



class RAGEvaluator(BaseEvaluator):
    """
    Retrieval 전용 평가기 (논문용 최소 지표):
      - Hit@K (single-label일 때 통상 Recall@K로 보고)
      - Precision@K
      - MRR
      - MAP
      - nDCG
    특징:
      - K-sweep: {1, 3, self.k}
      - 기본은 strict_raw_topk=True: retrieved_chunks 없으면 에러
      - LLM 호출, 환각/문맥/활용도 지표 전부 제거
    """

    def __init__(self, language: Language, k: int = 5, strict_raw_topk: bool = True):
        self.language = language
        self.k = k
        self.strict_raw_topk = strict_raw_topk  # True면 retrieved_chunks 없으면 예외
        self.leaky_eval = False  # fallback 사용 여부 기록
        logger.info(f"RAGEvaluator (retrieval-only) initialized: k={k}, strict_raw_topk={strict_raw_topk}")

    # ---------- Public API ----------

    # RAGEvaluator 클래스 내부에 추가
    async def evaluate_single_response(
            self,
            response: RAGResponse,
            ground_truth_answer: str
    ) -> Dict[str, float]:
        """
        BaseEvaluator의 추상 메서드 구현 (retrieval-only).
        내부적으로 동기 평가 함수를 호출한다.
        """
        return self._evaluate_single_response_sync(response, ground_truth_answer)

    async def evaluate_responses(
        self,
        responses: List[RAGResponse],
        ground_truth_answers: List[str]
    ) -> EvaluationResult:
        """
        Retrieval 핵심 지표만 계산.
        반환값의 legacy 필드 호환을 위해 EvaluationResult를 사용하되,
        보조 지표들은 0.0으로 채움.
        """
        if not responses:
            return self._create_empty_result()

        Ks = sorted(set([1, 3, self.k]))
        aggregate: Dict[int, Dict[str, List[float]]] = {
            K: {m: [] for m in ("hit_at_k", "precision_at_k", "recall_at_k", "mrr", "map", "ndcg")} for K in Ks
        }

        for K in Ks:
            old_k = self.k
            self.k = K

            # 동기 평가 (LLM 없음) — I/O 병목이 없어 asyncio 불필요
            for i, resp in enumerate(responses):
                if i >= len(ground_truth_answers):
                    break
                r = self._evaluate_single_response_sync(resp, ground_truth_answers[i])
                for key in aggregate[K].keys():
                    aggregate[K][key].append(r[key])

            self.k = old_k

        def mean(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        summary_at_k = {
            K: {key: mean(aggregate[K][key]) for key in aggregate[K].keys()} for K in Ks
        }

        # legacy 호환: recall_at_k는 Hit@K로 매핑 (single-label 리콜과 동일)
        final_legacy_recall_at_k = summary_at_k[self.k]["hit_at_k"]
        final_mrr = summary_at_k[self.k]["mrr"]
        strategy = responses[0].strategy.value if responses and responses[0].strategy else "unknown"

        return EvaluationResult(
            strategy=strategy,
            language=self.language,
            # 아래 3개는 더 이상 계산하지 않으므로 0.0으로 채움 (dataclass 호환)
            hallucination_auroc=0.0,
            context_relevance_rmse=0.0,
            utilization_rmse=0.0,
            recall_at_k=final_legacy_recall_at_k,  # legacy -> Hit@K
            mrr=final_mrr,
            num_samples=len(responses),
            metadata={
                "at_k": summary_at_k,       # {1:{...},3:{...},K:{...}}
                "strict_raw_topk": self.strict_raw_topk,
                "leaky_eval": self.leaky_eval,
            }
        )

    # ---------- Core per-sample evaluation ----------

    def _evaluate_single_response_sync(self, response: RAGResponse, ground_truth_answer: str) -> Dict[str, float]:
        K = self.k
        ranked_texts_k = self._get_ranked_chunks(response, K)  # 관련 문서 추출
        rel_k = self._binary_relevance_vector(ranked_texts_k, ground_truth_answer) # 정답이랑 비교

        # 단일 정답 스팬 가정(오픈QA 관행) → Recall@K = Hit@K
        total_relevant = 1

        hit_at_k      = self._hit_at_k(rel_k)
        precision_at_k= self._precision_at_k(rel_k)
        recall_at_k   = self._recall_at_k_ratio(rel_k, total_relevant)  # [0,1] 클램프 내장
        mrr_at_k      = self._rr(rel_k)
        map_at_k      = self._average_precision(rel_k)
        ndcg_at_k     = self._ndcg(rel_k)

        return {
            "hit_at_k": hit_at_k,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "mrr": mrr_at_k,
            "map": map_at_k,
            "ndcg": ndcg_at_k,
        }

    def _get_ranked_chunks(self, response: RAGResponse, k: int) -> List[str]:
        """
        평가용 Top-K 랭크 리스트(텍스트 배열).
        """
        chunks = getattr(response, "retrieved_chunks", None)

        if not chunks:
            if self.strict_raw_topk:
                raise ValueError("retrieved_chunks missing: raw retrieval Top-K required for evaluation")

        # 랭크 순으로 정렬
        if chunks and hasattr(chunks[0], "rank"):
            chunks = sorted(chunks, key=lambda c: getattr(c, "rank", 10 ** 9))

        # 텍스트 추출 (더 견고하게)
        texts = []
        for c in chunks[:k]:
            text = ""
            if hasattr(c, "content"):
                text = c.content
            elif hasattr(c, "page_content"):
                text = c.page_content
            elif hasattr(c, "text"):
                text = c.text
            elif isinstance(c, str):
                text = c
            elif isinstance(c, dict):
                text = c.get("content", c.get("page_content", c.get("text", "")))
            texts.append(text if text else "")

        return texts

    def _binary_relevance_vector(self, ranked_texts: List[str], answer: str) -> List[int]:
        return [1 if _contains_answer(t, answer) else 0 for t in ranked_texts]

    def _hit_at_k(self, rel: List[int]) -> float:
        return 1.0 if any(rel) else 0.0

    def _precision_at_k(self, rel: List[int]) -> float:
        return sum(rel) / max(1, len(rel))

    def _recall_at_k_ratio(self, rel: List[int], total_relevant: int) -> float:
        """비율형 Recall@K (single-label이면 Hit@K와 동일). [0,1] 클램프."""
        denom = max(1, total_relevant)
        return min(1.0, max(0.0, sum(rel) / denom))

    def _rr(self, rel: List[int]) -> float:
        for idx, r in enumerate(rel, start=1):
            if r == 1:
                return 1.0 / idx
        return 0.0

    def _average_precision(self, rel: List[int]) -> float:
        num_rel = 0
        ap_sum = 0.0
        for i, r in enumerate(rel, start=1):
            if r == 1:
                num_rel += 1
                ap_sum += num_rel / i
        return ap_sum / max(1, num_rel)

    def _ndcg(self, rel: List[int]) -> float:
        dcg = 0.0
        for i, r in enumerate(rel, start=1):
            if r:
                dcg += 1.0 / math.log2(i + 1)
        R = sum(rel)
        if R == 0:
            return 0.0
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, R + 1))
        return dcg / idcg

    # ---------- Empty result ----------

    def _create_empty_result(self) -> EvaluationResult:
        return EvaluationResult(
            strategy="unknown",
            language=self.language,
            hallucination_auroc=0.0,
            context_relevance_rmse=0.0,
            utilization_rmse=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            num_samples=0,
            metadata={"at_k": {}, "strict_raw_topk": self.strict_raw_topk, "leaky_eval": self.leaky_eval}
        )
