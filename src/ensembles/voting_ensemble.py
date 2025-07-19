"""
투표 기반 앙상블
Voting-based ensemble
"""

from typing import List
import numpy as np
from loguru import logger

from src.config import ChunkingStrategy
from src.data_structures import RAGResponse, Chunk
from .base_ensemble import BaseEnsemble


class VotingEnsemble(BaseEnsemble):
    """투표 기반 앙상블"""

    def __init__(self, voting_method: str = "weighted"):
        """
        Args:
            voting_method: "simple" (단순 다수결) 또는 "weighted" (가중 투표)
        """
        self.voting_method = voting_method

    async def combine_responses(self, responses: List[RAGResponse]) -> RAGResponse:
        """투표를 통해 최적 응답 선택"""
        if not responses:
            raise ValueError("응답 리스트가 비어있습니다.")

        if len(responses) == 1:
            return responses[0]

        # 투표 방식에 따라 처리
        if self.voting_method == "simple":
            return await self._simple_voting(responses)
        else:
            return await self._weighted_voting(responses)

    async def _simple_voting(self, responses: List[RAGResponse]) -> RAGResponse:
        """단순 다수결 투표"""
        # 응답 텍스트의 유사성 기반 그룹화
        response_groups = self._group_similar_responses(responses)

        # 가장 많은 투표를 받은 그룹 선택
        largest_group = max(response_groups, key=len)

        # 그룹 내에서 가장 높은 신뢰도를 가진 응답 선택
        best_response = max(largest_group, key=lambda r: r.confidence)

        # 메타데이터 업데이트
        best_response.metadata["ensemble_method"] = "voting_simple"
        best_response.metadata["votes"] = len(largest_group)
        best_response.metadata["total_candidates"] = len(responses)

        logger.info(f"단순 투표 결과: {len(largest_group)}/{len(responses)} 표")

        return best_response

    async def _weighted_voting(self, responses: List[RAGResponse]) -> RAGResponse:
        """가중 투표 (신뢰도 기반)"""
        # 각 응답에 대한 점수 계산
        scored_responses = []

        for response in responses:
            # 기본 점수: 신뢰도
            score = response.confidence

            # 청크 관련성 점수 추가 (있는 경우)
            chunk_relevance = np.mean([
                chunk.metadata.get("relevance_score", 0.5)
                for chunk in response.chunks_used
            ]) if response.chunks_used else 0.5

            score = 0.7 * score + 0.3 * chunk_relevance

            scored_responses.append((score, response))

        # 점수 기준 정렬
        scored_responses.sort(key=lambda x: x[0], reverse=True)

        # 상위 응답들의 유사성 체크
        top_responses = [r for _, r in scored_responses[:3]]

        # 유사한 응답들의 점수 합산
        similarity_groups = self._group_similar_responses(top_responses)
        group_scores = []

        for group in similarity_groups:
            group_score = sum(score for score, resp in scored_responses if resp in group)
            group_scores.append((group_score, group))

        # 최고 점수 그룹에서 최적 응답 선택
        best_group = max(group_scores, key=lambda x: x[0])[1]
        best_response = max(best_group, key=lambda r: r.confidence)

        # 모든 청크 통합
        all_chunks = []
        for response in responses:
            all_chunks.extend(response.chunks_used)

        # 중복 제거
        unique_chunks = self._deduplicate_chunks(all_chunks)

        # 최종 응답 구성
        final_response = RAGResponse(
            query_id=best_response.query_id,
            response=best_response.response,
            chunks_used=unique_chunks,
            confidence=max(score for score, _ in scored_responses),
            strategy=ChunkingStrategy.SEMANTIC,  # 기본값
            generation_time=np.mean([r.generation_time for r in responses]),
            model_used=best_response.model_used,
            metadata={
                "ensemble_method": "voting_weighted",
                "best_score": max(score for score, _ in scored_responses),
                "num_responses": len(responses),
                "score_distribution": [score for score, _ in scored_responses]
            }
        )

        return final_response

    def _group_similar_responses(self, responses: List[RAGResponse]) -> List[List[RAGResponse]]:
        """유사한 응답들을 그룹화"""
        groups = []

        for response in responses:
            # 기존 그룹에서 유사한 것 찾기
            found_group = False

            for group in groups:
                if self._are_responses_similar(response, group[0]):
                    group.append(response)
                    found_group = True
                    break

            # 새 그룹 생성
            if not found_group:
                groups.append([response])

        return groups

    def _are_responses_similar(self, resp1: RAGResponse, resp2: RAGResponse) -> bool:
        """두 응답의 유사성 판단"""
        # 간단한 휴리스틱: 길이와 첫 문장 비교
        len_ratio = len(resp1.response) / len(resp2.response) if len(resp2.response) > 0 else 0

        if 0.7 <= len_ratio <= 1.3:
            # 첫 50자 비교
            start1 = resp1.response[:50].lower()
            start2 = resp2.response[:50].lower()

            # 공통 단어 비율
            words1 = set(start1.split())
            words2 = set(start2.split())

            if words1 and words2:
                overlap = len(words1.intersection(words2))
                similarity = overlap / min(len(words1), len(words2))
                return similarity > 0.5

        return False

    def _deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """중복 청크 제거"""
        seen = set()
        unique_chunks = []

        for chunk in chunks:
            chunk_key = (chunk.document_id, chunk.start_idx, chunk.end_idx)
            if chunk_key not in seen:
                seen.add(chunk_key)
                unique_chunks.append(chunk)

        return unique_chunks