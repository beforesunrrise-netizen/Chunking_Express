"""
융합 기반 앙상블
Fusion-based ensemble
"""

from typing import List, Dict
import openai
from loguru import logger
from src.config import APIConfig
from src.config import Language, ChunkingStrategy, config
from src.data_structures import RAGResponse, Chunk
from .base_ensemble import BaseEnsemble


class FusionEnsemble(BaseEnsemble):
    """융합 기반 앙상블 - GPT-4를 사용하여 응답들을 통합"""

    def __init__(self, language: Language):
        self.language = language
        config_data = APIConfig()
        self.client = openai.AsyncOpenAI(api_key=config_data.openai_api_key)
        self.model = config.model.gpt_model

    async def combine_responses(self, responses: List[RAGResponse]) -> RAGResponse:
        """GPT-4를 사용하여 응답들을 융합"""
        if not responses:
            raise ValueError("응답 리스트가 비어있습니다.")

        if len(responses) == 1:
            return responses[0]

        # 융합을 위한 프롬프트 생성
        fusion_prompt = self._create_fusion_prompt(responses)

        try:
            # GPT-4 호출
            result = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_fusion_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": fusion_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )

            fused_response_text = result.choices[0].message.content

            # 모든 청크 수집 및 중복 제거
            all_chunks = self._collect_unique_chunks(responses)

            # 신뢰도 계산
            avg_confidence = sum(r.confidence for r in responses) / len(responses)
            fusion_confidence = min(0.95, avg_confidence * 1.1)  # 융합 보너스

            # 융합된 응답 생성
            fused_response = RAGResponse(
                query_id=responses[0].query_id,
                response=fused_response_text,
                chunks_used=all_chunks,
                confidence=fusion_confidence,
                strategy=ChunkingStrategy.SEMANTIC,  # 기본값
                generation_time=sum(r.generation_time for r in responses) / len(responses),
                model_used=self.model,
                metadata={
                    "ensemble_method": "fusion",
                    "num_responses_fused": len(responses),
                    "source_strategies": [r.strategy.value for r in responses if r.strategy],
                    "fusion_model": self.model,
                    "avg_source_confidence": avg_confidence
                }
            )

            logger.info(f"융합 완료: {len(responses)}개 응답 통합, 신뢰도: {fusion_confidence:.3f}")

            return fused_response

        except Exception as e:
            logger.error(f"응답 융합 실패: {e}")
            # 실패 시 가장 높은 신뢰도 응답 반환
            return max(responses, key=lambda r: r.confidence)

    def _get_fusion_system_prompt(self) -> str:
        """융합 시스템 프롬프트"""
        if self.language == Language.KOREAN:
            return """당신은 여러 AI 응답을 하나의 포괄적이고 정확한 답변으로 통합하는 전문가입니다.

다음 지침을 따르세요:
1. 모든 응답의 핵심 정보를 포함시키되, 중복은 제거하세요.
2. 상충하는 정보가 있다면, 가장 논리적이고 일관된 정보를 선택하세요.
3. 응답은 명확하고 구조화된 형태로 작성하세요.
4. 원본 응답들보다 더 포괄적이고 유용한 답변을 만드세요.
5. 추가적인 해석이나 추론은 하지 마세요."""
        else:
            return """You are an expert at integrating multiple AI responses into one comprehensive and accurate answer.

Follow these guidelines:
1. Include key information from all responses while removing duplicates.
2. If there's conflicting information, choose the most logical and consistent one.
3. Write the response in a clear and structured manner.
4. Create a more comprehensive and useful answer than the original responses.
5. Do not add additional interpretation or inference."""

    def _create_fusion_prompt(self, responses: List[RAGResponse]) -> str:
        """융합 프롬프트 생성"""
        # 응답들을 전략별로 그룹화
        strategy_responses = {}
        for response in responses:
            strategy = response.strategy.value if response.strategy else "unknown"
            if strategy not in strategy_responses:
                strategy_responses[strategy] = []
            strategy_responses[strategy].append(response.response)

        # 프롬프트 구성
        response_texts = []
        for strategy, texts in strategy_responses.items():
            for i, text in enumerate(texts):
                response_texts.append(f"[{strategy} 전략 응답 {i + 1}]\n{text}")

        combined_responses = "\n\n".join(response_texts)

        if self.language == Language.KOREAN:
            return f"""다음은 동일한 질문에 대한 여러 개의 응답입니다. 
이들을 하나의 최적화된 답변으로 통합하세요.

{combined_responses}

위 응답들의 장점을 결합하여 가장 정확하고 완전한 답변을 생성하세요."""
        else:
            return f"""The following are multiple responses to the same question. 
Integrate them into one optimized answer.

{combined_responses}

Combine the strengths of the above responses to generate the most accurate and complete answer."""

    def _collect_unique_chunks(self, responses: List[RAGResponse]) -> List[Chunk]:
        """모든 응답에서 고유한 청크 수집"""
        chunk_map: Dict[str, Chunk] = {}

        for response in responses:
            for chunk in response.chunks_used:
                # 청크 ID를 키로 사용
                if chunk.id not in chunk_map:
                    chunk_map[chunk.id] = chunk
                else:
                    # 같은 청크가 여러 응답에서 사용된 경우,
                    # 더 높은 관련성 점수를 가진 것 선택
                    existing_score = chunk_map[chunk.id].metadata.get("relevance_score", 0)
                    new_score = chunk.metadata.get("relevance_score", 0)

                    if new_score > existing_score:
                        chunk_map[chunk.id] = chunk

        # 관련성 점수 기준 정렬
        unique_chunks = list(chunk_map.values())
        unique_chunks.sort(
            key=lambda c: c.metadata.get("relevance_score", 0),
            reverse=True
        )

        return unique_chunks