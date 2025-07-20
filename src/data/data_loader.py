import os
import pandas as pd
from datasets import load_dataset
from typing import Dict, Any


class SquadDataProcessor:
    """
    SQuAD 형식의 데이터셋을 RAG 실험을 위한 최종 JSON 형식으로 변환합니다.
    출력 형식: {"id":..., "title":..., "question":..., "context":..., "answer":...}
    """

    def __init__(self, dataset_name: str = "squad", split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        self.ds = None

    def _load_data(self):
        if self.ds is None:
            print(f"데이터셋 로드 중: {self.dataset_name}, 스플릿: {self.split}")
            self.ds = load_dataset(self.dataset_name, split=self.split)
            print("데이터셋 로드 완료.")

    def process_and_save(self, num_samples: int, output_dir: str = "data", random_seed: int = 42):
        self._load_data()

        if num_samples > len(self.ds):
            print(f"경고: 요청된 샘플 수({num_samples})가 데이터셋 크기({len(self.ds)})보다 큽니다. 전체 데이터를 사용합니다.")
            num_samples = len(self.ds)

        print(f"{num_samples}개의 샘플을 무작위로 추출합니다...")
        sampled_dataset = self.ds.shuffle(seed=random_seed).select(range(num_samples))

        df = sampled_dataset.to_pandas()

        # 'answers' 딕셔너리에서 'text' 리스트의 첫 번째 요소를 추출
        df['answer'] = df['answers'].apply(
            lambda ans_dict: ans_dict['text'][0] if ans_dict['text'] else ""
        )

        # 답변이 없는 샘플은 제외
        df = df[df['answer'] != ''].copy()

        # --- ★★★ 최종 수정 로직 ★★★ ---
        # id와 title을 포함한 모든 필수 컬럼을 선택합니다.
        final_df = df[['id', 'title', 'question', 'context', 'answer']]
        # --- ★★★ 수정 완료 ★★★ ---


        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"rag_squad_{self.split}_{len(final_df)}_samples.json"
        output_path = os.path.join(output_dir, file_name)

        print(f"처리된 데이터를 '{output_path}'에 저장합니다...")
        final_df.to_json(output_path, orient="records", force_ascii=False, indent=2)

        print("-" * 50)
        print("✅ 데이터 처리 완료")
        print(f"최종 생성 파일: {output_path}")
        print(f"총 {len(final_df)}개의 유효한 샘플이 저장되었습니다.")
        print("\n📋 최종 데이터 구조:")
        print("   - id: 고유 식별자")
        print("   - title: 문서 제목")
        print("   - question: 사용자 질문")
        print("   - context: 정답을 포함한 원본 문서")
        print("   - answer: 질문에 대한 정답")
        print("\n이제 이 파일을 사용하여 RAG 테스트를 진행할 수 있습니다.")
        print("-" * 50)

        return output_path


# --- 스크립트 실행 예시 ---
if __name__ == "__main__":
    processor = SquadDataProcessor(dataset_name="squad", split="train")
    processor.process_and_save(num_samples=100)