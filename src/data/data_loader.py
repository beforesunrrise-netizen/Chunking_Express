import os
import pandas as pd
from datasets import load_dataset


class SquadDataProcessor:
    """
    SQuAD 데이터셋을 처리하고 저장하는 클래스.
    """

    def __init__(self, dataset_name: str = "squad", split: str = "train"):
        """
        초기화 시 데이터셋 이름과 스플릿을 설정합니다.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.ds = None

    def _load_data(self):
        """데이터셋을 로드합니다."""
        print("🔄 Loading dataset...")
        if self.ds is None:
            self.ds = load_dataset(self.dataset_name, split=self.split)

    def _find_column(self, columns, keywords, fallback=None):
        """키워드에 맞는 컬럼 이름을 찾아 반환합니다."""
        for key in keywords:
            matches = [col for col in columns if key in col.lower()]
            if matches:
                return matches[0]
        if fallback:
            return fallback
        raise ValueError(f"No column found matching keywords: {keywords}")

    def _extract_answer(self, ans):
        """답변 구조에서 텍스트만 추출합니다."""
        if isinstance(ans, dict) and "text" in ans:
            return ans["text"][0] if isinstance(ans["text"], list) and ans["text"] else ""
        return ""

    def process_and_save(self, n_samples: int, output_dir: str = "data", random_seed: int = 42):
        """
        데이터셋을 로드하고, 랜덤 샘플링, 전처리 후 JSON으로 저장합니다.

        Args:
            n_samples (int): 랜덤으로 추출할 샘플의 개수.
            output_dir (str): 결과 파일을 저장할 디렉토리.
            random_seed (int): 재현성을 위한 랜덤 시드.
        """
        # 1. 데이터 로드
        self._load_data()

        # 2. 랜덤 샘플링
        print(f"🔀 Shuffling and selecting {n_samples} random samples...")
        if n_samples > len(self.ds):
            print(
                f"⚠️ Warning: n_samples ({n_samples}) is larger than the dataset size ({len(self.ds)}). Using all samples.")
            n_samples = len(self.ds)

        random_ds = self.ds.shuffle(seed=random_seed).select(range(n_samples))

        # 3. DataFrame 변환 및 전처리
        print("📄 Converting to DataFrame and processing...")
        df = random_ds.to_pandas()

        question_col = self._find_column(df.columns, ["question", "query"])
        context_col = self._find_column(df.columns, ["context", "paragraph"])
        answer_col = self._find_column(df.columns, ["answers", "answer"])

        df_filtered = df[[question_col, context_col, answer_col]].dropna()
        df_filtered.columns = ["question", "context", "answer"]
        df_filtered["answer"] = df_filtered["answer"].apply(self._extract_answer)

        # 4. 파일 저장
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"squad_{self.split}_{n_samples}_random.json")

        print(f"💾 Saving to '{output_path}'...")
        df_filtered.to_json(output_path, orient="records", force_ascii=False, indent=2)
        print(f"🎉 Successfully saved {len(df_filtered)} samples to '{output_path}'")
        return output_path


# --- 클래스 실행 예시 ---
if __name__ == "__main__":
    # 프로세서 객체 생성
    processor = SquadDataProcessor()
    # 100개 샘플을 처리하고 저장
    processor.process_and_save(n_samples=5)
    # 200개 샘플을 다른 시드로 처리하고 저장
    # processor.process_and_save(n_samples=200, random_seed=123)