from datasets import load_dataset
import json, os, random
from typing import Dict, Any, Iterable, Optional, List, Union
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import re


class DocumentCategory(Enum):
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    ECONOMICS = "economics"
    LITERATURE = "literature"
    HISTORY = "history"
    LEGAL = "legal"
    MEDICAL = "medical"
    GENERAL = "general"
    NEWS = "news"


@dataclass
class StandardizedDocument:
    """통일된 문서 형식"""
    id: str
    title: str
    content: str
    category: DocumentCategory
    metadata: Dict[str, Any]
    questions: List[Dict[str, str]]  # [{"question": str, "answer": str}, ...]
    content_length: int
    language: str = "en"

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['category'] = self.category.value
        return data


def categorize_document(text: str, title: str = "", dataset_hint: str = "") -> DocumentCategory:
    """문서 내용을 기반으로 카테고리 자동 분류"""
    text_lower = (text + title).lower()

    # 데이터셋 힌트 기반 카테고리
    if "covid" in dataset_hint.lower() or "medical" in dataset_hint.lower():
        return DocumentCategory.MEDICAL
    elif "tech" in dataset_hint.lower():
        return DocumentCategory.TECHNOLOGY
    elif "news" in dataset_hint.lower():
        return DocumentCategory.NEWS

    category_keywords = {
        DocumentCategory.SCIENCE: ["research", "hypothesis", "experiment", "theory", "scientific", "study", "analysis",
                                   "observation"],
        DocumentCategory.TECHNOLOGY: ["software", "algorithm", "code", "system", "network", "ai", "computer", "digital",
                                      "programming", "api"],
        DocumentCategory.BUSINESS: ["company", "business", "market", "strategy", "management", "revenue", "profit",
                                    "corporate"],
        DocumentCategory.ECONOMICS: ["economy", "gdp", "inflation", "finance", "monetary", "fiscal", "trade",
                                     "economic"],
        DocumentCategory.LITERATURE: ["novel", "story", "character", "plot", "narrative", "poem", "fiction", "author"],
        DocumentCategory.HISTORY: ["historical", "century", "war", "ancient", "dynasty", "civilization", "era",
                                   "period"],
        DocumentCategory.LEGAL: ["law", "legal", "court", "regulation", "compliance", "statute", "jurisdiction",
                                 "attorney"],
        DocumentCategory.MEDICAL: ["patient", "treatment", "disease", "medical", "clinical", "diagnosis", "therapy",
                                   "covid", "virus", "vaccine"],
        DocumentCategory.NEWS: ["news", "report", "journalist", "press", "media", "breaking", "headline", "article"]
    }

    scores = {}
    for category, keywords in category_keywords.items():
        scores[category] = sum(2 if kw in text_lower[:1000] else 1 for kw in keywords if kw in text_lower)

    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return DocumentCategory.GENERAL


class DatasetAdapter(ABC):
    """데이터셋 어댑터 추상 클래스"""

    def __init__(self):
        self.dataset_name = ""

    @abstractmethod
    def extract_data(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """데이터셋 특정 형식을 통일된 형식으로 변환"""
        pass

    def clean_text(self, text: str) -> str:
        """텍스트 정제 유틸리티"""
        if not text:
            return ""
        # 과도한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # HTML 태그 제거 (있다면)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    def process_example(self, example: Dict[str, Any], dataset_name: str, split: str) -> StandardizedDocument:
        """예제를 StandardizedDocument로 변환"""
        self.dataset_name = dataset_name
        data = self.extract_data(example)

        # 컨텐츠 정제
        data["content"] = self.clean_text(data.get("content", ""))

        category = categorize_document(
            data.get("content", ""),
            data.get("title", ""),
            dataset_name
        )

        # ID가 없으면 생성
        if not data.get("id"):
            data["id"] = f"{dataset_name}_{split}_{hash(data.get('content', '')[:100])}"

        return StandardizedDocument(
            id=str(data.get("id", "")),
            title=data.get("title", ""),
            content=data.get("content", ""),
            category=category,
            metadata={
                "source": dataset_name,
                "split": split,
                "original_format": self.__class__.__name__.replace("Adapter", "").lower(),
                **data.get("extra_metadata", {})
            },
            questions=data.get("questions", []),
            content_length=len(data.get("content", "")),
            language=data.get("language", "en")
        )


class SQuADAdapter(DatasetAdapter):
    """SQuAD 1.1 및 2.0 데이터셋 어댑터"""

    def extract_data(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # SQuAD 데이터 구조 처리
        context = example.get("context", "")
        question = example.get("question", "")

        # 답변 처리 (SQuAD 1.1 vs 2.0)
        answers = example.get("answers", {})
        if isinstance(answers, dict):
            answer_texts = answers.get("text", [])
            answer = answer_texts[0] if answer_texts else ""
        else:
            answer = ""

        # 제목 추출 (있다면)
        title = example.get("title", "")
        if not title and context:
            # 컨텍스트의 첫 문장을 제목으로 사용
            first_sentence = context.split('.')[0][:100]
            title = first_sentence if len(first_sentence) < 100 else ""

        return {
            "id": example.get("id", ""),
            "title": title,
            "content": context,
            "questions": [{
                "question": question,
                "answer": answer
            }],
            "extra_metadata": {
                "is_impossible": example.get("is_impossible", False),  # SQuAD 2.0
                "answer_start": answers.get("answer_start", [0])[0] if isinstance(answers, dict) else 0
            }
        }


class NewsQAAdapter(DatasetAdapter):
    """NewsQA 데이터셋 어댑터"""

    def extract_data(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # NewsQA는 CNN 기사 기반
        story_text = example.get("story_text", "")
        question = example.get("question", "")

        # 답변 처리
        answer = example.get("answer", {})
        if isinstance(answer, dict):
            answer_text = answer.get("text", "")
        else:
            answer_text = str(answer) if answer else ""

        # 스토리 ID를 제목으로 사용
        story_id = example.get("story_id", "")
        title = f"Story {story_id}" if story_id else ""

        return {
            "id": example.get("story_id", ""),
            "title": title,
            "content": story_text,
            "questions": [{
                "question": question,
                "answer": answer_text
            }],
            "extra_metadata": {
                "story_id": story_id,
                "answer_char_ranges": answer.get("char_ranges", []) if isinstance(answer, dict) else []
            }
        }


class TechQAAdapter(DatasetAdapter):
    """TechQA 데이터셋 어댑터"""

    def extract_data(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # TechQA는 기술 문서 기반 QA
        document = example.get("document", {})

        # 문서 텍스트 추출
        if isinstance(document, dict):
            doc_text = document.get("text", "")
            doc_title = document.get("title", "")
        else:
            doc_text = str(document)
            doc_title = ""

        question = example.get("question", "")

        # 답변 처리
        answer = example.get("answer", {})
        if isinstance(answer, dict):
            answer_text = answer.get("text", "")
            if not answer_text:
                # 답변이 문서의 span으로 제공되는 경우
                start = answer.get("start", 0)
                end = answer.get("end", 0)
                if start and end and doc_text:
                    answer_text = doc_text[start:end]
        else:
            answer_text = str(answer) if answer else ""

        return {
            "id": example.get("id", ""),
            "title": doc_title or "Technical Document",
            "content": doc_text,
            "questions": [{
                "question": question,
                "answer": answer_text
            }],
            "extra_metadata": {
                "domain": example.get("domain", "technology"),
                "subdomain": example.get("subdomain", "")
            }
        }


class COVIDQAAdapter(DatasetAdapter):
    """COVID-QA 데이터셋 어댑터"""

    def extract_data(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # COVID-QA는 COVID-19 관련 과학 논문 기반
        context = example.get("context", "")
        question = example.get("question", "")

        # 답변 처리
        answers = example.get("answers", {})
        if isinstance(answers, dict):
            answer_texts = answers.get("text", [])
            answer = answer_texts[0] if answer_texts else ""
        else:
            answer = str(answers) if answers else ""

        # 문서 메타데이터
        document_id = example.get("document_id", "")
        title = example.get("title", "")
        if not title:
            title = f"COVID-19 Document {document_id}" if document_id else "COVID-19 Research"

        return {
            "id": example.get("id", "") or document_id,
            "title": title,
            "content": context,
            "questions": [{
                "question": question,
                "answer": answer
            }],
            "extra_metadata": {
                "document_id": document_id,
                "source_type": example.get("source", "scientific_paper"),
                "is_medical": True
            }
        }


class DatasetAdapterFactory:
    """데이터셋 어댑터 팩토리"""

    _adapters = {
        # 기존 데이터셋
        "deepmind/narrativeqa": "NarrativeQAAdapter",  # 클래스 이름을 문자열로 저장

        # 요청한 데이터셋들
        "squad": SQuADAdapter,
        "newsqa": NewsQAAdapter,
        "techqa": TechQAAdapter,
        "covid_qa": COVIDQAAdapter,

        # 추가 변형 (허깅페이스에 있는 다양한 이름들)
        "rajpurkar/squad": SQuADAdapter,
        "microsoft/newsqa": NewsQAAdapter,
        "ibm/techqa": TechQAAdapter,
        "deepset/covid_qa": COVIDQAAdapter,
    }

    @classmethod
    def get_adapter(cls, dataset_name: str) -> DatasetAdapter:
        """데이터셋 이름에 맞는 어댑터 반환"""
        # 소문자로 정규화
        normalized_name = dataset_name.lower().replace("-", "_")

        # 직접 매칭 시도
        adapter = cls._adapters.get(dataset_name)
        if not adapter:
            adapter = cls._adapters.get(normalized_name)

        # 부분 매칭 시도
        if not adapter:
            for key, value in cls._adapters.items():
                if normalized_name in key.lower() or key.lower() in normalized_name:
                    adapter = value
                    break

        if not adapter:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(cls._adapters.keys())}")

        # 클래스 또는 클래스 이름 처리
        if isinstance(adapter, str):
            # 동적 import (NarrativeQA 등 기존 클래스용)
            import sys
            module = sys.modules[__name__]
            adapter_class = getattr(module, adapter)
            return adapter_class()
        else:
            return adapter()

    @classmethod
    def register_adapter(cls, dataset_name: str, adapter_class: type):
        """새로운 어댑터 등록"""
        cls._adapters[dataset_name] = adapter_class


def validate_document(doc: StandardizedDocument) -> bool:
    """문서 유효성 검증"""
    if not doc.content or len(doc.content.strip()) < 10:
        return False
    if not doc.questions or not doc.questions[0].get("question"):
        return False
    return True


def iter_standardized_examples(
        dataset_name: str,
        split: str = "train",
        config_name: Optional[str] = None,
        streaming: bool = False
) -> Iterable[StandardizedDocument]:
    """모든 데이터셋을 통일된 형식으로 변환"""

    # 어댑터 가져오기
    adapter = DatasetAdapterFactory.get_adapter(dataset_name)

    # 데이터셋 로드
    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return

    # 각 예제를 StandardizedDocument로 변환
    for idx, example in enumerate(dataset):
        try:
            doc = adapter.process_example(example, dataset_name, split)
            if validate_document(doc):
                yield doc
            else:
                print(f"Skipping invalid document at index {idx}")
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue


def sample_and_save_standardized(
        out_path: str,
        dataset_name: str,
        split: str = "train",
        n: Optional[int] = None,
        seed: int = 42,
        config_name: Optional[str] = None,
        streaming: bool = False
) -> str:
    """통일된 형식으로 저장"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    random.seed(seed)

    iterator = iter_standardized_examples(dataset_name, split, config_name, streaming)

    saved_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        if n is not None and not streaming:
            # 전체 로드 후 샘플링
            all_examples = list(iterator)
            examples_to_write = random.sample(all_examples, min(n, len(all_examples)))
            for doc in examples_to_write:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False))
                f.write("\n")
                saved_count += 1
        else:
            # 스트리밍 또는 전체 저장
            for doc in iterator:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False))
                f.write("\n")
                saved_count += 1
                if n and saved_count >= n:
                    break

    print(f"저장 완료: {out_path} ({saved_count} samples from {dataset_name}/{split})")
    return out_path


def process_multiple_datasets(
        datasets: List[Dict[str, Any]],
        output_dir: str = "data"
) -> Dict[str, str]:
    """여러 데이터셋 일괄 처리"""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for dataset_info in datasets:
        dataset_name = dataset_info["name"]
        split = dataset_info.get("split", "train")
        n_samples = dataset_info.get("n_samples", None)
        config = dataset_info.get("config", None)

        output_filename = f"{dataset_name.replace('/', '_')}_{split}_{n_samples or 'all'}_samples.jsonl"
        output_path = os.path.join(output_dir, output_filename)

        try:
            print(f"\nProcessing {dataset_name}...")
            result_path = sample_and_save_standardized(
                out_path=output_path,
                dataset_name=dataset_name,
                split=split,
                n=n_samples,
                config_name=config
            )
            results[dataset_name] = result_path
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            results[dataset_name] = f"Error: {e}"

    return results


if __name__ == "__main__":
    # 요청한 4개 데이터셋 처리
    datasets_to_process = [
        {"name": "squad", "split": "train", "n_samples": 100},
        {"name": "newsqa", "split": "train", "n_samples": 100},
        {"name": "techqa", "split": "train", "n_samples": 100},
        {"name": "covid_qa", "split": "train", "n_samples": 100},
    ]

    # 일괄 처리
    results = process_multiple_datasets(datasets_to_process)

    # 결과 출력
    print("\n" + "=" * 50)
    print("처리 완료 요약:")
    print("=" * 50)
    for dataset_name, result in results.items():
        print(f"{dataset_name}: {result}")

    # 통합 데이터셋 생성 (선택적)
    print("\n통합 데이터셋 생성 중...")
    combined_data = []
    for dataset_info in datasets_to_process:
        dataset_name = dataset_info["name"]
        output_filename = f"{dataset_name.replace('/', '_')}_{dataset_info['split']}_{dataset_info['n_samples']}_samples.jsonl"
        filepath = os.path.join("data", output_filename)

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    combined_data.append(json.loads(line))

    # 통합 파일 저장
    combined_path = "data/combined_qa_dataset.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for doc in combined_data:
            f.write(json.dumps(doc, ensure_ascii=False))
            f.write("\n")

    print(f"통합 데이터셋 저장: {combined_path} ({len(combined_data)} samples)")