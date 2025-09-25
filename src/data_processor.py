"""
데이터 처리 모듈
Data processing module
"""

import json
from typing import List, Tuple
import openai
from loguru import logger

from src.config import Language, config
from src.data_structures import Document, Query


class DataProcessor:
    """데이터 처리 클래스"""

    def __init__(self):
        self.logger = logger
        self.client = openai.AsyncOpenAI()

    async def load_data(self, language: Language) -> Tuple[List[Document], List[Query]]:
        """데이터 로드"""
        data_path = config.get_data_path(language)

        if not data_path.exists():
            self.logger.warning(f"데이터 파일이 없습니다: {data_path}")
            return self._create_sample_data(language)

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []
            queries = []

            # 데이터 구조에 따라 파싱
            if isinstance(data, dict) and "data" in data:
                data_items = data["data"]
            else:
                data_items = data

            for item in data_items[:config.experiment.sample_size]:
                doc = Document(
                    id=item.get("id", f"doc_{len(documents)}"),
                    content=item.get("context", item.get("text", "")),
                    language=language,
                    title=item.get("title", ""),
                    source=item.get("source", ""),
                    metadata=item.get("metadata", {})
                )
                documents.append(doc)

                query = Query(
                    id=item.get("id", f"query_{len(queries)}"),
                    question=item.get("question", ""),
                    language=language,
                    expected_answer=item.get("answer", item.get("answers", {}).get("text", [""])[0] if isinstance(
                        item.get("answers"), dict) else ""),
                    context_id=doc.id,
                    metadata=item.get("query_metadata", {})
                )
                queries.append(query)

            self.logger.info(f"{language.value} 데이터 로드 완료: {len(documents)}개 문서, {len(queries)}개 쿼리")
            return documents, queries

        except Exception as e:
            self.logger.error(f"데이터 로드 실패: {e}")
            return self._create_sample_data(language)

    async def translate_data(
            self,
            documents: List[Document],
            queries: List[Query],
            target_language: Language
    ) -> Tuple[List[Document], List[Query]]:
        """데이터 번역"""
        self.logger.info(f"{target_language.value}로 번역 시작")

        translated_docs = []
        translated_queries = []

        for doc in documents:
            translated_content = await self._translate_text(doc.content, target_language)
            translated_doc = Document(
                id=f"{doc.id}_{target_language.value}",
                content=translated_content,
                language=target_language,
                title=await self._translate_text(doc.title, target_language) if doc.title else "",
                source=doc.source,
                metadata={**doc.metadata, "original_id": doc.id}
            )
            translated_docs.append(translated_doc)

        for query in queries:
            translated_question = await self._translate_text(query.question, target_language)
            translated_answer = await self._translate_text(query.expected_answer,
                                                           target_language) if query.expected_answer else ""

            translated_query = Query(
                id=f"{query.id}_{target_language.value}",
                question=translated_question,
                language=target_language,
                expected_answer=translated_answer,
                context_id=f"{query.context_id}_{target_language.value}" if query.context_id else None,
                metadata={**query.metadata, "original_id": query.id}
            )
            translated_queries.append(translated_query)

        return translated_docs, translated_queries

    async def _translate_text(self, text: str, target_language: Language) -> str:
        """텍스트 번역"""
        if not text:
            return ""

        target_lang_name = "Korean" if target_language == Language.KOREAN else "English"

        try:
            response = await self.client.chat.completions.create(
                model=config.model.gpt_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate the following text to {target_lang_name}. Maintain the same meaning and tone. Only provide the translation without any additional explanation."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.1
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"번역 실패: {e}")
            return text

    async def load_multi_domain_data(self, language: Language) -> Tuple[List[Document], List[Query]]:
        """논문용 다중 도메인 데이터 로딩"""
        self.logger.info(f"다중 도메인 데이터 로딩 시작: {language.value}")

        all_documents = []
        all_queries = []

        domain_configs = {
            "medical": {
                "datasets": ["covid_qa_deepset", "pubmed_qa"],
                "min_samples": config.experiment.min_samples_per_domain,
                "max_samples": config.experiment.max_samples_per_domain
            },
            "scientific": {
                "datasets": ["scifact", "natural_questions"],
                "min_samples": config.experiment.min_samples_per_domain,
                "max_samples": config.experiment.max_samples_per_domain
            },
            "news": {
                "datasets": ["cnn_dailymail", "bbc_news"],
                "min_samples": config.experiment.min_samples_per_domain,
                "max_samples": config.experiment.max_samples_per_domain
            },
            "legal": {
                "datasets": ["legal_documents", "court_cases"],
                "min_samples": config.experiment.min_samples_per_domain,
                "max_samples": config.experiment.max_samples_per_domain
            },
            "technical": {
                "datasets": ["stack_overflow", "github_issues"],
                "min_samples": config.experiment.min_samples_per_domain,
                "max_samples": config.experiment.max_samples_per_domain
            },
            "conversational": {
                "datasets": ["squad", "ms_marco"],
                "min_samples": config.experiment.min_samples_per_domain,
                "max_samples": config.experiment.max_samples_per_domain
            }
        }

        # 설정된 도메인들만 로드
        target_domains = config.experiment.target_domains

        for domain in target_domains:
            if domain not in domain_configs:
                self.logger.warning(f"알 수 없는 도메인: {domain}")
                continue

            self.logger.info(f"도메인 '{domain}' 데이터 로딩 중...")
            domain_docs, domain_queries = await self._load_domain_data(
                domain, domain_configs[domain], language
            )

            # 도메인 정보를 메타데이터에 추가
            for doc in domain_docs:
                doc.metadata["domain"] = domain
                doc.metadata["domain_category"] = self._get_domain_category(domain)

            for query in domain_queries:
                query.metadata["domain"] = domain
                query.metadata["domain_category"] = self._get_domain_category(domain)

            all_documents.extend(domain_docs)
            all_queries.extend(domain_queries)

            self.logger.info(f"도메인 '{domain}' 완료: {len(domain_docs)}개 문서, {len(domain_queries)}개 쿼리")

        self.logger.info(f"전체 다중 도메인 데이터 로딩 완료: {len(all_documents)}개 문서, {len(all_queries)}개 쿼리")
        return all_documents, all_queries

    async def _load_domain_data(self, domain: str, config_dict: dict, language: Language) -> Tuple[List[Document], List[Query]]:
        """개별 도메인 데이터 로딩"""
        # 실제 구현에서는 각 도메인별로 데이터 소스를 다르게 처리
        # 현재는 도메인별 샘플 데이터 생성
        if domain == "medical":
            return self._create_medical_sample_data(language, config_dict)
        elif domain == "scientific":
            return self._create_scientific_sample_data(language, config_dict)
        elif domain == "news":
            return self._create_news_sample_data(language, config_dict)
        elif domain == "legal":
            return self._create_legal_sample_data(language, config_dict)
        elif domain == "technical":
            return self._create_technical_sample_data(language, config_dict)
        elif domain == "conversational":
            return self._create_conversational_sample_data(language, config_dict)
        else:
            return [], []

    def _get_domain_category(self, domain: str) -> str:
        """도메인 카테고리 매핑"""
        category_map = {
            "medical": "specialized",
            "scientific": "specialized",
            "news": "general",
            "legal": "specialized",
            "technical": "specialized",
            "conversational": "general"
        }
        return category_map.get(domain, "general")

    def _create_medical_sample_data(self, language: Language, config_dict: dict) -> Tuple[List[Document], List[Query]]:
        """의료 도메인 샘플 데이터"""
        min_samples = config_dict["min_samples"]

        if language == Language.KOREAN:
            samples = [
                {
                    "id": f"medical_kr_{i}",
                    "context": f"""코로나19(COVID-19)는 SARS-CoV-2 바이러스에 의해 발생하는 호흡기 감염 질환입니다.
                    2019년 12월 중국 우한에서 처음 발견되었으며, 2020년 3월 WHO에 의해 팬데믹으로 선언되었습니다.
                    주요 증상으로는 발열, 기침, 호흡곤란, 피로감, 미각 및 후각 상실 등이 있습니다.
                    대부분의 환자는 경미한 증상을 보이지만, 일부는 중증 폐렴으로 진행될 수 있습니다.
                    특히 고령자나 기저질환이 있는 사람들에게서 중증으로 발전할 위험이 높습니다. 샘플 {i}""",
                    "question": f"코로나19의 주요 증상은 무엇인가요? (샘플 {i})",
                    "answer": "발열, 기침, 호흡곤란, 피로감, 미각 및 후각 상실"
                }
                for i in range(min_samples)
            ]
        else:
            samples = [
                {
                    "id": f"medical_en_{i}",
                    "context": f"""COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus.
                    It was first identified in Wuhan, China in December 2019 and was declared a pandemic by the WHO in March 2020.
                    Common symptoms include fever, cough, shortness of breath, fatigue, and loss of taste or smell.
                    While most patients experience mild symptoms, some may develop severe pneumonia.
                    The elderly and those with underlying conditions are at higher risk of severe illness. Sample {i}""",
                    "question": f"What are the main symptoms of COVID-19? (Sample {i})",
                    "answer": "fever, cough, shortness of breath, fatigue, loss of taste or smell"
                }
                for i in range(min_samples)
            ]

        return self._convert_samples_to_docs_queries(samples, language)

    def _create_scientific_sample_data(self, language: Language, config_dict: dict) -> Tuple[List[Document], List[Query]]:
        """과학 도메인 샘플 데이터"""
        min_samples = config_dict["min_samples"]

        if language == Language.KOREAN:
            samples = [
                {
                    "id": f"scientific_kr_{i}",
                    "context": f"""머신러닝에서 과적합(overfitting)은 모델이 훈련 데이터에 과도하게 맞춰져서
                    새로운 데이터에 대한 일반화 성능이 떨어지는 현상입니다. 이는 모델의 복잡도가 너무 높거나
                    훈련 데이터가 부족할 때 주로 발생합니다. 과적합을 방지하기 위한 방법으로는 정규화(regularization),
                    드롭아웃(dropout), 교차 검증(cross-validation) 등이 있습니다.
                    또한 더 많은 훈련 데이터를 수집하거나 모델의 복잡도를 줄이는 방법도 효과적입니다. 샘플 {i}""",
                    "question": f"머신러닝에서 과적합을 방지하는 방법은? (샘플 {i})",
                    "answer": "정규화, 드롭아웃, 교차 검증, 더 많은 데이터 수집, 모델 복잡도 감소"
                }
                for i in range(min_samples)
            ]
        else:
            samples = [
                {
                    "id": f"scientific_en_{i}",
                    "context": f"""Overfitting in machine learning occurs when a model learns the training data too well,
                    resulting in poor generalization to new, unseen data. This typically happens when the model is too complex
                    or when there is insufficient training data. Methods to prevent overfitting include regularization techniques,
                    dropout, cross-validation, and early stopping. Additionally, collecting more training data or
                    reducing model complexity can be effective strategies. Sample {i}""",
                    "question": f"What are methods to prevent overfitting in machine learning? (Sample {i})",
                    "answer": "regularization, dropout, cross-validation, more training data, reducing model complexity"
                }
                for i in range(min_samples)
            ]

        return self._convert_samples_to_docs_queries(samples, language)

    def _create_news_sample_data(self, language: Language, config_dict: dict) -> Tuple[List[Document], List[Query]]:
        """뉴스 도메인 샘플 데이터"""
        min_samples = config_dict["min_samples"]

        if language == Language.KOREAN:
            samples = [
                {
                    "id": f"news_kr_{i}",
                    "context": f"""한국 정부가 AI 기술 발전을 위한 새로운 정책을 발표했습니다.
                    이번 정책에는 AI 연구개발 예산 확대, 인재 양성 프로그램, 그리고 AI 윤리 가이드라인이 포함되어 있습니다.
                    특히 교육 부문에서 AI 활용을 늘리고, 의료 분야에서의 AI 도입을 가속화할 계획입니다.
                    정부는 향후 5년간 AI 분야에 1조원을 투자할 예정이라고 밝혔습니다.
                    이러한 투자를 통해 한국이 글로벌 AI 강국으로 발돋움할 것으로 기대됩니다. 샘플 {i}""",
                    "question": f"한국 정부의 AI 정책에 포함된 주요 내용은? (샘플 {i})",
                    "answer": "연구개발 예산 확대, 인재 양성 프로그램, AI 윤리 가이드라인, 교육 및 의료 분야 활용"
                }
                for i in range(min_samples)
            ]
        else:
            samples = [
                {
                    "id": f"news_en_{i}",
                    "context": f"""The Korean government announced new policies to advance AI technology development.
                    The policy includes increased AI R&D budget, talent development programs, and AI ethics guidelines.
                    In particular, it plans to increase AI utilization in education and accelerate AI adoption in healthcare.
                    The government announced plans to invest 1 trillion won in the AI sector over the next five years.
                    Through this investment, Korea is expected to emerge as a global AI powerhouse. Sample {i}""",
                    "question": f"What are the main components of Korea's AI policy? (Sample {i})",
                    "answer": "increased R&D budget, talent development programs, AI ethics guidelines, education and healthcare utilization"
                }
                for i in range(min_samples)
            ]

        return self._convert_samples_to_docs_queries(samples, language)

    def _create_legal_sample_data(self, language: Language, config_dict: dict) -> Tuple[List[Document], List[Query]]:
        """법률 도메인 샘플 데이터"""
        min_samples = config_dict["min_samples"]

        samples = [
            {
                "id": f"legal_{language.value}_{i}",
                "context": f"법률 문서 샘플 {i} - 개인정보보호법에 관한 내용...",
                "question": f"개인정보보호법의 주요 내용은? (샘플 {i})",
                "answer": "개인정보 수집, 이용, 제공에 관한 규정"
            }
            for i in range(min_samples)
        ]

        return self._convert_samples_to_docs_queries(samples, language)

    def _create_technical_sample_data(self, language: Language, config_dict: dict) -> Tuple[List[Document], List[Query]]:
        """기술 도메인 샘플 데이터"""
        min_samples = config_dict["min_samples"]

        samples = [
            {
                "id": f"technical_{language.value}_{i}",
                "context": f"기술 문서 샘플 {i} - REST API 설계 가이드라인...",
                "question": f"REST API 설계 시 고려사항은? (샘플 {i})",
                "answer": "HTTP 메서드 활용, 상태코드 적절 사용, 리소스 중심 설계"
            }
            for i in range(min_samples)
        ]

        return self._convert_samples_to_docs_queries(samples, language)

    def _create_conversational_sample_data(self, language: Language, config_dict: dict) -> Tuple[List[Document], List[Query]]:
        """대화 도메인 샘플 데이터"""
        min_samples = config_dict["min_samples"]

        samples = [
            {
                "id": f"conversational_{language.value}_{i}",
                "context": f"대화형 QA 샘플 {i} - 일상 대화 내용...",
                "question": f"이 대화에서 주요 논점은? (샘플 {i})",
                "answer": "일상적인 정보 교환"
            }
            for i in range(min_samples)
        ]

        return self._convert_samples_to_docs_queries(samples, language)

    def _convert_samples_to_docs_queries(self, samples: List[dict], language: Language) -> Tuple[List[Document], List[Query]]:
        """샘플 데이터를 Document, Query 객체로 변환"""
        documents = []
        queries = []

        for sample in samples:
            doc = Document(
                id=sample["id"],
                content=sample["context"],
                language=language,
                metadata={"sample": True}
            )
            documents.append(doc)

            query = Query(
                id=sample["id"],
                question=sample["question"],
                language=language,
                expected_answer=sample["answer"],
                context_id=sample["id"]
            )
            queries.append(query)

        return documents, queries

    def _create_sample_data(self, language: Language) -> Tuple[List[Document], List[Query]]:
        """기존 샘플 데이터 생성 (하위 호환성)"""
        if language == Language.KOREAN:
            samples = [
                {
                    "id": "sample_kr_1",
                    "context": """코로나19(COVID-19)는 SARS-CoV-2 바이러스에 의해 발생하는 호흡기 감염 질환입니다. 
                    2019년 12월 중국 우한에서 처음 발견되었으며, 2020년 3월 WHO에 의해 팬데믹으로 선언되었습니다. 
                    주요 증상으로는 발열, 기침, 호흡곤란, 피로감, 미각 및 후각 상실 등이 있습니다. 
                    대부분의 환자는 경미한 증상을 보이지만, 일부는 중증 폐렴으로 진행될 수 있습니다.
                    특히 고령자나 기저질환이 있는 사람들에게서 중증으로 발전할 위험이 높습니다.""",
                    "question": "코로나19의 주요 증상은 무엇인가요?",
                    "answer": "발열, 기침, 호흡곤란, 피로감, 미각 및 후각 상실"
                },
                {
                    "id": "sample_kr_2",
                    "context": """코로나19 백신은 SARS-CoV-2 바이러스에 대한 면역을 형성하여 COVID-19를 예방합니다. 
                    주요 백신 종류로는 mRNA 백신(화이자, 모더나), 바이러스 벡터 백신(아스트라제네카, 얀센), 
                    불활성화 백신(시노백, 시노팜) 등이 있습니다. 
                    백신 접종은 중증 질환과 사망을 크게 줄이는 것으로 입증되었습니다.
                    전 세계적으로 수십억 명이 안전하게 접종받았으며, 부작용은 대부분 경미합니다.""",
                    "question": "코로나19 백신의 종류에는 어떤 것들이 있나요?",
                    "answer": "mRNA 백신(화이자, 모더나), 바이러스 벡터 백신(아스트라제네카, 얀센), 불활성화 백신(시노백, 시노팜)"
                },
                {
                    "id": "sample_kr_3",
                    "context": """코로나19의 전파는 주로 감염된 사람이 기침, 재채기, 말하기, 노래하기 또는 숨을 쉴 때 
                    나오는 호흡기 비말을 통해 이루어집니다. 
                    밀폐된 공간에서는 공기 중에 떠다니는 작은 입자(에어로졸)를 통한 전파도 가능합니다. 
                    또한 오염된 표면을 만진 후 얼굴을 만지는 것으로도 감염될 수 있습니다.
                    사회적 거리두기, 마스크 착용, 손 위생은 전파를 줄이는 효과적인 방법입니다.""",
                    "question": "코로나19는 어떻게 전파되나요?",
                    "answer": "호흡기 비말, 에어로졸, 오염된 표면 접촉을 통해 전파됩니다"
                }
            ]
        else:
            samples = [
                {
                    "id": "sample_en_1",
                    "context": """COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. 
                    It was first identified in Wuhan, China in December 2019 and was declared a pandemic by the WHO in March 2020. 
                    Common symptoms include fever, cough, shortness of breath, fatigue, and loss of taste or smell. 
                    While most patients experience mild symptoms, some may develop severe pneumonia.
                    The elderly and those with underlying conditions are at higher risk of severe illness.""",
                    "question": "What are the main symptoms of COVID-19?",
                    "answer": "fever, cough, shortness of breath, fatigue, loss of taste or smell"
                },
                {
                    "id": "sample_en_2",
                    "context": """COVID-19 vaccines work by creating immunity against the SARS-CoV-2 virus to prevent COVID-19. 
                    Major vaccine types include mRNA vaccines (Pfizer, Moderna), viral vector vaccines (AstraZeneca, Johnson & Johnson), 
                    and inactivated vaccines (Sinovac, Sinopharm). 
                    Vaccination has been proven to significantly reduce severe illness and death.
                    Billions of people worldwide have been safely vaccinated, with mostly mild side effects.""",
                    "question": "What types of COVID-19 vaccines are available?",
                    "answer": "mRNA vaccines (Pfizer, Moderna), viral vector vaccines (AstraZeneca, Johnson & Johnson), inactivated vaccines (Sinovac, Sinopharm)"
                },
                {
                    "id": "sample_en_3",
                    "context": """COVID-19 spreads primarily through respiratory droplets when an infected person coughs, 
                    sneezes, talks, sings, or breathes. 
                    In enclosed spaces, transmission can also occur through smaller airborne particles (aerosols). 
                    The virus can also spread by touching contaminated surfaces and then touching the face.
                    Social distancing, mask-wearing, and hand hygiene are effective methods to reduce transmission.""",
                    "question": "How does COVID-19 spread?",
                    "answer": "Through respiratory droplets, aerosols, and contaminated surface contact"
                }
            ]

        documents = []
        queries = []

        for sample in samples:
            doc = Document(
                id=sample["id"],
                content=sample["context"],
                language=language,
                metadata={"sample": True}
            )
            documents.append(doc)

            query = Query(
                id=sample["id"],
                question=sample["question"],
                language=language,
                expected_answer=sample["answer"],
                context_id=sample["id"]
            )
            queries.append(query)

        self.logger.info(f"샘플 데이터 생성 완료: {len(documents)}개")
        return documents, queries

    def save_processed_data(
            self,
            documents: List[Document],
            queries: List[Query],
            language: Language
    ):
        """처리된 데이터 저장"""
        output_path = config.paths.data_dir / f"processed_{language.value}.json"

        data = []
        for doc, query in zip(documents, queries):
            data.append({
                "id": doc.id,
                "context": doc.content,
                "question": query.question,
                "answer": query.expected_answer,
                "metadata": {
                    "doc_metadata": doc.metadata,
                    "query_metadata": query.metadata
                }
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"처리된 데이터 저장: {output_path}")