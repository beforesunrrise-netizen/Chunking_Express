"""
통계 분석 모듈
Statistical analysis module
"""

from typing import List, Dict, Any
import numpy as np
from scipy import stats
from loguru import logger

from src.config import Language
from src.data_structures import EvaluationResult


class StatisticalAnalyzer:
    """통계 분석 클래스"""

    def __init__(self):
        self.logger = logger

    def analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """전체 결과 분석"""
        analysis = {
            "descriptive_statistics": self._calculate_descriptive_stats(results),
            "comparative_analysis": self._perform_comparative_analysis(results),
            "statistical_tests": self._perform_statistical_tests(results),
            "effect_sizes": self._calculate_effect_sizes(results),
            "confidence_intervals": self._calculate_confidence_intervals(results)
        }

        return analysis

    def _calculate_descriptive_stats(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """기술 통계 계산"""
        stats_by_strategy = {}
        stats_by_language = {}

        # 전략별 통계
        for result in results:
            strategy = result.strategy
            if strategy not in stats_by_strategy:
                stats_by_strategy[strategy] = {
                    "auroc_scores": [],
                    "context_rmse_scores": [],
                    "utilization_rmse_scores": [],
                    "languages": []
                }

            stats_by_strategy[strategy]["auroc_scores"].append(result.hallucination_auroc)
            stats_by_strategy[strategy]["context_rmse_scores"].append(result.context_relevance_rmse)
            stats_by_strategy[strategy]["utilization_rmse_scores"].append(result.utilization_rmse)
            stats_by_strategy[strategy]["languages"].append(result.language.value)

        # 언어별 통계
        for result in results:
            lang = result.language.value
            if lang not in stats_by_language:
                stats_by_language[lang] = {
                    "auroc_scores": [],
                    "context_rmse_scores": [],
                    "utilization_rmse_scores": [],
                    "strategies": []
                }

            stats_by_language[lang]["auroc_scores"].append(result.hallucination_auroc)
            stats_by_language[lang]["context_rmse_scores"].append(result.context_relevance_rmse)
            stats_by_language[lang]["utilization_rmse_scores"].append(result.utilization_rmse)
            stats_by_language[lang]["strategies"].append(result.strategy)

        # 통계 계산
        descriptive_stats = {
            "by_strategy": {},
            "by_language": {},
            "overall": {}
        }

        # 전략별 통계
        for strategy, scores in stats_by_strategy.items():
            descriptive_stats["by_strategy"][strategy] = self._compute_stats(scores)

        # 언어별 통계
        for lang, scores in stats_by_language.items():
            descriptive_stats["by_language"][lang] = self._compute_stats(scores)

        # 전체 통계
        all_auroc = [r.hallucination_auroc for r in results]
        all_context = [r.context_relevance_rmse for r in results]
        all_util = [r.utilization_rmse for r in results]

        descriptive_stats["overall"] = {
            "auroc": self._compute_single_metric_stats(all_auroc),
            "context_rmse": self._compute_single_metric_stats(all_context),
            "utilization_rmse": self._compute_single_metric_stats(all_util),
            "total_samples": len(results)
        }

        return descriptive_stats

    def _compute_stats(self, scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """점수 집합에 대한 통계 계산"""
        return {
            "auroc": self._compute_single_metric_stats(scores["auroc_scores"]),
            "context_rmse": self._compute_single_metric_stats(scores["context_rmse_scores"]),
            "utilization_rmse": self._compute_single_metric_stats(scores["utilization_rmse_scores"]),
            "n_samples": len(scores["auroc_scores"])
        }

    def _compute_single_metric_stats(self, values: List[float]) -> Dict[str, float]:
        """단일 메트릭에 대한 통계"""
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75))
        }

    def _perform_comparative_analysis(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """비교 분석"""
        comparisons = {
            "language_comparison": self._compare_languages(results),
            "strategy_comparison": self._compare_strategies(results),
            "ensemble_effectiveness": self._analyze_ensemble_effectiveness(results)
        }

        return comparisons

    def _compare_languages(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """언어별 비교"""
        en_results = [r for r in results if r.language == Language.ENGLISH]
        kr_results = [r for r in results if r.language == Language.KOREAN]

        comparison = {
            "english": {
                "n_samples": len(en_results),
                "avg_auroc": np.mean([r.hallucination_auroc for r in en_results]) if en_results else 0,
                "best_strategy": max(en_results, key=lambda r: r.hallucination_auroc).strategy if en_results else None
            },
            "korean": {
                "n_samples": len(kr_results),
                "avg_auroc": np.mean([r.hallucination_auroc for r in kr_results]) if kr_results else 0,
                "best_strategy": max(kr_results, key=lambda r: r.hallucination_auroc).strategy if kr_results else None
            }
        }

        # 성능 격차
        if en_results and kr_results:
            comparison["performance_gap"] = abs(
                comparison["english"]["avg_auroc"] - comparison["korean"]["avg_auroc"]
            )
            comparison["better_language"] = "english" if comparison["english"]["avg_auroc"] > comparison["korean"]["avg_auroc"] else "korean"

        return comparison

    def _compare_strategies(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """전략별 비교"""
        strategy_comparison = {}

        for strategy in set(r.strategy for r in results):
            strategy_results = [r for r in results if r.strategy == strategy]

            en_results = [r for r in strategy_results if r.language == Language.ENGLISH]
            kr_results = [r for r in strategy_results if r.language == Language.KOREAN]

            strategy_comparison[strategy] = {
                "overall_auroc": np.mean([r.hallucination_auroc for r in strategy_results]),
                "english_auroc": np.mean([r.hallucination_auroc for r in en_results]) if en_results else None,
                "korean_auroc": np.mean([r.hallucination_auroc for r in kr_results]) if kr_results else None,
                "language_consistency": self._calculate_consistency(en_results, kr_results) if en_results and kr_results else None
            }

        # 최고 성능 전략
        best_strategy = max(strategy_comparison.items(), key=lambda x: x[1]["overall_auroc"])
        strategy_comparison["best_overall"] = best_strategy[0]

        return strategy_comparison

    def _analyze_ensemble_effectiveness(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """앙상블 효과 분석"""
        individual_results = [r for r in results if "ensemble" not in r.strategy]
        ensemble_results = [r for r in results if "ensemble" in r.strategy]

        if not individual_results or not ensemble_results:
            return {"analysis": "insufficient_data"}

        individual_auroc = [r.hallucination_auroc for r in individual_results]
        ensemble_auroc = [r.hallucination_auroc for r in ensemble_results]

        effectiveness = {
            "individual_mean": np.mean(individual_auroc),
            "ensemble_mean": np.mean(ensemble_auroc),
            "improvement": np.mean(ensemble_auroc) - np.mean(individual_auroc),
            "improvement_percentage": ((np.mean(ensemble_auroc) - np.mean(individual_auroc)) / np.mean(individual_auroc)) * 100,
            "best_individual": max(individual_auroc),
            "best_ensemble": max(ensemble_auroc),
            "consistency_improved": np.std(ensemble_auroc) < np.std(individual_auroc)
        }

        return effectiveness

    def _calculate_consistency(self, en_results: List[EvaluationResult], kr_results: List[EvaluationResult]) -> float:
        """언어 간 일관성 계산"""
        if not en_results or not kr_results:
            return 0.0

        en_auroc = [r.hallucination_auroc for r in en_results]
        kr_auroc = [r.hallucination_auroc for r in kr_results]

        # 평균 차이의 역수를 일관성 지표로 사용
        diff = abs(np.mean(en_auroc) - np.mean(kr_auroc))
        consistency = 1 / (1 + diff)  # 0~1 범위로 정규화

        return consistency

    def _perform_statistical_tests(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """통계적 검정"""
        tests = {}

        # 1. 언어 간 대응표본 t-검정
        tests["language_comparison"] = self._paired_language_test(results)

        # 2. 베이스라인과 제안 방법 비교
        tests["baseline_comparison"] = self._baseline_comparison_test(results)

        # 3. 전략 간 ANOVA
        tests["strategy_anova"] = self._perform_anova(results)

        # 4. 앙상블 vs 개별 전략 검정
        tests["ensemble_test"] = self._ensemble_comparison_test(results)

        return tests

    def _paired_language_test(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """언어 간 대응표본 t-검정"""
        # 같은 전략에 대한 언어별 결과 매칭
        paired_data = []

        strategies = set(r.strategy for r in results if "ensemble" not in r.strategy)

        for strategy in strategies:
            en_result = next((r for r in results if r.strategy == strategy and r.language == Language.ENGLISH), None)
            kr_result = next((r for r in results if r.strategy == strategy and r.language == Language.KOREAN), None)

            if en_result and kr_result:
                paired_data.append((en_result.hallucination_auroc, kr_result.hallucination_auroc))

        if len(paired_data) < 2:
            return {"error": "insufficient_paired_data"}

        en_scores, kr_scores = zip(*paired_data)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(en_scores, kr_scores)

        # Effect size (Cohen's d)
        cohens_d = self._calculate_cohens_d(list(en_scores), list(kr_scores))

        return {
            "test": "paired_t_test",
            "n_pairs": len(paired_data),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "effect_size_interpretation": self._interpret_effect_size(cohens_d),
            "significant": p_value < 0.05,
            "mean_difference": float(np.mean(en_scores) - np.mean(kr_scores))
        }

    def _baseline_comparison_test(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """베이스라인과 비교"""
        baseline_auroc = 0.57  # GPT-3.5 Judge baseline

        # 제안된 방법들의 AUROC
        proposed_scores = [r.hallucination_auroc for r in results]

        if not proposed_scores:
            return {"error": "no_results"}

        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(proposed_scores, baseline_auroc)

        # 개선율
        mean_proposed = np.mean(proposed_scores)
        improvement = ((mean_proposed - baseline_auroc) / baseline_auroc) * 100

        return {
            "test": "one_sample_t_test",
            "baseline": baseline_auroc,
            "proposed_mean": float(mean_proposed),
            "proposed_std": float(np.std(proposed_scores)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "improvement_percentage": float(improvement),
            "better_than_baseline": mean_proposed > baseline_auroc
        }

    def _perform_anova(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """전략 간 분산분석"""
        # 전략별 AUROC 점수 그룹화
        strategy_groups = {}

        for result in results:
            if "ensemble" not in result.strategy:  # 개별 전략만
                if result.strategy not in strategy_groups:
                    strategy_groups[result.strategy] = []
                strategy_groups[result.strategy].append(result.hallucination_auroc)

        if len(strategy_groups) < 2:
            return {"error": "insufficient_groups"}

        # 각 그룹이 최소 2개 이상의 샘플을 가져야 함
        valid_groups = [scores for scores in strategy_groups.values() if len(scores) >= 2]

        if len(valid_groups) < 2:
            return {"error": "insufficient_samples_per_group"}

        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*valid_groups)

        # 사후 검정 (Tukey HSD는 별도 구현 필요)
        return {
            "test": "one_way_anova",
            "n_groups": len(valid_groups),
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "interpretation": "전략 간 유의미한 차이 있음" if p_value < 0.05 else "전략 간 유의미한 차이 없음"
        }

    def _ensemble_comparison_test(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """앙상블 vs 개별 전략 비교"""
        individual_scores = [r.hallucination_auroc for r in results if "ensemble" not in r.strategy]
        ensemble_scores = [r.hallucination_auroc for r in results if "ensemble" in r.strategy]

        if not individual_scores or not ensemble_scores:
            return {"error": "insufficient_data"}

        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(ensemble_scores, individual_scores)

        # Effect size
        cohens_d = self._calculate_cohens_d_independent(ensemble_scores, individual_scores)

        return {
            "test": "independent_t_test",
            "individual_mean": float(np.mean(individual_scores)),
            "ensemble_mean": float(np.mean(ensemble_scores)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "effect_size_interpretation": self._interpret_effect_size(cohens_d),
            "significant": p_value < 0.05,
            "ensemble_better": np.mean(ensemble_scores) > np.mean(individual_scores)
        }

    def _calculate_effect_sizes(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """효과 크기 계산"""
        effect_sizes = {}

        # 1. 언어 간 효과 크기
        en_results = [r.hallucination_auroc for r in results if r.language == Language.ENGLISH]
        kr_results = [r.hallucination_auroc for r in results if r.language == Language.KOREAN]

        if en_results and kr_results:
            effect_sizes["language_effect"] = {
                "cohens_d": float(self._calculate_cohens_d_independent(en_results, kr_results)),
                "interpretation": self._interpret_effect_size(
                    self._calculate_cohens_d_independent(en_results, kr_results)
                )
            }

        # 2. 각 전략의 베이스라인 대비 효과 크기
        baseline = 0.57
        strategy_effects = {}

        for strategy in set(r.strategy for r in results):
            strategy_scores = [r.hallucination_auroc for r in results if r.strategy == strategy]
            if strategy_scores:
                d = (np.mean(strategy_scores) - baseline) / np.std(strategy_scores) if np.std(strategy_scores) > 0 else 0
                strategy_effects[strategy] = {
                    "cohens_d": float(d),
                    "interpretation": self._interpret_effect_size(d)
                }

        effect_sizes["strategy_effects"] = strategy_effects

        return effect_sizes

    def _calculate_confidence_intervals(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """신뢰구간 계산"""
        confidence_intervals = {}

        # 전체 AUROC의 95% 신뢰구간
        all_auroc = [r.hallucination_auroc for r in results]
        if all_auroc:
            ci_overall = self._compute_confidence_interval(all_auroc)
            confidence_intervals["overall_auroc"] = ci_overall

        # 언어별 신뢰구간
        for lang in [Language.ENGLISH, Language.KOREAN]:
            lang_scores = [r.hallucination_auroc for r in results if r.language == lang]
            if lang_scores:
                confidence_intervals[f"{lang.value}_auroc"] = self._compute_confidence_interval(lang_scores)

        # 전략별 신뢰구간
        strategy_ci = {}
        for strategy in set(r.strategy for r in results):
            strategy_scores = [r.hallucination_auroc for r in results if r.strategy == strategy]
            if strategy_scores:
                strategy_ci[strategy] = self._compute_confidence_interval(strategy_scores)

        confidence_intervals["by_strategy"] = strategy_ci

        return confidence_intervals

    def _compute_confidence_interval(
        self,
        data: List[float],
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """95% 신뢰구간 계산"""
        n = len(data)
        if n < 2:
            return {"lower": 0, "upper": 0, "mean": np.mean(data) if data else 0}

        mean = np.mean(data)
        sem = stats.sem(data)  # 표준 오차

        # t-분포 사용
        interval = stats.t.interval(confidence, n-1, loc=mean, scale=sem)

        return {
            "mean": float(mean),
            "lower": float(interval[0]),
            "upper": float(interval[1]),
            "margin_of_error": float(interval[1] - mean)
        }

    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Cohen's d 계산 (대응표본)"""
        if not group1 or not group2:
            return 0.0

        diff = np.array(group1) - np.array(group2)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)

        if std_diff == 0:
            return 0.0

        return mean_diff / std_diff

    def _calculate_cohens_d_independent(self, group1: List[float], group2: List[float]) -> float:
        """Cohen's d 계산 (독립표본)"""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0

        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """효과 크기 해석"""
        d = abs(cohens_d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def generate_statistical_report(self, analysis: Dict[str, Any]) -> str:
        """통계 분석 보고서 생성"""
        report = []
        report.append("통계 분석 보고서")
        report.append("=" * 50)

        # 기술 통계
        report.append("\n1. 기술 통계")
        report.append("-" * 30)
        overall = analysis["descriptive_statistics"]["overall"]
        report.append(f"전체 AUROC: {overall['auroc']['mean']:.3f} ± {overall['auroc']['std']:.3f}")
        report.append(f"전체 샘플 수: {overall['total_samples']}")

        # 언어별 비교
        report.append("\n2. 언어별 성능 비교")
        report.append("-" * 30)
        lang_comp = analysis["comparative_analysis"]["language_comparison"]
        report.append(f"영어 평균 AUROC: {lang_comp['english']['avg_auroc']:.3f}")
        report.append(f"한국어 평균 AUROC: {lang_comp['korean']['avg_auroc']:.3f}")
        if "performance_gap" in lang_comp:
            report.append(f"성능 격차: {lang_comp['performance_gap']:.3f}")

        # 통계적 검정
        report.append("\n3. 통계적 검정 결과")
        report.append("-" * 30)

        # 언어 간 검정
        lang_test = analysis["statistical_tests"].get("language_comparison", {})
        if "error" not in lang_test:
            report.append(f"언어 간 대응표본 t-검정:")
            report.append(f"  - t = {lang_test.get('t_statistic', 'N/A'):.3f}")
            report.append(f"  - p = {lang_test.get('p_value', 'N/A'):.4f}")
            report.append(f"  - Cohen's d = {lang_test.get('cohens_d', 'N/A'):.3f}")
            report.append(f"  - 유의성: {'유의함' if lang_test.get('significant', False) else '유의하지 않음'}")

        # 베이스라인 비교
        baseline_test = analysis["statistical_tests"].get("baseline_comparison", {})
        if "error" not in baseline_test:
            report.append(f"\n베이스라인 대비 성능:")
            report.append(f"  - 베이스라인 AUROC: {baseline_test.get('baseline', 'N/A'):.3f}")
            report.append(f"  - 제안 방법 평균: {baseline_test.get('proposed_mean', 'N/A'):.3f}")
            report.append(f"  - 개선율: {baseline_test.get('improvement_percentage', 'N/A'):.1f}%")
            report.append(f"  - p = {baseline_test.get('p_value', 'N/A'):.4f}")

        # 앙상블 효과
        report.append("\n4. 앙상블 효과 분석")
        report.append("-" * 30)
        ensemble_eff = analysis["comparative_analysis"]["ensemble_effectiveness"]
        if "analysis" not in ensemble_eff:
            report.append(f"개별 전략 평균: {ensemble_eff.get('individual_mean', 'N/A'):.3f}")
            report.append(f"앙상블 평균: {ensemble_eff.get('ensemble_mean', 'N/A'):.3f}")
            report.append(f"개선율: {ensemble_eff.get('improvement_percentage', 'N/A'):.1f}%")

        return "\n".join(report)