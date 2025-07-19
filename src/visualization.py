"""
결과 시각화 모듈 (최종 수정)
Results Visualization Module (Final Correction)
"""

import json
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from loguru import logger
from config import Language
from src.data_structures import EvaluationResult


class ResultsVisualizer:
    """결과 시각화 클래스"""

    def __init__(self):
        self.logger = logger

    def create_all_visualizations(
            self,
            results: List[EvaluationResult],
            analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """모든 시각화 HTML 문서를 생성합니다."""
        visualizations = {
            "performance_comparison.html": self._create_performance_chart(results),
            "language_analysis.html": self._create_language_analysis_chart(results, analysis),
            "ensemble_effectiveness.html": self._create_ensemble_chart(results),
            "statistical_summary.html": self._create_statistical_summary(analysis),
            "comprehensive_dashboard.html": self._create_comprehensive_dashboard(results, analysis)
        }
        return visualizations

    # ---------------------------------------------------------------------------- #
    # HTML 및 JavaScript 템플릿 헬퍼
    # ---------------------------------------------------------------------------- #

    def _get_html_template(self, title: str, body_content: str, script_content: str) -> str:
        """공통 HTML 템플릿을 생성합니다."""
        return f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-box-and-violin-plot/build/Chart.BoxPlot.min.js"></script>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
                h1 {{ text-align: center; color: #333; margin-bottom: 30px; }}
                .chart-container {{ margin: 30px 0; padding: 20px; background-color: #fafafa; border-radius: 8px; }}
                .chart-title {{ text-align: center; font-size: 18px; font-weight: bold; margin-bottom: 20px; color: #555; }}
                canvas {{ max-height: 400px; width: 100%; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                {body_content}
            </div>
            <script>
                {script_content}
            </script>
        </body>
        </html>
        """

    def _create_bar_chart_script(self, chart_id: str, chart_data: Dict) -> str:
        """Chart.js 바 차트 스크립트를 생성합니다."""
        options = {
            'responsive': True,
            'maintainAspectRatio': False,
            'scales': {
                'y': {'beginAtZero': True, 'title': {'display': True, 'text': chart_data.get('y_label', '')}},
                'x': {'ticks': {'autoSkip': False, 'maxRotation': 45, 'minRotation': 45}}
            },
            'plugins': {
                'legend': {'position': 'top'},
                'tooltip': {
                    'callbacks': {'label': "(context) => context.dataset.label + ': ' + context.parsed.y.toFixed(3)"}}
            }
        }
        if 'y_max' in chart_data:
            options['scales']['y']['max'] = chart_data['y_max']

        # JavaScript 함수를 문자열로 삽입하기 위해 콜백 부분만 치환
        options_str = json.dumps(options).replace(
            '"(context) => context.dataset.label + \': \' + context.parsed.y.toFixed(3)"',
            "(context) => context.dataset.label + ': ' + context.parsed.y.toFixed(3)"
        )

        return f"""
            const {chart_id}Ctx = document.getElementById('{chart_id}').getContext('2d');
            new Chart({chart_id}Ctx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(chart_data['labels'])},
                    datasets: {json.dumps(chart_data['datasets'])}
                }},
                options: {options_str}
            }});
        """

    # ---------------------------------------------------------------------------- #
    # 개별 시각화 생성 메서드
    # ---------------------------------------------------------------------------- #

    def _create_performance_chart(self, results: List[EvaluationResult]) -> str:
        """전략별 성능 비교 차트 HTML을 생성합니다."""
        strategies = sorted(list(set(r.strategy for r in results)))

        data_labels = {'labels': strategies}
        metrics = {
            'auroc': {'y_label': 'AUROC Score (Higher is Better)', 'y_max': 1.0, 'datasets': []},
            'context_rmse': {'y_label': 'RMSE (Lower is Better)', 'datasets': []},
            'utilization_rmse': {'y_label': 'RMSE (Lower is Better)', 'datasets': []},
        }

        lang_map = {
            "english": {"enum": Language.ENGLISH, "color": "rgba(54, 162, 235, 0.7)"},
            "korean": {"enum": Language.KOREAN, "color": "rgba(255, 99, 132, 0.7)"},
        }

        for lang_name, lang_info in lang_map.items():
            dataset_auroc = {'label': lang_name.capitalize(), 'data': [], 'backgroundColor': lang_info['color']}
            dataset_context = {'label': lang_name.capitalize(), 'data': [],
                               'backgroundColor': lang_info['color'].replace('0.7', '0.6')}
            dataset_util = {'label': lang_name.capitalize(), 'data': [],
                            'backgroundColor': lang_info['color'].replace('0.7', '0.5')}

            for strategy in strategies:
                result = next((r for r in results if r.strategy == strategy and r.language == lang_info["enum"]), None)
                dataset_auroc['data'].append(result.hallucination_auroc if result else 0)
                dataset_context['data'].append(result.context_relevance_rmse if result else 0)
                dataset_util['data'].append(result.utilization_rmse if result else 0)

            metrics['auroc']['datasets'].append(dataset_auroc)
            metrics['context_rmse']['datasets'].append(dataset_context)
            metrics['utilization_rmse']['datasets'].append(dataset_util)

        best_auroc = max(r.hallucination_auroc for r in results) if results else 0

        body = f"""
        <div class="chart-container">
            <div class="chart-title">Hallucination AUROC by Strategy and Language</div>
            <canvas id="aurocChart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">Context Relevance RMSE by Strategy and Language</div>
            <canvas id="contextChart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">Utilization RMSE by Strategy and Language</div>
            <canvas id="utilizationChart"></canvas>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 40px; text-align: center;">
            <div style="padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff;">
                <div style="font-size: 24px; font-weight: bold; color: #007bff;">{len(strategies)}</div>
                <div style="color: #666; margin-top: 5px;">Strategies Tested</div>
            </div>
            <div style="padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #28a745;">
                <div style="font-size: 24px; font-weight: bold; color: #28a745;">{len(results)}</div>
                <div style="color: #666; margin-top: 5px;">Total Experiments</div>
            </div>
            <div style="padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #ffc107;">
                <div style="font-size: 24px; font-weight: bold; color: #ffc107;">{best_auroc:.3f}</div>
                <div style="color: #666; margin-top: 5px;">Best AUROC</div>
            </div>
        </div>
        """

        script = f"""
        {self._create_bar_chart_script('aurocChart', {{**data_labels, **metrics['auroc']}})}
        {self._create_bar_chart_script('contextChart', {{**data_labels, **metrics['context_rmse']}})}
        {self._create_bar_chart_script('utilizationChart', {{**data_labels, **metrics['utilization_rmse']}})}
        """

        return self._get_html_template("RAG Chunking Strategy Performance Comparison", body, script)

    def _create_language_analysis_chart(self, results: List[EvaluationResult], analysis: Dict[str, Any]) -> str:
        """언어별 성능 분석 레이더 차트 HTML을 생성합니다."""
        lang_test = analysis.get("statistical_tests", {}).get("language_comparison", {})
        lang_comp = analysis.get("comparative_analysis", {}).get("language_comparison", {})

        # f-string 중첩을 피하기 위해 통계 결과 HTML을 미리 생성
        stats_html = ""
        if "error" in lang_test or not lang_test:
            stats_html = "<p>통계 검증을 위한 데이터가 부족합니다.</p>"
        else:
            is_significant = lang_test.get('significant', False)
            conclusion_color = '#2e7d32' if is_significant else '#757575'
            conclusion_text = '통계적으로 유의미한 차이가 발견되었습니다.' if is_significant else '통계적으로 유의미한 차이가 없습니다.'
            stats_html = f"""
                <p><b>t-statistic:</b> {lang_test.get('t_statistic', 0):.3f}</p>
                <p><b>p-value:</b> {lang_test.get('p_value', 1):.4f}</p>
                <p><b>Cohen's d:</b> {lang_test.get('cohens_d', 0):.3f} ({lang_test.get('effect_size_interpretation', 'N/A')})</p>
                <p style="font-weight: bold; color: {conclusion_color};">{conclusion_text}</p>
            """

        strategies = sorted(list(set(r.strategy for r in results)))
        english_scores = [
            next((r.hallucination_auroc for r in results if r.strategy == s and r.language == Language.ENGLISH), 0) for
            s in strategies]
        korean_scores = [
            next((r.hallucination_auroc for r in results if r.strategy == s and r.language == Language.KOREAN), 0) for s
            in strategies]

        body = f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; text-align: center;">
            <div style="padding: 25px; background-color: #f8f9fa; border-radius: 8px;">
                <div style="color: #666; font-size: 14px;">English Average AUROC</div>
                <div style="font-size: 36px; font-weight: bold; margin: 10px 0; color: #3498db;">{lang_comp.get('english', {}).get('avg_auroc', 0):.3f}</div>
            </div>
            <div style="padding: 25px; background-color: #f8f9fa; border-radius: 8px;">
                <div style="color: #666; font-size: 14px;">Korean Average AUROC</div>
                <div style="font-size: 36px; font-weight: bold; margin: 10px 0; color: #e74c3c;">{lang_comp.get('korean', {}).get('avg_auroc', 0):.3f}</div>
            </div>
        </div>
        <div style="background-color: #f0f9ff; padding: 20px; border-left: 5px solid #03A9F4; border-radius: 8px; margin: 30px 0;">
            <h2 style="margin-top:0;">통계적 유의성 검증 (Paired t-test)</h2>
            {stats_html}
        </div>
        <div class="chart-container" style="height: 500px;">
            <canvas id="radarChart"></canvas>
        </div>
        """

        script = f"""
        const radarCtx = document.getElementById('radarChart').getContext('2d');
        new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: {json.dumps(strategies)},
                datasets: [
                    {{ label: 'English', data: {json.dumps(english_scores)}, borderColor: 'rgba(54, 162, 235, 1)', backgroundColor: 'rgba(54, 162, 235, 0.2)' }},
                    {{ label: 'Korean', data: {json.dumps(korean_scores)}, borderColor: 'rgba(255, 99, 132, 1)', backgroundColor: 'rgba(255, 99, 132, 0.2)' }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ r: {{ beginAtZero: true, max: 1.0, ticks: {{ stepSize: 0.2 }} }} }},
                plugins: {{ title: {{ display: true, text: 'Strategy Performance by Language', font: {{ size: 18 }} }} }}
            }}
        }});
        """

        return self._get_html_template("Language Performance Analysis", body, script)

    def _create_ensemble_chart(self, results: List[EvaluationResult]) -> str:
        """앙상블 전략의 효과 분석 차트 HTML을 생성합니다."""
        individual_results = [r for r in results if "ensemble" not in r.strategy]
        ensemble_results = [r for r in results if "ensemble" in r.strategy]

        individual_avg = np.mean([r.hallucination_auroc for r in individual_results]) if individual_results else 0
        ensemble_avg = np.mean([r.hallucination_auroc for r in ensemble_results]) if ensemble_results else 0
        improvement = ((ensemble_avg - individual_avg) / individual_avg * 100) if individual_avg > 0 else 0

        ensemble_methods = {}
        for r in ensemble_results:
            method = r.strategy.replace("ensemble_", "")
            if method not in ensemble_methods:
                ensemble_methods[method] = []
            ensemble_methods[method].append(r.hallucination_auroc)

        ensemble_method_labels = list(ensemble_methods.keys())
        ensemble_method_scores = [np.mean(scores) for scores in ensemble_methods.values()]

        all_strategies = sorted(list(set(r.strategy for r in results)))
        all_avg_scores = [np.mean([res.hallucination_auroc for res in results if res.strategy == s]) for s in
                          all_strategies]
        background_colors = ['rgba(153, 102, 255, 0.8)' if 'ensemble' in s else 'rgba(75, 192, 192, 0.8)' for s in
                             all_strategies]

        # 차트 데이터 미리 구성
        method_chart_data = {
            'labels': ensemble_method_labels,
            'datasets': [{'label': 'Average AUROC', 'data': ensemble_method_scores,
                          'backgroundColor': 'rgba(255, 159, 64, 0.8)'}],
            'y_label': 'AUROC Score', 'y_max': 1.0
        }
        detailed_chart_data = {
            'labels': all_strategies,
            'datasets': [{'label': 'Average AUROC', 'data': all_avg_scores, 'backgroundColor': background_colors}],
            'y_label': 'AUROC Score', 'y_max': 1.0
        }

        body = f"""
        <div style="background-color: #e3f2fd; padding: 25px; border-radius: 8px; margin-bottom: 30px; text-align: center;">
            <div style="color: #666; font-size: 16px;">Overall Improvement</div>
            <div style="font-size: 48px; font-weight: bold; color: #1976d2; margin: 15px 0;">{improvement:.1f}%</div>
            <div style="color: #666; font-size: 16px;">Ensemble vs. Individual Strategies (AUROC)</div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
            <div class="chart-container" style="height: 350px;"><canvas id="comparisonDoughnut"></canvas></div>
            <div class="chart-container" style="height: 350px;"><canvas id="methodBarChart"></canvas></div>
        </div>
        <div class="chart-container" style="height: 450px;">
            <div class="chart-title">Detailed Performance of All Strategies</div>
            <canvas id="detailedBarChart"></canvas>
        </div>
        """

        script = f"""
        const doughnutCtx = document.getElementById('comparisonDoughnut').getContext('2d');
        new Chart(doughnutCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Individual Strategies', 'Ensemble Strategies'],
                datasets: [{{
                    data: [{individual_avg:.3f}, {ensemble_avg:.3f}],
                    backgroundColor: ['rgba(75, 192, 192, 0.8)', 'rgba(153, 102, 255, 0.8)']
                }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ display: true, text: 'Average AUROC Comparison' }} }} }}
        }});

        {self._create_bar_chart_script('methodBarChart', method_chart_data)}
        {self._create_bar_chart_script('detailedBarChart', detailed_chart_data)}
        """

        return self._get_html_template("Ensemble Strategy Effectiveness", body, script)

    def _create_statistical_summary(self, analysis: Dict[str, Any]) -> str:
        """통계 분석 요약 페이지 HTML을 생성합니다."""
        stats = analysis.get("statistical_tests", {})

        def generate_test_card(test_key, test_name, test_data):
            if not test_data or "error" in test_data:
                return f"""<div class="test-card"><div class="test-name">{test_name}</div><p>결과 없음 또는 오류 발생.</p></div>"""

            is_significant = test_data.get('significant', False)
            sig_class = 'significant' if is_significant else 'not-significant'

            p_value_html = f"<p><b>p-value:</b> {test_data['p_value']:.4f}</p>" if 'p_value' in test_data else ""
            stat_html = ""
            if 't_statistic' in test_data:
                stat_html = f"<p><b>t-statistic:</b> {test_data['t_statistic']:.3f}</p>"
            elif 'f_statistic' in test_data:
                stat_html = f"<p><b>F-statistic:</b> {test_data['f_statistic']:.3f}</p>"

            effect_size_html = f"<p><b>Cohen's d:</b> {test_data['cohens_d']:.3f} ({test_data['effect_size_interpretation']})</p>" if 'cohens_d' in test_data else ""
            conclusion_text = '통계적으로 유의미한 차이가 있습니다.' if is_significant else '통계적으로 유의미한 차이가 없습니다.'

            return f"""
            <div class="test-card {sig_class}">
                <div class="test-name">{test_name}</div>
                {p_value_html}
                {stat_html}
                {effect_size_html}
                <p><b>결론:</b> {conclusion_text}</p>
            </div>
            """

        test_names = {
            "language_comparison": "언어별 성능 비교 (Paired t-test)",
            "ensemble_test": "앙상블 vs 개별 전략 비교 (t-test)",
            "strategy_anova": "전략 간 성능 비교 (ANOVA)",
        }

        cards_html = "".join([generate_test_card(key, name, stats.get(key, {})) for key, name in test_names.items()])

        body = f"""
        <style>
            .test-card {{ margin: 20px 0; padding: 25px; border-radius: 8px; border-left: 5px solid; }}
            .test-name {{ font-size: 20px; font-weight: bold; margin-bottom: 15px; }}
            .significant {{ border-color: #28a745; background-color: #e8f5e9; }}
            .not-significant {{ border-color: #dc3545; background-color: #fbeae5; }}
        </style>
        {cards_html}
        """

        return self._get_html_template("Statistical Analysis Summary", body, "")

    def _create_comprehensive_dashboard(self, results: List[EvaluationResult], analysis: Dict[str, Any]) -> str:
        """모든 분석 결과를 종합한 대시보드 HTML을 생성합니다."""
        if not results:
            return self._get_html_template("Comprehensive Dashboard", "<p>No data to analyze.</p>", "")

        best_result = max(results, key=lambda r: r.hallucination_auroc)
        baseline_auroc = analysis.get("comparative_analysis", {}).get("baseline_comparison", {}).get("baseline_auroc",
                                                                                                     0.57)
        improvement = ((
                                   best_result.hallucination_auroc - baseline_auroc) / baseline_auroc * 100) if baseline_auroc > 0 else 0
        significant_tests = sum(1 for test in analysis.get('statistical_tests', {}).values() if
                                isinstance(test, dict) and test.get('significant', False))

        # --- Chart Data Preparation ---
        strategies = sorted(list(set(r.strategy for r in results)))
        avg_scores = [np.mean([res.hallucination_auroc for res in results if res.strategy == s]) for s in strategies]
        overview_chart_data = {
            'labels': strategies,
            'datasets': [{'label': 'Average AUROC', 'data': avg_scores,
                          'backgroundColor': ['rgba(153, 102, 255, 0.7)' if 'ens' in s else 'rgba(54, 162, 235, 0.7)'
                                              for s in strategies]}],
            'y_label': 'AUROC', 'y_max': 1.0
        }

        en_scores = [r.hallucination_auroc for r in results if r.language == Language.ENGLISH]
        kr_scores = [r.hallucination_auroc for r in results if r.language == Language.KOREAN]

        individual_avg = np.mean([r.hallucination_auroc for r in results if 'ensemble' not in r.strategy]) or 0
        ensemble_avg = np.mean([r.hallucination_auroc for r in results if 'ensemble' in r.strategy]) or 0

        ci_data = analysis.get('confidence_intervals', {}).get('by_strategy', {})
        ci_labels = list(ci_data.keys())
        ci_values = [[d['lower'], d['upper']] for d in ci_data.values()]

        body = f"""
        <style>
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .metric-card {{ background-color: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center; }}
            .metric-value {{ font-size: 36px; font-weight: bold; margin: 10px 0; }}
            .metric-label {{ color: #666; font-size: 14px; }}
            .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .chart-card {{ background-color: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
            .footer {{ text-align: center; margin-top: 40px; color: #666; }}
            @media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
        </style>
        <div class="metrics-grid">
            <div class="metric-card"><div class="metric-label">Best Performance (AUROC)</div><div class="metric-value" style="color:#4caf50;">{best_result.hallucination_auroc:.3f}</div><div class="metric-label">{best_result.strategy}</div></div>
            <div class="metric-card"><div class="metric-label">Improvement vs Baseline</div><div class="metric-value" style="color:#2196f3;">{improvement:.1f}%</div><div class="metric-label">Over Baseline ({baseline_auroc:.2f})</div></div>
            <div class="metric-card"><div class="metric-label">Total Experiments</div><div class="metric-value" style="color:#ff9800;">{len(results)}</div><div class="metric-label">Strategies × Languages</div></div>
            <div class="metric-card"><div class="metric-label">Significant Tests</div><div class="metric-value" style="color:#9c27b0;">{significant_tests}</div><div class="metric-label">at p < 0.05</div></div>
        </div>
        <div class="charts-grid">
            <div class="chart-card"><div class="chart-title">Strategy Performance Overview</div><canvas id="overviewChart"></canvas></div>
            <div class="chart-card"><div class="chart-title">Language Performance Distribution</div><canvas id="languageChart"></canvas></div>
            <div class="chart-card"><div class="chart-title">Ensemble vs Individual</div><canvas id="ensembleChart"></canvas></div>
            <div class="chart-card"><div class="chart-title">95% Confidence Intervals by Strategy</div><canvas id="confidenceChart"></canvas></div>
        </div>
        <div class="footer"><p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div>
        """

        script = f"""
        {self._create_bar_chart_script('overviewChart', overview_chart_data)}

        const langCtx = document.getElementById('languageChart').getContext('2d');
        new Chart(langCtx, {{
            type: 'boxplot',
            data: {{
                labels: ['English', 'Korean'],
                datasets: [{{
                    label: 'AUROC Distribution',
                    data: [{json.dumps(en_scores)}, {json.dumps(kr_scores)}],
                    backgroundColor: ['rgba(54, 162, 235, 0.5)', 'rgba(255, 99, 132, 0.5)'],
                    borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)'],
                    borderWidth: 1
                }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false, plugins:{{ legend: {{ display: false }} }} }}
        }});

        const ensCtx = document.getElementById('ensembleChart').getContext('2d');
        new Chart(ensCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Individual', 'Ensemble'],
                datasets: [{{ data: [{individual_avg}, {ensemble_avg}], backgroundColor: ['rgba(75, 192, 192, 0.8)', 'rgba(153, 102, 255, 0.8)'] }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false }}
        }});

        const confCtx = document.getElementById('confidenceChart').getContext('2d');
        new Chart(confCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(ci_labels)},
                datasets: [{{
                    label: '95% Confidence Interval',
                    data: {json.dumps(ci_values)},
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderWidth: 1,
                    borderSkipped: false
                }}]
            }},
            options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins:{{ legend: {{ display: false }} }}, scales: {{ x: {{ min: 0.5, max: 1.0 }} }} }}
        }});
        """

        return self._get_html_template("RAG Chunking Strategy Comprehensive Dashboard", body, script)