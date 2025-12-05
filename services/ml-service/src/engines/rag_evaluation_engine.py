"""
RAG Evaluation Engine

Comprehensive evaluation metrics for RAG (Retrieval-Augmented Generation) systems:
- Retrieval metrics: Precision@k, Recall@k, MRR, nDCG
- Generation metrics: Faithfulness, Answer Relevance, Hallucination Detection
- End-to-end metrics: Helpfulness, Latency

Supports both automated (LLM-as-judge) and human evaluation workflows.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import logging

from core.gemma_summarizer import GemmaSummarizer

logger = logging.getLogger(__name__)


class RAGEvaluationEngine:
    """Evaluate RAG system performance"""
    
    def __init__(self):
        self.name = "RAG Evaluation Engine"
        self.evaluation_cache = {}
    
    def get_engine_info(self) -> Dict[str, str]:
        """Get engine display information."""
        return {
            "name": "rag_evaluation",
            "display_name": "RAG Evaluation",
            "icon": "🎯",
            "task_type": "detection"
        }
    
    def get_config_schema(self) -> List[Any]:
        """Get configuration schema for UI."""
        from core.premium_models import ConfigParameter
        return [
            ConfigParameter(
                name="k_values",
                type="select",
                default=[1, 3, 5, 10],
                range=[1, 3, 5, 10, 20],
                description="K values for Precision@k and Recall@k"
            ),
            ConfigParameter(
                name="use_llm_judge",
                type="bool",
                default=False,
                range=None,
                description="Use LLM as judge for generation quality"
            )
        ]
    
    def get_methodology_info(self) -> Dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "RAG System Evaluation",
            "url": "https://arxiv.org/abs/2309.01431",
            "steps": [
                {"step_number": 1, "title": "Retrieval Evaluation", "description": "Calculate Precision@k, Recall@k, MRR, nDCG"},
                {"step_number": 2, "title": "Generation Evaluation", "description": "Assess faithfulness and answer relevance"},
                {"step_number": 3, "title": "End-to-End Metrics", "description": "Measure overall helpfulness and latency"},
                {"step_number": 4, "title": "Summary Generation", "description": "Aggregate metrics and generate insights"}
            ],
            "limitations": ["Ground truth required for retrieval metrics", "LLM judge may have biases"],
            "assumptions": ["Test cases have relevant documents annotated", "RAG responses include retrieved documents"]
        }
    
    def analyze(self, df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Standard analyze() interface for compatibility with engine registry.
        For RAG evaluation, this requires special handling.
        """
        config = config or {}
        # Extract test cases and responses from dataframe if structured appropriately
        if 'query' in df.columns and 'retrieved_docs' in df.columns:
            test_cases = df[['query', 'relevant_docs']].to_dict('records') if 'relevant_docs' in df.columns else []
            rag_responses = df[['query', 'retrieved_docs', 'generated_answer']].to_dict('records') if 'generated_answer' in df.columns else []
            return self.evaluate(test_cases, rag_responses, config)
        
        # Use Gemma fallback when required columns are missing
        return GemmaSummarizer.generate_fallback_summary(
            df,
            engine_name="rag_evaluation",
            error_reason="RAG evaluation requires 'query', 'relevant_docs', 'retrieved_docs', and 'generated_answer' columns",
            config=config
        )
    
    def evaluate(self, test_set: List[Dict], rag_responses: List[Dict], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive RAG evaluation
        
        Args:
            test_set: List of test cases, each with:
                - query: str
                - relevant_docs: List[str] (ground truth)
                - expected_answer: str (optional)
            rag_responses: List of RAG system responses, each with:
                - query: str
                - retrieved_docs: List[str]
                - generated_answer: str
                - retrieval_scores: List[float] (optional)
            config: Evaluation configuration
                - k_values: List of k for Precision@k, Recall@k
                - use_llm_judge: bool (default False)
                - llm_endpoint: str (if using LLM judge)
        
        Returns:
            Dictionary with all evaluation metrics and visualizations
        """
        k_values = config.get('k_values', [1, 3, 5, 10])
        
        results = {
            'retrieval_metrics': self._evaluate_retrieval(test_set, rag_responses, k_values),
            'generation_metrics': self._evaluate_generation(test_set, rag_responses, config),
            'end_to_end_metrics': self._evaluate_end_to_end(test_set, rag_responses),
            'visualizations': [],
            'summary': {}
        }
        
        # Generate visualizations
        results['visualizations'] = self._generate_eval_visualizations(results)
        
        # Generate summary
        results['summary'] = self._generate_eval_summary(results)
        
        return results
    
    def _evaluate_retrieval(self, test_set: List[Dict], rag_responses: List[Dict], k_values: List[int]) -> Dict:
        """Calculate retrieval quality metrics"""
        metrics = {
            'precision_at_k': {},
            'recall_at_k': {},
            'f1_at_k': {},
            'mrr': 0.0,
            'ndcg_at_k': {},
            'context_relevance': 0.0
        }
        
        precision_at_k_scores = {k: [] for k in k_values}
        recall_at_k_scores = {k: [] for k in k_values}
        reciprocal_ranks = []
        ndcg_at_k_scores = {k: [] for k in k_values}
        
        for test_case, response in zip(test_set, rag_responses):
            relevant_docs = set(test_case.get('relevant_docs', []))
            retrieved_docs = response.get('retrieved_docs', [])
            
            if not relevant_docs or not retrieved_docs:
                continue
            
            # Calculate Precision@k and Recall@k for each k
            for k in k_values:
                top_k = set(retrieved_docs[:k])
                
                # Precision@k
                precision = len(top_k & relevant_docs) / k if k > 0 else 0
                precision_at_k_scores[k].append(precision)
                
                # Recall@k
                recall = len(top_k & relevant_docs) / len(relevant_docs) if len(relevant_docs) > 0 else 0
                recall_at_k_scores[k].append(recall)
            
            # Mean Reciprocal Rank (MRR)
            for rank, doc in enumerate(retrieved_docs, 1):
                if doc in relevant_docs:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
            
            # nDCG@k
            for k in k_values:
                ndcg = self._calculate_ndcg(retrieved_docs[:k], relevant_docs, k)
                ndcg_at_k_scores[k].append(ndcg)
        
        # Aggregate metrics
        for k in k_values:
            metrics['precision_at_k'][f'p@{k}'] = float(np.mean(precision_at_k_scores[k])) if precision_at_k_scores[k] else 0.0
            metrics['recall_at_k'][f'r@{k}'] = float(np.mean(recall_at_k_scores[k])) if recall_at_k_scores[k] else 0.0
            
            # F1@k
            p = metrics['precision_at_k'][f'p@{k}']
            r = metrics['recall_at_k'][f'r@{k}']
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            metrics['f1_at_k'][f'f1@{k}'] = float(f1)
            
            metrics['ndcg_at_k'][f'ndcg@{k}'] = float(np.mean(ndcg_at_k_scores[k])) if ndcg_at_k_scores[k] else 0.0
        
        metrics['mrr'] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        
        return metrics
    
    def _calculate_ndcg(self, retrieved: List[str], relevant: set, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        # Binary relevance: 1 if doc is relevant, 0 otherwise
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k], 1):
            if doc in relevant:
                # DCG formula: sum(rel_i / log2(i + 1))
                dcg += 1.0 / np.log2(i + 1)
        
        # Ideal DCG: all relevant docs ranked first
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _evaluate_generation(self, test_set: List[Dict], rag_responses: List[Dict], config: Dict) -> Dict:
        """Evaluate generation quality"""
        metrics = {
            'faithfulness': 0.0,
            'answer_relevance': 0.0,
            'hallucination_rate': 0.0,
            'correctness': 0.0
        }
        
        use_llm_judge = config.get('use_llm_judge', False)
        
        faithfulness_scores = []
        relevance_scores = []
        hallucination_flags = []
        correctness_scores = []
        
        for test_case, response in zip(test_set, rag_responses):
            generated_answer = response.get('generated_answer', '')
            retrieved_context = ' '.join(response.get('retrieved_docs', []))
            expected_answer = test_case.get('expected_answer')
            
            if not generated_answer:
                continue
            
            # Faithfulness: Is answer grounded in context?
            if use_llm_judge:
                faithfulness = self._llm_judge_faithfulness(generated_answer, retrieved_context, config)
            else:
                # Simple heuristic: check if answer words appear in context
                faithfulness = self._heuristic_faithfulness(generated_answer, retrieved_context)
            faithfulness_scores.append(faithfulness)
            
            # Answer Relevance: Does it address the query?
            if use_llm_judge:
                relevance = self._llm_judge_relevance(generated_answer, test_case['query'], config)
            else:
                # Simple heuristic: keyword overlap with query
                relevance = self._heuristic_relevance(generated_answer, test_case['query'])
            relevance_scores.append(relevance)
            
            # Hallucination: Unsupported claims
            has_hallucination = self._detect_hallucination(generated_answer, retrieved_context)
            hallucination_flags.append(1 if has_hallucination else 0)
            
            # Correctness: Compare to expected answer (if available)
            if expected_answer:
                correctness = self._calculate_correctness(generated_answer, expected_answer)
                correctness_scores.append(correctness)
        
        # Aggregate
        metrics['faithfulness'] = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0
        metrics['answer_relevance'] = float(np.mean(relevance_scores)) if relevance_scores else 0.0
        metrics['hallucination_rate'] = float(np.mean(hallucination_flags)) if hallucination_flags else 0.0
        metrics['correctness'] = float(np.mean(correctness_scores)) if correctness_scores else None
        
        return metrics
    
    def _heuristic_faithfulness(self, answer: str, context: str) -> float:
        """Simple faithfulness check: word overlap"""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if not answer_words:
            return 0.0
        
        # What fraction of answer words appear in context?
        overlap = len(answer_words & context_words)
        return overlap / len(answer_words)
    
    def _heuristic_relevance(self, answer: str, query: str) -> float:
        """Simple relevance check: query keyword presence"""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & answer_words)
        return overlap / len(query_words)
    
    def _detect_hallucination(self, answer: str, context: str) -> bool:
        """Detect potential hallucinations (claims not in context)"""
        # Simplified: if answer significantly longer than context and low overlap
        answer_len = len(answer.split())
        context_len = len(context.split())
        
        if answer_len > context_len * 0.5:  # Answer is substantial compared to context
            faithfulness = self._heuristic_faithfulness(answer, context)
            if faithfulness < 0.3:  # Low overlap suggests hallucination
                return True
        
        return False
    
    def _calculate_correctness(self, generated: str, expected: str) -> float:
        """Calculate answer correctness (simple word overlap)"""
        generated_words = set(generated.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        # F1-style metric
        overlap = len(generated_words & expected_words)
        precision = overlap / len(generated_words) if generated_words else 0
        recall = overlap / len(expected_words) if expected_words else 0
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def _llm_judge_faithfulness(self, answer: str, context: str, config: Dict) -> float:
        """Use LLM to judge faithfulness (placeholder for actual LLM call)"""
        # TODO: Implement actual LLM API call to Gemma service
        logger.warning("LLM judge not yet implemented, using heuristic")
        return self._heuristic_faithfulness(answer, context)
    
    def _llm_judge_relevance(self, answer: str, query: str, config: Dict) -> float:
        """Use LLM to judge relevance (placeholder)"""
        # TODO: Implement actual LLM API call
        logger.warning("LLM judge not yet implemented, using heuristic")
        return self._heuristic_relevance(answer, query)
    
    def _evaluate_end_to_end(self, test_set: List[Dict], rag_responses: List[Dict]) -> Dict:
        """End-to-end system metrics"""
        metrics = {
            'average_latency_ms': 0.0,
            'helpfulness_score': 0.0,
            'conciseness_score': 0.0
        }
        
        latencies = []
        answer_lengths = []
        
        for response in rag_responses:
            if 'latency_ms' in response:
                latencies.append(response['latency_ms'])
            
            answer = response.get('generated_answer', '')
            answer_lengths.append(len(answer.split()))
        
        metrics['average_latency_ms'] = float(np.mean(latencies)) if latencies else 0.0
        
        # Conciseness: prefer shorter answers (normalized)
        if answer_lengths:
            avg_length = np.mean(answer_lengths)
            # Score: 1.0 for 50-word answers, decreases for longer
            metrics['conciseness_score'] = min(1.0, 50 / avg_length) if avg_length > 0 else 0.0
        
        return metrics
    
    def _generate_eval_visualizations(self, results: Dict) -> List[Dict]:
        """Generate visualization metadata for eval results"""
        visualizations = []
        
        # Precision-Recall curve
        if 'retrieval_metrics' in results:
            retrieval = results['retrieval_metrics']
            
            # Bar chart for Precision@k values
            p_at_k = retrieval.get('precision_at_k', {})
            if p_at_k:
                visualizations.append({
                    'type': 'bar_chart',
                    'title': 'Precision@k',
                    'data': p_at_k,
                    'description': 'Precision at different k values'
                })
            
            # Recall@k
            r_at_k = retrieval.get('recall_at_k', {})
            if r_at_k:
                visualizations.append({
                    'type': 'bar_chart',
                    'title': 'Recall@k',
                    'data': r_at_k,
                    'description': 'Recall at different k values'
                })
            
            # MRR gauge
            mrr = retrieval.get('mrr', 0)
            visualizations.append({
                'type': 'gauge_chart',
                'title': 'Mean Reciprocal Rank (MRR)',
                'data': {'value': mrr, 'max': 1.0},
                'description': 'Higher is better (1.0 = perfect ranking)'
            })
        
        # Generation metrics radar chart
        if 'generation_metrics' in results:
            gen_metrics = results['generation_metrics']
            visualizations.append({
                'type': 'radar_chart',
                'title': 'Generation Quality',
                'data': {
                    'Faithfulness': gen_metrics.get('faithfulness', 0) * 100,
                    'Relevance': gen_metrics.get('answer_relevance', 0) * 100,
                    'No Hallucination': (1 - gen_metrics.get('hallucination_rate', 0)) * 100
                },
                'description': 'Generation metrics (higher is better)'
            })
        
        return visualizations
    
    def _generate_eval_summary(self, results: Dict) -> Dict:
        """Generate executive summary of evaluation"""
        summary = {
            'overall_grade': 'N/A',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Extract key metrics
        retrieval = results.get('retrieval_metrics', {})
        generation = results.get('generation_metrics', {})
        
        mrr = retrieval.get('mrr', 0)
        p_at_5 = retrieval.get('precision_at_k', {}).get('p@5', 0)
        faithfulness = generation.get('faithfulness', 0)
        hallucination_rate = generation.get('hallucination_rate', 0)
        
        # Grade the system
        if mrr > 0.8 and faithfulness > 0.8 and hallucination_rate < 0.1:
            summary['overall_grade'] = 'A (Excellent)'
        elif mrr > 0.6 and faithfulness > 0.6 and hallucination_rate < 0.2:
            summary['overall_grade'] = 'B (Good)'
        elif mrr > 0.4 and faithfulness > 0.4:
            summary['overall_grade'] = 'C (Fair)'
        else:
            summary['overall_grade'] = 'D (Needs Improvement)'
        
        # Identify strengths
        if mrr > 0.7:
            summary['strengths'].append('Strong retrieval quality (MRR > 0.7)')
        if faithfulness > 0.7:
            summary['strengths'].append('High faithfulness to context')
        if hallucination_rate < 0.1:
            summary['strengths'].append('Low hallucination rate')
        
        # Identify weaknesses
        if mrr < 0.5:
            summary['weaknesses'].append('Poor retrieval ranking (MRR < 0.5)')
            summary['recommendations'].append('Improve embedding model or rerank retrieved documents')
        if faithfulness < 0.5:
            summary['weaknesses'].append('Low faithfulness to context')
            summary['recommendations'].append('Tune generation prompt to emphasize grounding')
        if hallucination_rate > 0.2:
            summary['weaknesses'].append('High hallucination rate')
            summary['recommendations'].append('Add hallucination detection filter or reduce temperature')
        
        return summary
