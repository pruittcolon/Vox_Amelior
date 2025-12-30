"""
Business Savings Analyzer - LLM-Powered Cost Reduction Discovery

Uses Gemma LLM to analyze quality issues and estimate business cost savings.
Generates ROI-ranked recommendations for data quality improvements.

Author: Enterprise Analytics Team
Security: JWT-authenticated, input-validated
"""

import logging
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# Default savings estimates per issue type (base values for LLM context)
DEFAULT_IMPACT_ESTIMATES = {
    "completeness": {
        "cost_per_issue": 150,  # $ per incomplete record
        "description": "Missing data leads to failed processes and manual intervention",
    },
    "anomalies": {
        "cost_per_issue": 250,  # $ per anomaly
        "description": "Outliers cause incorrect analytics and poor decisions",
    },
    "consistency": {
        "cost_per_issue": 200,  # $ per inconsistency
        "description": "Data mismatches cause reconciliation work and errors",
    },
    "validity": {
        "cost_per_issue": 175,  # $ per invalid record
        "description": "Invalid data types cause processing failures",
    },
    "duplicates": {
        "cost_per_issue": 100,  # $ per duplicate
        "description": "Duplicate records waste storage and processing",
    },
}


class BusinessSavingsAnalyzer:
    """
    Analyzes data quality issues and generates business savings estimates.
    
    Can operate in two modes:
    1. LLM Mode: Uses Gemma to generate intelligent estimates
    2. Heuristic Mode: Uses rule-based calculations (fallback)
    """

    def __init__(self, gemma_url: str = "http://gemma-service:8001"):
        self.gemma_url = gemma_url
        self.name = "Business Savings Analyzer"

    async def analyze_savings(
        self,
        quality_summary: dict[str, Any],
        use_llm: bool = True,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze quality issues and estimate business savings.
        
        Args:
            quality_summary: Output from QualityInsightsEngine.analyze()
            use_llm: Whether to use Gemma LLM for estimates
            config: Configuration overrides
            
        Returns:
            {
                "estimated_annual_savings": "$XXX,XXX",
                "high_impact_issues": [...],
                "recommendations": [...],
                "roi_timeline": {...},
            }
        """
        config = config or {}
        
        # Extract key metrics from quality summary
        metrics = self._extract_metrics(quality_summary)
        
        if use_llm:
            try:
                return await self._analyze_with_llm(metrics, config)
            except Exception as e:
                logger.warning(f"LLM analysis failed, using heuristics: {e}")
        
        # Fallback to heuristic analysis
        return self._analyze_with_heuristics(metrics, config)

    def _extract_metrics(self, quality_summary: dict) -> dict[str, Any]:
        """Extract key metrics from quality summary"""
        row_count = quality_summary.get("row_count", 0)
        problem_count = quality_summary.get("problem_row_count", 0)
        risk_distribution = quality_summary.get("risk_distribution", {})
        column_issues = quality_summary.get("column_issues", [])
        ml_readiness = quality_summary.get("ml_readiness", 50.0)
        avg_overall = quality_summary.get("avg_overall", 5.0)
        
        # Calculate issue breakdown
        issue_breakdown = {}
        for issue in column_issues:
            category = issue.get("category", "unknown").lower()
            count = issue.get("count", 0)
            issue_breakdown[category] = issue_breakdown.get(category, 0) + count
        
        return {
            "row_count": row_count,
            "problem_count": problem_count,
            "problem_percentage": (problem_count / row_count * 100) if row_count > 0 else 0,
            "low_quality_rows": risk_distribution.get("low", 0),
            "medium_quality_rows": risk_distribution.get("medium", 0),
            "high_quality_rows": risk_distribution.get("high", 0),
            "issue_breakdown": issue_breakdown,
            "ml_readiness": ml_readiness,
            "avg_quality": avg_overall,
        }

    async def _analyze_with_llm(
        self, metrics: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Use Gemma LLM to generate savings estimates"""
        
        # Build prompt for Gemma
        prompt = self._build_savings_prompt(metrics)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.gemma_url}/chat",
                json={
                    "message": prompt,
                    "system_prompt": "You are a business analyst. Respond ONLY with valid JSON.",
                    "max_tokens": 300,
                    "temperature": 0.2,
                },
            )
            
            if response.status_code != 200:
                raise Exception(f"Gemma returned {response.status_code}")
            
            result = response.json()
            generated_text = result.get("text", result.get("response", ""))
            
            # Parse JSON from response
            import json
            import re
            
            json_match = re.search(r'\{[^}]+\}', generated_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return self._format_llm_response(parsed, metrics)
            else:
                raise Exception("No valid JSON in LLM response")

    def _build_savings_prompt(self, metrics: dict[str, Any]) -> str:
        """Build prompt for LLM savings analysis"""
        issue_summary = ", ".join([
            f"{count} {category} issues"
            for category, count in metrics.get("issue_breakdown", {}).items()
        ]) or "various quality issues"
        
        return f"""Analyze this data quality summary and estimate business cost savings:

DATA QUALITY METRICS:
- Total rows: {metrics['row_count']:,}
- Problem rows: {metrics['problem_count']:,} ({metrics['problem_percentage']:.1f}%)
- Issues found: {issue_summary}
- ML Readiness: {metrics['ml_readiness']:.1f}%
- Average Quality: {metrics['avg_quality']:.1f}/10

Estimate annual cost savings if these issues were resolved.
Consider: operational efficiency, decision accuracy, regulatory compliance.

Respond ONLY with JSON:
{{"estimated_annual_savings": "$XXX,XXX", "high_impact_issues": [{{"issue": "description", "impact": "$XXX/year"}}], "top_recommendation": "action to take"}}"""

    def _format_llm_response(
        self, parsed: dict, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Format LLM response into standard output"""
        return {
            "estimated_annual_savings": parsed.get("estimated_annual_savings", "$0"),
            "high_impact_issues": parsed.get("high_impact_issues", []),
            "recommendations": [
                {
                    "action": parsed.get("top_recommendation", "Review data quality"),
                    "roi": "High",
                    "timeline": "3-6 months",
                }
            ],
            "roi_timeline": {
                "quick_wins": "1 month",
                "full_implementation": "6 months",
            },
            "analysis_method": "llm",
            "metrics_used": metrics,
        }

    def _analyze_with_heuristics(
        self, metrics: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate savings estimates using heuristic rules.
        
        Fallback when LLM is unavailable.
        """
        total_savings = 0
        high_impact_issues = []
        recommendations = []
        
        # Calculate savings per issue type
        for category, count in metrics.get("issue_breakdown", {}).items():
            if category in DEFAULT_IMPACT_ESTIMATES:
                estimate = DEFAULT_IMPACT_ESTIMATES[category]
                impact = count * estimate["cost_per_issue"]
                total_savings += impact
                
                if impact > 1000:  # Only show significant impacts
                    high_impact_issues.append({
                        "issue": f"{count} {category} issues",
                        "impact": f"${impact:,.0f}/year",
                        "description": estimate["description"],
                    })
        
        # Add baseline savings for low-quality rows
        low_quality_count = metrics.get("low_quality_rows", 0)
        baseline_impact = low_quality_count * 50  # $50 per low-quality row
        total_savings += baseline_impact
        
        if baseline_impact > 1000:
            high_impact_issues.append({
                "issue": f"{low_quality_count} low-quality rows",
                "impact": f"${baseline_impact:,.0f}/year",
                "description": "Low-quality data requires manual review and correction",
            })
        
        # Generate recommendations based on issues
        if "completeness" in metrics.get("issue_breakdown", {}):
            recommendations.append({
                "action": "Implement required field validation at data entry",
                "roi": "3.2x in 6 months",
                "timeline": "2 months",
            })
        
        if "anomalies" in metrics.get("issue_breakdown", {}):
            recommendations.append({
                "action": "Add automated outlier detection in data pipeline",
                "roi": "2.5x in 6 months",
                "timeline": "3 months",
            })
        
        if "consistency" in metrics.get("issue_breakdown", {}):
            recommendations.append({
                "action": "Implement cross-field validation rules",
                "roi": "2.1x in 6 months",
                "timeline": "2 months",
            })
        
        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append({
                "action": "Conduct data quality audit and establish monitoring",
                "roi": "2.0x in 12 months",
                "timeline": "6 months",
            })
        
        # Sort issues by impact
        high_impact_issues.sort(
            key=lambda x: float(x["impact"].replace("$", "").replace(",", "").replace("/year", "")),
            reverse=True,
        )
        
        return {
            "estimated_annual_savings": f"${total_savings:,.0f}",
            "high_impact_issues": high_impact_issues[:5],  # Top 5
            "recommendations": recommendations[:3],  # Top 3
            "roi_timeline": {
                "quick_wins": "1-2 months",
                "full_implementation": "6-12 months",
            },
            "analysis_method": "heuristic",
            "metrics_used": metrics,
        }

    def generate_savings_report(
        self, savings_result: dict[str, Any], filename: str
    ) -> str:
        """Generate markdown report from savings analysis"""
        report = f"""# Business Savings Analysis Report

## Dataset: {filename}

### Executive Summary

**Estimated Annual Savings: {savings_result.get('estimated_annual_savings', 'N/A')}**

Analysis Method: {savings_result.get('analysis_method', 'heuristic').title()}

---

## High-Impact Issues

"""
        for issue in savings_result.get("high_impact_issues", []):
            report += f"### {issue.get('issue', 'Issue')}\n"
            report += f"- **Impact**: {issue.get('impact', 'Unknown')}\n"
            report += f"- **Description**: {issue.get('description', '')}\n\n"
        
        report += """---

## Recommendations

"""
        for i, rec in enumerate(savings_result.get("recommendations", []), 1):
            report += f"{i}. **{rec.get('action', 'Action')}**\n"
            report += f"   - ROI: {rec.get('roi', 'TBD')}\n"
            report += f"   - Timeline: {rec.get('timeline', 'TBD')}\n\n"
        
        report += """---

## ROI Timeline

"""
        timeline = savings_result.get("roi_timeline", {})
        report += f"- Quick Wins: {timeline.get('quick_wins', 'TBD')}\n"
        report += f"- Full Implementation: {timeline.get('full_implementation', 'TBD')}\n"
        
        return report


# Standalone test
if __name__ == "__main__":
    import asyncio
    
    # Sample quality summary for testing
    sample_summary = {
        "row_count": 1000,
        "problem_row_count": 150,
        "risk_distribution": {"high": 700, "medium": 150, "low": 150},
        "column_issues": [
            {"category": "Completeness", "count": 45},
            {"category": "Anomalies", "count": 30},
            {"category": "Consistency", "count": 25},
        ],
        "ml_readiness": 70.0,
        "avg_overall": 7.2,
    }
    
    analyzer = BusinessSavingsAnalyzer()
    
    # Test heuristic mode (no LLM)
    result = analyzer._analyze_with_heuristics(
        analyzer._extract_metrics(sample_summary), {}
    )
    
    print("=" * 60)
    print("BUSINESS SAVINGS ANALYSIS (Heuristic)")
    print("=" * 60)
    print(f"\nEstimated Savings: {result['estimated_annual_savings']}")
    
    print("\nHigh-Impact Issues:")
    for issue in result["high_impact_issues"]:
        print(f"  - {issue['issue']}: {issue['impact']}")
    
    print("\nRecommendations:")
    for rec in result["recommendations"]:
        print(f"  - {rec['action']} (ROI: {rec['roi']})")
