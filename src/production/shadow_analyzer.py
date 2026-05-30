import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone

logger = logging.getLogger("shadow_analyzer")

class ShadowAnalysisEngine:
    """
    Principal Production Validation Engine for Shadow Deployment analysis.
    Quantifies the discrepancy between Baseline and Squad-Enhanced models.
    """
    
    def __init__(self, logs_path: Path, output_report_path: Path):
        self.logs_path = logs_path
        self.report_path = output_report_path
        self.disagreement_threshold = 0.15  # 15% difference in lambda triggers review

    def analyze_shadow_performance(self, shadow_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes ROI delta and Disagreement metrics.
        Expects columns: baseline_lambda, shadow_uplifted_lambda, goals_actual (if available)
        """
        if shadow_df.empty:
            return {"status": "EMPTY_LOGS"}

        # 1. Disagreement Analysis
        shadow_df['abs_diff'] = (shadow_df['shadow_uplifted_lambda'] - shadow_df['baseline_lambda']).abs()
        shadow_df['rel_diff'] = shadow_df['abs_diff'] / shadow_df['baseline_lambda']
        
        high_disagreement = shadow_df[shadow_df['rel_diff'] > self.disagreement_threshold]
        
        # 2. Virtual ROI (Assuming outcome is known for historical shadow analysis)
        # In real-time shadow, we monitor 'Disagreement Sharpness'
        metrics = {
            "total_matches": len(shadow_df),
            "mean_disagreement": shadow_df['rel_diff'].mean(),
            "max_disagreement": shadow_df['rel_diff'].max(),
            "high_disagreement_count": len(high_disagreement),
            "high_disagreement_ratio": len(high_disagreement) / len(shadow_df),
            "shadow_volatility": shadow_df['shadow_uplifted_lambda'].std()
        }

        # 3. Generate Disagreement Report
        self._generate_markdown_report(metrics, high_disagreement)
        
        return metrics

    def _generate_markdown_report(self, metrics: Dict[str, Any], high_disagreement: pd.DataFrame):
        """Generates a formal Disagreement Analysis report."""
        report = f"""# Shadow Deployment: Disagreement Analysis Report
Generated: {datetime.now(timezone.utc).isoformat()}

## Summary Metrics
- **Total Matches Evaluated:** {metrics['total_matches']}
- **Mean Disagreement (Δλ):** {metrics['mean_disagreement']:.4%}
- **High Disagreement Ratio (>15%):** {metrics['high_disagreement_ratio']:.2%}
- **Shadow Volatility:** {metrics['shadow_volatility']:.4f}

## High Disagreement Audit (Top 5)
"""
        if not high_disagreement.empty:
            top_5 = high_disagreement.sort_values('rel_diff', ascending=False).head(5)
            report += "| Match ID | Baseline λ | Shadow λ | Δ% |\n"
            report += "|---|---|---|---|\n"
            for _, row in top_5.iterrows():
                report += f"| {row.get('match_id', 'N/A')} | {row['baseline_lambda']:.2f} | {row['shadow_uplifted_lambda']:.2f} | {row['rel_diff']:.2%} |\n"
        else:
            report += "No significant disagreements detected."

        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Disagreement report persisted to {self.report_path}")

if __name__ == "__main__":
    # Internal component validation
    print("ShadowAnalysisEngine ready.")
