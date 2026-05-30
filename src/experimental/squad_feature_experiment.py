import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.experimental.poisson_squad_uplift import PoissonSquadUplift
from src.experimental.temporal_cross_validation import PurgedTemporalCV
from src.experimental.uplift_evaluation import UpliftEvaluationSuite
from src.experimental.feature_ablation import FeatureAblationStudy
import logging

logger = logging.getLogger("experimental_orchestrator")

class SquadFeatureExperiment:
    """
    Main orchestrator for Phase E.
    Runs the full experimental suite: CV -> Ablation -> Calibration -> Report.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = pd.read_parquet(data_path)
        self.df['kickoff_timestamp'] = pd.to_datetime(self.df['kickoff_timestamp'])

    def run(self):
        logger.info("Starting Squad Feature Experiment...")
        
        # 1. Setup Cross-Validation
        cv = PurgedTemporalCV(n_splits=5, embargo_days=15)
        
        # 2. Define Feature Groups for Ablation
        groups = {
            "continuity_only": ["continuity_index"],
            "positional_stability": ["defenders_continuity", "midfielders_continuity", "forwards_continuity"],
            "all_features": [
                "continuity_index", "defenders_continuity", 
                "midfielders_continuity", "forwards_continuity",
                "squad_size_delta", "announcement_lead_hours"
            ]
        }
        
        # 3. Run Ablation Study
        ablation = FeatureAblationStudy(baseline_model=None, cv=cv)
        ablation.run_study(self.df, groups)
        
        # 4. Final Assessment
        self._generate_experimental_report(ablation.results)

    def _generate_experimental_report(self, results: dict):
        logger.info("--- EXPERIMENTAL REPORT ---")
        for group, uplift in results.items():
            status = "PASS" if uplift >= 0.005 else "REJECT"
            logger.info(f"Group {group}: Uplift={uplift:.4f} -> [{status}]")
        logger.info("----------------------------")

if __name__ == "__main__":
    # Placeholder for execution
    pass
