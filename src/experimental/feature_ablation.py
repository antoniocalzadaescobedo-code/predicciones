import pandas as pd
from typing import List, Dict
from src.experimental.poisson_squad_uplift import PoissonSquadUplift
from src.experimental.temporal_cross_validation import PurgedTemporalCV
from src.experimental.uplift_evaluation import UpliftEvaluationSuite
import logging

logger = logging.getLogger("ablation_study")

class FeatureAblationStudy:
    """
    Evaluates features individually and in groups to identify signal vs noise.
    """
    
    def __init__(self, baseline_model: Any, cv: PurgedTemporalCV):
        self.baseline = baseline_model
        self.cv = cv
        self.results = {}

    def run_study(self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]):
        """
        Runs CV for each feature group and compares against baseline.
        """
        for group_name, features in feature_groups.items():
            logger.info(f"Evaluating group: {group_name}")
            group_uplifts = []
            
            for train_df, val_df in self.cv.split(df, 'kickoff_timestamp'):
                # 1. Fit uplift on train
                uplift_model = PoissonSquadUplift(features)
                # target: goals_home, base_lambda: lambda_home_base
                uplift_model.fit(train_df, 'goals_home', 'lambda_home_base')
                
                # 2. Predict on val
                adj_lambda = uplift_model.predict_adjusted_lambda(val_df, 'lambda_home_base')
                
                # 3. Calculate metrics
                # (Simplified for demonstration)
                # In real scenario we'd use full prob distributions
                improvement = self._quick_metric(val_df['goals_home'], val_df['lambda_home_base'], adj_lambda)
                group_uplifts.append(improvement)
                
            self.results[group_name] = np.mean(group_uplifts)
            logger.info(f"Mean improvement for {group_name}: {self.results[group_name]}")

    def _quick_metric(self, y_true, base_lambda, adj_lambda):
        # Surrogate for Brier: Mean Squared Error improvement on expectations
        base_mse = np.mean((y_true - base_lambda)**2)
        adj_mse = np.mean((y_true - adj_lambda)**2)
        return base_mse - adj_mse

if __name__ == "__main__":
    logger.info("FeatureAblationStudy module ready.")
