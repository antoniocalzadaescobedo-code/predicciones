import numpy as np
import pandas as pd
from typing import Dict, Any, Callable
from scipy.stats import brier_score_loss
import logging

logger = logging.getLogger("uplift_eval")

class UpliftEvaluationSuite:
    """
    Diagnostic suite for measuring the impact of squad features.
    Calculates Brier Score, LogLoss, and Calibration improvements.
    """
    
    @staticmethod
    def calculate_brier_improvement(y_true: np.ndarray, base_probs: np.ndarray, adj_probs: np.ndarray) -> float:
        """
        Calculates improvement in Brier Score. Positive value means model is better.
        """
        # Note: brier_score_loss expects 1D binary indicators
        base_brier = np.mean((base_probs - y_true)**2)
        adj_brier = np.mean((adj_probs - y_true)**2)
        return base_brier - adj_brier

    @staticmethod
    def block_bootstrap_confidence_interval(
        df: pd.DataFrame, 
        metric_fn: Callable[[pd.DataFrame], float],
        block_col: str = 'fifa_window',
        n_iterations: int = 1000
    ) -> Tuple[float, float]:
        """
        Block bootstrap to estimate uncertainty under temporal autocorrelation.
        """
        blocks = df[block_col].unique()
        boot_metrics = []
        
        for _ in range(n_iterations):
            resampled_blocks = np.random.choice(blocks, size=len(blocks), replace=True)
            boot_sample = pd.concat([df[df[block_col] == b] for b in resampled_blocks])
            boot_metrics.append(metric_fn(boot_sample))
            
        return np.percentile(boot_metrics, [2.5, 97.5])

    @staticmethod
    def calculate_calibration_slope(y_true: np.ndarray, probs: np.ndarray) -> float:
        """
        Measures if the model is overconfident (>1) or underconfident (<1).
        Ideal is 1.0.
        """
        # Simple logistic regression of outcome on predicted logit
        epsilon = 1e-15
        logits = np.log(probs / (1 - probs + epsilon))
        
        # Add constant and fit
        X = sm.add_constant(logits)
        model = sm.Logit(y_true, X).fit(disp=0)
        return model.params[1]

if __name__ == "__main__":
    logger.info("UpliftEvaluationSuite ready.")
