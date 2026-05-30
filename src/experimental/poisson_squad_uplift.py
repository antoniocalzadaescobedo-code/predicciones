import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import statsmodels.api as sm
import logging

logger = logging.getLogger("poisson_uplift")

class PoissonSquadUplift:
    """
    Parametric adjustment engine for Poisson expectations based on squad features.
    Uses a log-linear model: lambda_adj = lambda_base * exp(beta * X)
    This ensures lambda remains positive and follows the Poisson link function.
    """
    
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.betas: Optional[np.ndarray] = None
        self.model = None

    def fit(self, df: pd.DataFrame, target_col: str, base_lambda_col: str):
        """
        Learns the sensitivity (beta) of goal scoring to squad features.
        We use base_lambda as an 'offset' in the Poisson GLM.
        ln(target) = ln(base_lambda) + beta * X
        """
        X = df[self.feature_columns]
        X = sm.add_constant(X)
        
        # log(base_lambda) acts as the baseline expectation
        offset = np.log(df[base_lambda_col])
        
        try:
            self.model = sm.GLM(df[target_col], X, 
                                family=sm.families.Poisson(),
                                offset=offset)
            res = self.model.fit()
            self.betas = res.params
            logger.info(f"Poisson Uplift fit successful. Beta coefficients:\n{res.summary().tables[1]}")
        except Exception as e:
            logger.error(f"Failed to fit Poisson Uplift: {e}")
            raise

    def predict_adjusted_lambda(self, df: pd.DataFrame, base_lambda_col: str) -> pd.Series:
        """
        Calculates the adjusted expectation: lambda_adj = lambda_base * exp(X * beta)
        """
        if self.betas is None:
            raise ValueError("Model must be fit before prediction.")
            
        X = df[self.feature_columns]
        X = sm.add_constant(X, has_constant='add')
        
        # Ensure alignment with fitted features
        uplift_term = np.dot(X, self.betas)
        return df[base_lambda_col] * np.exp(uplift_term)

if __name__ == "__main__":
    logger.info("PoissonSquadUplift module ready for experimental use.")
