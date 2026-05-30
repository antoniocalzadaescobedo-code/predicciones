import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from src.experimental.poisson_squad_uplift import PoissonSquadUplift
from src.experimental.temporal_cross_validation import PurgedTemporalCV
from src.experimental.uplift_evaluation import UpliftEvaluationSuite

logger = logging.getLogger("backtest_runner")

@dataclass(frozen=True)
class BacktestIteration:
    fold: int
    train_end: str
    val_start: str
    baseline_brier: float
    uplift_brier: float
    improvement: float
    features_used: List[str]

class HistoricalBacktestRunner:
    """
    Orchestrates a full historical replay of match predictions.
    Compares Baseline (Elo-Poisson) vs Squad-Enhanced Poisson.
    """
    
    def __init__(self, data_path: Path, output_dir: Path):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = pd.read_parquet(data_path)
        self.df['kickoff_timestamp'] = pd.to_datetime(self.df['kickoff_timestamp'])
        self.cv = PurgedTemporalCV(n_splits=5, embargo_days=15)
        self.evaluator = UpliftEvaluationSuite()

    def run_backtest(self, features: List[str]) -> List[BacktestIteration]:
        """
        Executes the backtest using a rolling purged window.
        """
        results = []
        logger.info(f"Starting backtest with features: {features}")
        
        for fold, (train_df, val_df) in enumerate(self.cv.split(self.df, 'kickoff_timestamp')):
            # Ensure strict anti-leakage
            assert train_df['kickoff_timestamp'].max() < val_df['kickoff_timestamp'].min()
            
            # 1. Fit Baseline (using existing lambda columns)
            # 2. Fit Uplift Model
            uplift_model = PoissonSquadUplift(features)
            uplift_model.fit(train_df, 'goals_home', 'lambda_home_base')
            
            # 3. Predict on Validation
            base_lambda = val_df['lambda_home_base']
            adj_lambda = uplift_model.predict_adjusted_lambda(val_df, 'lambda_home_base')
            
            # 4. Evaluate (using expectations as probabilities for simplicity in this module)
            # In production, we'd use full prob distributions for Home/Draw/Away
            y_true = val_df['goals_home']
            
            # Calculate metrics
            base_brier = np.mean((base_lambda - y_true)**2)
            adj_brier = np.mean((adj_lambda - y_true)**2)
            improvement = base_brier - adj_brier
            
            iteration = BacktestIteration(
                fold=fold,
                train_end=train_df['kickoff_timestamp'].max().isoformat(),
                val_start=val_df['kickoff_timestamp'].min().isoformat(),
                baseline_brier=float(base_brier),
                uplift_brier=float(adj_brier),
                improvement=float(improvement),
                features_used=features
            )
            results.append(iteration)
            logger.info(f"Fold {fold}: Improvement = {improvement:.6f}")
            
        self._save_results(results)
        return results

    def _save_results(self, results: List[BacktestIteration]):
        res_df = pd.DataFrame([asdict(r) for r in results])
        res_df.to_parquet(self.output_dir / "backtest_iterations.parquet", compression='snappy')
        logger.info(f"Backtest logs saved to {self.output_dir}")

if __name__ == "__main__":
    # Internal test execution
    pass
