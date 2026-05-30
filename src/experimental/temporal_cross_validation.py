import pandas as pd
from datetime import timedelta
from typing import List, Tuple, Generator
import logging

logger = logging.getLogger("temporal_cv")

class PurgedTemporalCV:
    """
    Implements Purged Time-Series Cross-Validation for international football.
    Enforces a 15-day embargo between training and validation sets to prevent 
    leakage across FIFA match windows.
    """
    
    def __init__(self, n_splits: int = 5, embargo_days: int = 15):
        self.n_splits = n_splits
        self.embargo_days = timedelta(days=embargo_days)

    def split(self, df: pd.DataFrame, timestamp_col: str) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generates expanding window splits with an embargo gap.
        """
        df = df.sort_values(timestamp_col)
        total_days = (df[timestamp_col].max() - df[timestamp_col].min()).days
        split_size = total_days // (self.n_splits + 1)
        
        start_ts = df[timestamp_col].min()
        
        for i in range(1, self.n_splits + 1):
            train_end_ts = start_ts + timedelta(days=i * split_size)
            val_start_ts = train_end_ts + self.embargo_days
            val_end_ts = val_start_ts + timedelta(days=split_size)
            
            train_idx = df[df[timestamp_col] < train_end_ts].index
            val_idx = df[(df[timestamp_col] >= val_start_ts) & (df[timestamp_col] < val_end_ts)].index
            
            if len(val_idx) == 0:
                logger.warning(f"Fold {i} generated an empty validation set. Skipping.")
                continue
                
            yield df.loc[train_idx], df.loc[val_idx]

if __name__ == "__main__":
    logger.info("PurgedTemporalCV module ready.")
