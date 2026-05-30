import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from src.experimental.historical_backtest_runner import HistoricalBacktestRunner

def test_backtest_anti_leakage_enforcement(tmp_path):
    """
    Ensures that the backtest runner strictly respects chronological order.
    """
    # Create mock dataset
    data = []
    base_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(100):
        ts = base_ts + timedelta(days=i)
        data.append({
            "kickoff_timestamp": ts,
            "goals_home": np.random.poisson(1.5),
            "lambda_home_base": 1.5,
            "continuity_index": 0.8,
            "team_id": 1
        })
    
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test_data.parquet"
    df.to_parquet(parquet_path)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    runner = HistoricalBacktestRunner(parquet_path, output_dir)
    results = runner.run_backtest(features=["continuity_index"])
    
    # Verify chronological integrity across folds
    for i in range(len(results) - 1):
        assert results[i].train_end < results[i+1].val_start

def test_backtest_output_integrity(tmp_path):
    """
    Checks if the backtest results are correctly persisted in Snappy Parquet.
    """
    parquet_path = tmp_path / "test_data.parquet"
    pd.DataFrame({
        "kickoff_timestamp": [datetime.now(timezone.utc) + timedelta(days=i) for i in range(20)],
        "goals_home": [1] * 20,
        "lambda_home_base": [1.0] * 20,
        "continuity_index": [0.8] * 20,
        "team_id": [1] * 20
    }).to_parquet(parquet_path)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    runner = HistoricalBacktestRunner(parquet_path, output_dir)
    runner.run_backtest(features=["continuity_index"])
    
    log_file = output_dir / "backtest_iterations.parquet"
    assert log_file.exists()
    
    log_df = pd.read_parquet(log_file)
    assert not log_df.empty
    assert "improvement" in log_df.columns
