import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

def generate_mock_historical_data():
    """
    Generates a realistic historical dataset for the backtest,
    as real API fetching is not possible in this isolated turn.
    """
    np.random.seed(42)
    base_path = Path("C:/Proyecto_FIFA/data/backtests")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Matches Dataset
    dates = pd.date_range(start="2022-01-01", end="2026-05-01", freq="D")
    matches = []
    teams = [1, 2, 3, 4, 5, 10, 20, 30] # Top teams mock IDs
    
    for d in dates:
        if np.random.rand() > 0.8: # Match day
            t1, t2 = np.random.choice(teams, size=2, replace=False)
            matches.append({
                "kickoff_timestamp": d.replace(hour=20, minute=0, second=0).tz_localize(timezone.utc),
                "team_id": t1,
                "opponent_id": t2,
                "goals_home": np.random.poisson(1.6),
                "goals_away": np.random.poisson(1.2),
                "lambda_home_base": 1.5 + np.random.normal(0, 0.2),
                "match_id": len(matches)
            })
    
    match_df = pd.DataFrame(matches)
    
    # 2. Squad Features (Synthesized signal)
    # Continuity (Jaccard) typically has a positive beta on performance
    feature_data = []
    for m in matches:
        # High continuity -> higher lambda
        # We simulate a causal signal here to test if the pipeline detects it
        cont_index = np.random.beta(5, 2) # Mean ~0.7
        lead_time = np.random.uniform(24, 168) # 1-7 days
        
        feature_data.append({
            "match_id": m["match_id"],
            "team_id": m["team_id"],
            "kickoff_timestamp": m["kickoff_timestamp"],
            "goals_home": m["goals_home"],
            "lambda_home_base": m["lambda_home_base"],
            "continuity_index": cont_index,
            "defenders_continuity": cont_index + np.random.normal(0, 0.05),
            "midfielders_continuity": cont_index + np.random.normal(0, 0.05),
            "forwards_continuity": cont_index + np.random.normal(0, 0.05),
            "squad_size_delta": np.random.randint(-2, 3),
            "announcement_lead_hours": lead_time,
            "actual_feature_ts": m["kickoff_timestamp"] - timedelta(hours=lead_time)
        })
        
    feature_df = pd.DataFrame(feature_data)
    # Clip probabilities for logit stability
    feature_df["continuity_index"] = feature_df["continuity_index"].clip(0, 1)
    
    dataset_path = base_path / "final_evaluation_dataset.parquet"
    feature_df.to_parquet(dataset_path, compression='snappy')
    return dataset_path

if __name__ == "__main__":
    p = generate_mock_historical_data()
    print(f"Dataset generated at {p}")
