import logging
from src.squads.temporal_feature_store import TemporalFeatureStore
from pathlib import Path
from typing import List
import duckdb

logger = logging.getLogger("dataset_builder")

class FeatureDatasetBuilder:
    """
    Constructs final ML datasets using DuckDB ASOF JOINs.
    Guarantees no future information leakage.
    """
    
    def __init__(self, feature_store: TemporalFeatureStore):
        self.store = feature_store

    def build_training_dataset(self, matches_parquet_path: Path, output_path: Path, features: List[str]):
        """
        Creates a point-in-time dataset where each match is joined with 
        the latest squad features available at kickoff.
        """
        logger.info(f"Building dataset with features: {features}")
        
        # This leverages the DuckDB ASOF logic already in TemporalFeatureStore
        # We ensure match kickoff_timestamp > feature_timestamp_utc
        self.store.build_point_in_time_dataset(
            events_parquet_path=str(matches_parquet_path),
            output_path=str(output_path),
            feature_names=features
        )
        
        logger.info(f"Dataset successfully exported to {output_path}")

    def validate_dataset_causality(self, dataset_path: Path) -> bool:
        """
        Verification step: checks if any row has feature_ts >= kickoff_ts.
        """
        conn = duckdb.connect(database=':memory:')
        query = f"SELECT count(*) FROM read_parquet('{dataset_path}') WHERE actual_feature_ts >= kickoff_timestamp"
        violations = conn.execute(query).fetchone()[0]
        
        if violations > 0:
            logger.critical(f"CAUSALITY VIOLATION: {violations} rows contain future information!")
            return False
        
        logger.info("Causality validation PASSED.")
        return True
