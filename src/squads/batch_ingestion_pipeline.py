import logging
from pathlib import Path
from datetime import datetime, timezone
from src.squads.squad_persistence import SquadPersistence
from src.squads.temporal_feature_store import TemporalFeatureStore
from src.squads.temporal_features import SquadTemporalExtractor
from src.squads.historical_squad_fetcher import HistoricalSquadFetcher
from src.squads.feature_batch_processor import FeatureBatchProcessor
from src.squads.feature_dataset_builder import FeatureDatasetBuilder
from src.squads.batch_validation import BatchValidation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion_pipeline")

class BatchIngestionPipeline:
    """
    Central orchestrator for Phase D. 
    Manages the flow from raw API data to ML-ready datasets.
    """
    
    def __init__(self, base_path: Path, api_key: str):
        self.base_path = base_path
        self.db_path = base_path / "squads.db"
        
        # Initialize components
        self.persistence = SquadPersistence(self.db_path)
        self.store = TemporalFeatureStore(base_path)
        self.extractor = SquadTemporalExtractor()
        self.fetcher = HistoricalSquadFetcher(api_key)
        self.processor = FeatureBatchProcessor(self.persistence, self.extractor)
        self.builder = FeatureDatasetBuilder(self.store)

    def run_full_batch_pipeline(self, team_ids: list, matches_path: Path, output_dataset_path: Path):
        """
        Executes end-to-end batch processing.
        """
        logger.info("--- STARTING BATCH PIPELINE ---")
        
        # 1. Ingestion Phase
        for tid in team_ids:
            raw_squads = self.fetcher.fetch_team_snapshots(tid)
            for raw in raw_squads:
                if self.fetcher.validate_snapshot_integrity(raw):
                    snap = self.fetcher.transform_to_snapshot(raw, tid, "Team", datetime.now(timezone.utc))
                    self.persistence.save_snapshot(snap)

        # 2. Processing Phase (Snapshots -> Features)
        all_feature_records = []
        for tid in team_ids:
            recs = self.processor.process_team_feature_history(tid)
            all_feature_records.extend(recs)
        
        self.store.save_feature_records(all_feature_records)

        # 3. Validation Phase
        validator = BatchValidation(self.store.conn)
        metrics = validator.calculate_metrics()
        leakage = validator.detect_leakage_violations("feature_view")
        
        if leakage > 0:
            logger.critical(f"FATAL: {leakage} leakage violations detected during batch processing.")
            return

        # 4. Dataset Generation Phase
        if matches_path.exists():
            features_to_include = ["continuity_index", "squad_size"]
            self.builder.build_training_dataset(matches_path, output_dataset_path, features_to_include)
            self.builder.validate_dataset_causality(output_dataset_path)

        logger.info("--- BATCH PIPELINE COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    pass
