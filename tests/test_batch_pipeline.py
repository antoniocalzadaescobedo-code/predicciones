import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from src.squads.squad_persistence import SquadSnapshot, SquadPersistence
from src.squads.temporal_features import SquadTemporalExtractor
from src.squads.feature_batch_processor import FeatureBatchProcessor
from src.squads.temporal_feature_store import TemporalFeatureStore

def test_deterministic_feature_rebuild(tmp_path):
    """
    Ensures that re-running the processor over the same snapshots 
    results in identical feature values (determinism).
    """
    db_path = tmp_path / "test.db"
    persistence = SquadPersistence(db_path)
    extractor = SquadTemporalExtractor()
    processor = FeatureBatchProcessor(persistence, extractor)
    
    # 1. Setup snapshots
    tid = 1
    ts1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc)
    
    snap1 = SquadSnapshot(tid, "Team A", [{"name": "P1", "position": "Def"}], "Source", ts1, ts1)
    snap2 = SquadSnapshot(tid, "Team A", [{"name": "P1", "position": "Def"}, {"name": "P2", "position": "Def"}], "Source", ts2, ts2)
    
    persistence.save_snapshot(snap1)
    persistence.save_snapshot(snap2)
    
    # 2. Process twice
    run1 = processor.process_team_feature_history(tid)
    run2 = processor.process_team_feature_history(tid)
    
    assert len(run1) == len(run2)
    for r1, r2 in zip(run1, run2):
        assert r1.feature_value == r2.feature_value
        assert r1.feature_name == r2.feature_name

def test_out_of_order_snapshot_resilience(tmp_path):
    """
    Tests if the processor correctly sorts history even if snapshots 
    are persisted out of chronological order.
    """
    db_path = tmp_path / "test_sort.db"
    persistence = SquadPersistence(db_path)
    extractor = SquadTemporalExtractor()
    processor = FeatureBatchProcessor(persistence, extractor)
    
    ts_early = datetime(2025, 12, 1, 10, 0, tzinfo=timezone.utc)
    ts_late = datetime(2026, 12, 1, 10, 0, tzinfo=timezone.utc)
    
    # Persist late first
    snap_late = SquadSnapshot(1, "A", [], "S", ts_late, ts_late)
    snap_early = SquadSnapshot(1, "A", [], "S", ts_early, ts_early)
    
    persistence.save_snapshot(snap_late)
    persistence.save_snapshot(snap_early)
    
    history = persistence.get_snapshot_history(1)
    # The Persistence layer should handle sorting or the processor should
    recs = processor.process_team_feature_history(1)
    
    # Check that first record in list corresponds to ts_early
    # (Each snapshot generates multiple records, one per feature)
    first_ts = recs[0].feature_timestamp_utc
    assert first_ts == ts_early
