import pytest
from datetime import datetime, timezone, timedelta
from src.squads.temporal_features import SquadTemporalExtractor, SquadTemporalFeatures

class MockSnapshot:
    def __init__(self, team_id, players, announcement_ts):
        self.team_id = team_id
        self.players = players
        self.announcement_timestamp_utc = announcement_ts

def test_jaccard_identical_squads():
    extractor = SquadTemporalExtractor()
    s1 = {"P1_Def", "P2_Mid"}
    s2 = {"P1_Def", "P2_Mid"}
    assert extractor.compute_jaccard_similarity(s1, s2) == 1.0

def test_jaccard_disjoint_squads():
    extractor = SquadTemporalExtractor()
    s1 = {"P1_Def"}
    s2 = {"P2_Def"}
    assert extractor.compute_jaccard_similarity(s1, s2) == 0.0

def test_jaccard_empty_squads():
    extractor = SquadTemporalExtractor()
    assert extractor.compute_jaccard_similarity(set(), set()) == 1.0

def test_positional_continuity():
    extractor = SquadTemporalExtractor()
    ts = datetime.now(timezone.utc)
    
    curr = MockSnapshot(1, [{"name": "A", "position": "Defender"}, {"name": "B", "position": "Midfielder"}], ts)
    prev = MockSnapshot(1, [{"name": "A", "position": "Defender"}], ts - timedelta(days=30))
    
    features = extractor.build_features_from_snapshots(curr, prev, ts + timedelta(hours=5), 101)
    
    assert features.defenders_continuity == 1.0
    assert features.midfielders_continuity == 0.0
    assert features.continuity_index == 0.5 # A in both, B only in curr. Intersection=1, Union=2.

def test_lead_time_correctness():
    extractor = SquadTemporalExtractor()
    announcement = datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc)
    kickoff = datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc)
    
    lead = extractor.compute_announcement_lead_hours(announcement, kickoff)
    assert lead == 5.0

def test_leakage_exception():
    extractor = SquadTemporalExtractor()
    announcement = datetime(2026, 5, 20, 16, 0, tzinfo=timezone.utc)
    kickoff = datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc)
    
    with pytest.raises(ValueError, match="Anti-leakage violation"):
        extractor.compute_announcement_lead_hours(announcement, kickoff)

def test_squad_size_delta():
    extractor = SquadTemporalExtractor()
    ts = datetime.now(timezone.utc)
    curr = MockSnapshot(1, [{"name": "A", "position": "D"}, {"name": "B", "position": "D"}], ts)
    prev = MockSnapshot(1, [{"name": "A", "position": "D"}], ts - timedelta(days=1))
    
    features = extractor.build_features_from_snapshots(curr, prev, ts + timedelta(hours=1), 1)
    assert features.squad_size == 2
    assert features.squad_size_delta == 1

def test_first_ever_snapshot():
    # Test behavior when previous_snapshot is None
    extractor = SquadTemporalExtractor()
    ts = datetime.now(timezone.utc)
    curr = MockSnapshot(1, [{"name": "A", "position": "Defender"}], ts)
    
    features = extractor.build_features_from_snapshots(curr, None, ts + timedelta(hours=1), 1)
    assert features.continuity_index == 0.0
    assert features.squad_size_delta == 0
