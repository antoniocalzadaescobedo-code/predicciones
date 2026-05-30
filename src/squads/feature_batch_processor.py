import logging
from typing import List, Optional
from src.squads.squad_persistence import SquadSnapshot, SquadPersistence
from src.squads.temporal_features import SquadTemporalExtractor, SquadTemporalFeatures
from src.squads.temporal_feature_store import TemporalFeatureRecord, TemporalFeatureStore
from datetime import datetime, timezone

logger = logging.getLogger("batch_processor")

class FeatureBatchProcessor:
    """
    Transforms historical snapshots into point-in-time correct features.
    Iterates through the timeline to build continuity signals.
    """
    
    def __init__(self, persistence: SquadPersistence, extractor: SquadTemporalExtractor):
        self.persistence = persistence
        self.extractor = extractor

    def process_team_feature_history(self, team_id: int) -> List[TemporalFeatureRecord]:
        """
        Builds the entire feature timeline for a team by replaying history.
        """
        history = self.persistence.get_snapshot_history(team_id)
        if not history:
            logger.info(f"No history found for team {team_id}")
            return []

        # Sort by announcement to ensure continuity chain
        history.sort(key=lambda x: x.announcement_timestamp_utc)
        
        feature_records = []
        for i in range(len(history)):
            current = history[i]
            previous = history[i-1] if i > 0 else None
            
            # Prediction timestamp is assumed to be right after announcement for batch gen
            # Real prediction TS will be kicker_timestamp during dataset building
            fake_kickoff = current.announcement_timestamp_utc # Minimum lead time test
            
            features = self.extractor.build_features_from_snapshots(
                current_snapshot=current,
                previous_snapshot=previous,
                kickoff_timestamp=fake_kickoff,
                snapshot_id=i # Simplified ID for batch
            )
            
            # Map SquadTemporalFeatures to TemporalFeatureRecord
            feature_names = [
                "continuity_index", "defenders_continuity", 
                "midfielders_continuity", "forwards_continuity",
                "squad_size", "squad_size_delta"
            ]
            
            for name in feature_names:
                feature_records.append(TemporalFeatureRecord(
                    team_id=team_id,
                    feature_name=name,
                    feature_value=float(getattr(features, name)),
                    feature_timestamp_utc=current.announcement_timestamp_utc,
                    source_snapshot_id=i,
                    feature_version="1.0.0"
                ))
                
        return feature_records
