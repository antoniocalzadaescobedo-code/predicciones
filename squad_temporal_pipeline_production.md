# PRODUCTION-GRADE TEMPORAL SQUAD FEATURES PIPELINE
## Anti-Leakage, Causally Valid, Statistically Rigorous

## ARCHITECTURE OVERVIEW

**Separation of Concerns:**

```
SQLite (Operational):
- Cache operacional
- Metadata UI
- Provenance tracking

DuckDB + Parquet (Feature Store):
- Feature store temporal
- Analytics joins temporales
- Offline training
- Rolling windows
- Purged CV
```

**Why This Separation:**

- **SQLite**: Zero-latency reads for UI, transactional integrity for provenance
- **DuckDB + Parquet**: Columnar analytics, temporal joins, vectorized operations, compression

**Tradeoffs:**
- SQLite: Write-heavy for provenance, but negligible for metadata
- DuckDB: Read-optimized for analytics, but requires Parquet management
- Complexity: Dual storage, but justified by separation of concerns

## MODULE 1: SQUAD TEMPORAL PERSISTENCE (SQLite)

**Purpose:** Transactional storage of squad snapshots with provenance tracking.

**Schema:**

```sql
CREATE TABLE squad_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id TEXT NOT NULL,
    team_name TEXT NOT NULL,
    fetch_timestamp_utc TEXT NOT NULL,  -- ISO 8601
    players_json TEXT NOT NULL,         -- JSON completo
    squad_status TEXT NOT NULL,         -- Final/Preliminary
    announcement_date TEXT,             -- Si disponible
    api_version TEXT NOT NULL,          -- Version de API
    computation_version TEXT NOT NULL,  -- Version de código
    extraction_timestamp TEXT NOT NULL, -- Timestamp de extracción
    CHECK (datetime(fetch_timestamp_utc) IS NOT NULL),
    CHECK (datetime(extraction_timestamp) IS NOT NULL),
    INDEX idx_team_timestamp (team_id, fetch_timestamp_utc),
    INDEX idx_extraction_timestamp (extraction_timestamp)
);

CREATE TABLE feature_provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL,
    feature_value REAL NOT NULL,
    source_snapshot_id INTEGER NOT NULL,
    computation_version TEXT NOT NULL,
    extraction_timestamp TEXT NOT NULL,
    FOREIGN KEY (source_snapshot_id) REFERENCES squad_snapshots(id),
    CHECK (datetime(extraction_timestamp) IS NOT NULL),
    INDEX idx_feature_name (feature_name),
    INDEX idx_extraction_timestamp (extraction_timestamp)
);
```

**Anti-Leakage Enforcement:**

```python
assert datetime.fromisoformat(fetch_timestamp_utc) < datetime.fromisoformat(kickoff_timestamp)
```

**Implementation:**

```python
"""
squad_persistence.py
Transactional persistence of squad snapshots with provenance tracking.
"""

import sqlite3
import json
from datetime import datetime, UTC
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

class SquadStatus(Enum):
    FINAL = "Final"
    PRELIMINARY = "Preliminary"

@dataclass(frozen=True)
class PlayerSnapshot:
    """Immutable player snapshot for auditability."""
    player: str
    position: str
    jersey_number: int
    club: str
    age: int

@dataclass(frozen=True)
class SquadSnapshot:
    """Immutable squad snapshot for auditability."""
    team_id: str
    team_name: str
    fetch_timestamp_utc: str
    players: List[PlayerSnapshot]
    squad_status: SquadStatus
    announcement_date: Optional[str]
    api_version: str
    computation_version: str
    extraction_timestamp: str
    
    def __post_init__(self):
        """Validate timestamps are ISO 8601."""
        try:
            datetime.fromisoformat(self.fetch_timestamp_utc)
            datetime.fromisoformat(self.extraction_timestamp)
            if self.announcement_date:
                datetime.fromisoformat(self.announcement_date)
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format: {e}")

@dataclass(frozen=True)
class FeatureProvenance:
    """Immutable feature provenance for auditability."""
    feature_name: str
    feature_value: float
    source_snapshot_id: int
    computation_version: str
    extraction_timestamp: str

class SquadPersistence:
    """
    Transactional persistence of squad snapshots with strict anti-leakage.
    """
    
    def __init__(self, db_path: str = "data/squad_persistence.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite schema with constraints."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS squad_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id TEXT NOT NULL,
                team_name TEXT NOT NULL,
                fetch_timestamp_utc TEXT NOT NULL,
                players_json TEXT NOT NULL,
                squad_status TEXT NOT NULL,
                announcement_date TEXT,
                api_version TEXT NOT NULL,
                computation_version TEXT NOT NULL,
                extraction_timestamp TEXT NOT NULL,
                CHECK (datetime(fetch_timestamp_utc) IS NOT NULL),
                CHECK (datetime(extraction_timestamp) IS NOT NULL),
                INDEX idx_team_timestamp (team_id, fetch_timestamp_utc),
                INDEX idx_extraction_timestamp (extraction_timestamp)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                feature_value REAL NOT NULL,
                source_snapshot_id INTEGER NOT NULL,
                computation_version TEXT NOT NULL,
                extraction_timestamp TEXT NOT NULL,
                FOREIGN KEY (source_snapshot_id) REFERENCES squad_snapshots(id),
                CHECK (datetime(extraction_timestamp) IS NOT NULL),
                INDEX idx_feature_name (feature_name),
                INDEX idx_extraction_timestamp (extraction_timestamp)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_snapshot(self, snapshot: SquadSnapshot) -> int:
        """
        Save squad snapshot with transactional integrity.
        
        Args:
            snapshot: Immutable squad snapshot
        
        Returns:
            snapshot_id: Primary key of inserted row
        """
        # Validate snapshot structure
        snapshot.__post_init__()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO squad_snapshots 
                (team_id, team_name, fetch_timestamp_utc, players_json,
                 squad_status, announcement_date, api_version, 
                 computation_version, extraction_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.team_id,
                snapshot.team_name,
                snapshot.fetch_timestamp_utc,
                json.dumps([asdict(p) for p in snapshot.players]),
                snapshot.squad_status.value,
                snapshot.announcement_date,
                snapshot.api_version,
                snapshot.computation_version,
                snapshot.extraction_timestamp
            ))
            
            snapshot_id = cursor.lastrowid
            conn.commit()
            
            return snapshot_id
            
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to save snapshot: {e}")
        finally:
            conn.close()
    
    def get_latest_snapshot_before(self, team_id: str, 
                                   before_timestamp: str) -> Optional[SquadSnapshot]:
        """
        Get latest snapshot strictly before timestamp (anti-leakage).
        
        CRITICAL: fetch_timestamp_utc < before_timestamp
        
        Args:
            team_id: Team identifier
            before_timestamp: Timestamp boundary (kickoff)
        
        Returns:
            SquadSnapshot or None if no valid snapshot exists
        """
        # Validate timestamp format
        try:
            datetime.fromisoformat(before_timestamp)
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {before_timestamp}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, team_id, team_name, fetch_timestamp_utc, players_json,
                   squad_status, announcement_date, api_version, 
                   computation_version, extraction_timestamp
            FROM squad_snapshots
            WHERE team_id = ? AND fetch_timestamp_utc < ?
            ORDER BY fetch_timestamp_utc DESC
            LIMIT 1
        """, (team_id, before_timestamp))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Anti-leakage assertion (redundant but defensive)
        fetch_ts = datetime.fromisoformat(row[3])
        before_ts = datetime.fromisoformat(before_timestamp)
        assert fetch_ts < before_ts, f"Leakage detected: {fetch_ts} >= {before_ts}"
        
        return SquadSnapshot(
            team_id=row[1],
            team_name=row[2],
            fetch_timestamp_utc=row[3],
            players=[
                PlayerSnapshot(**p) for p in json.loads(row[4])
            ],
            squad_status=SquadStatus(row[5]),
            announcement_date=row[6],
            api_version=row[7],
            computation_version=row[8],
            extraction_timestamp=row[9]
        )
    
    def save_feature_provenance(self, provenance: FeatureProvenance) -> int:
        """
        Save feature provenance for auditability.
        
        Args:
            provenance: Immutable feature provenance
        
        Returns:
            provenance_id: Primary key of inserted row
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO feature_provenance 
                (feature_name, feature_value, source_snapshot_id,
                 computation_version, extraction_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                provenance.feature_name,
                provenance.feature_value,
                provenance.source_snapshot_id,
                provenance.computation_version,
                provenance.extraction_timestamp
            ))
            
            provenance_id = cursor.lastrowid
            conn.commit()
            
            return provenance_id
            
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to save provenance: {e}")
        finally:
            conn.close()
```

## MODULE 2: TEMPORAL FEATURE STORE (DuckDB + Parquet)

**Purpose:** High-performance feature extraction with temporal joins and analytics.

**Schema (DuckDB):**

```sql
-- Parquet files for columnar storage
CREATE OR REPLACE TABLE squad_snapshots_parquet AS 
SELECT * FROM read_parquet('data/squad_snapshots/*.parquet');

CREATE OR REPLACE TABLE match_features_parquet AS 
SELECT * FROM read_parquet('data/match_features/*.parquet');

-- Temporal join for feature extraction
CREATE OR REPLACE VIEW temporal_squad_features AS
SELECT 
    m.match_id,
    m.kickoff_timestamp,
    m.home_team,
    m.away_team,
    
    -- Home team features
    home_squad.fetch_timestamp_utc AS home_snapshot_timestamp,
    home_squad.squad_status AS home_squad_status,
    
    -- Away team features
    away_squad.fetch_timestamp_utc AS away_snapshot_timestamp,
    away_squad.squad_status AS away_squad_status,
    
    -- Anti-leakage validation
    m.kickoff_timestamp > home_squad.fetch_timestamp_utc AS home_anti_leakage_valid,
    m.kickoff_timestamp > away_squad.fetch_timestamp_utc AS away_anti_leakage_valid
    
FROM match_features_parquet m
LEFT JOIN squad_snapshots_parquet home_squad
    ON m.home_team = home_squad.team_id
    AND home_squad.fetch_timestamp_utc < m.kickoff_timestamp
LEFT JOIN (
    SELECT team_id, fetch_timestamp_utc, squad_status,
           ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY fetch_timestamp_utc DESC) as rn
    FROM squad_snapshots_parquet
) latest_home ON home_squad.team_id = latest_home.team_id 
    AND home_squad.fetch_timestamp_utc = latest_home.fetch_timestamp_utc
    AND latest_home.rn = 1

LEFT JOIN squad_snapshots_parquet away_squad
    ON m.away_team = away_squad.team_id
    AND away_squad.fetch_timestamp_utc < m.kickoff_timestamp
LEFT JOIN (
    SELECT team_id, fetch_timestamp_utc, squad_status,
           ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY fetch_timestamp_utc DESC) as rn
    FROM squad_snapshots_parquet
) latest_away ON away_squad.team_id = latest_away.team_id 
    AND away_squad.fetch_timestamp_utc = latest_away.fetch_timestamp_utc
    AND latest_away.rn = 1

WHERE home_anti_leakage_valid = TRUE
  AND away_anti_leakage_valid = TRUE;
```

**Implementation:**

```python
"""
temporal_feature_store.py
DuckDB + Parquet feature store for temporal squad features.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

class TemporalFeatureStore:
    """
    High-performance feature extraction with DuckDB + Parquet.
    Enforces anti-leakage via temporal joins.
    """
    
    def __init__(self, parquet_dir: str = "data/feature_store"):
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(database=":memory:")
    
    def ingest_squad_snapshots(self, snapshots_df: pd.DataFrame) -> None:
        """
        Ingest squad snapshots to Parquet with partitioning.
        
        Args:
            snapshots_df: DataFrame with squad snapshots
        """
        # Validate required columns
        required_cols = ['team_id', 'team_name', 'fetch_timestamp_utc', 
                        'players_json', 'squad_status']
        missing = set(required_cols) - set(snapshots_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Validate timestamps
        try:
            pd.to_datetime(snapshots_df['fetch_timestamp_utc'])
        except Exception as e:
            raise ValueError(f"Invalid fetch_timestamp_utc: {e}")
        
        # Write to Parquet with partitioning by team_id
        table = pa.Table.from_pandas(snapshots_df)
        output_path = self.parquet_dir / "squad_snapshots"
        
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=['team_id'],
            compression='snappy'
        )
    
    def ingest_match_features(self, matches_df: pd.DataFrame) -> None:
        """
        Ingest match features to Parquet.
        
        Args:
            matches_df: DataFrame with match features
        """
        required_cols = ['match_id', 'kickoff_timestamp', 'home_team', 'away_team']
        missing = set(required_cols) - set(matches_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Validate timestamps
        try:
            pd.to_datetime(matches_df['kickoff_timestamp'])
        except Exception as e:
            raise ValueError(f"Invalid kickoff_timestamp: {e}")
        
        table = pa.Table.from_pandas(matches_df)
        output_path = self.parquet_dir / "match_features"
        
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=['date'],  # Partition by date for temporal queries
            compression='snappy'
        )
    
    def extract_temporal_features(self, match_id: str, 
                                 kickoff_timestamp: str) -> pd.DataFrame:
        """
        Extract temporal features for a match with anti-leakage enforcement.
        
        CRITICAL: Only uses snapshots with fetch_timestamp_utc < kickoff_timestamp
        
        Args:
            match_id: Match identifier
            kickoff_timestamp: Match kickoff timestamp
        
        Returns:
            DataFrame with temporal features
        """
        # Anti-leakage validation
        kickoff_dt = datetime.fromisoformat(kickoff_timestamp)
        
        query = f"""
        WITH latest_home_snapshot AS (
            SELECT team_id, fetch_timestamp_utc, players_json, squad_status,
                   ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY fetch_timestamp_utc DESC) as rn
            FROM read_parquet('{self.parquet_dir}/squad_snapshots/*.parquet')
            WHERE fetch_timestamp_utc < '{kickoff_timestamp}'
        ),
        latest_away_snapshot AS (
            SELECT team_id, fetch_timestamp_utc, players_json, squad_status,
                   ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY fetch_timestamp_utc DESC) as rn
            FROM read_parquet('{self.parquet_dir}/squad_snapshots/*.parquet')
            WHERE fetch_timestamp_utc < '{kickoff_timestamp}'
        )
        SELECT 
            '{match_id}' as match_id,
            h.fetch_timestamp_utc as home_snapshot_timestamp,
            h.squad_status as home_squad_status,
            h.players_json as home_players_json,
            a.fetch_timestamp_utc as away_snapshot_timestamp,
            a.squad_status as away_squad_status,
            a.players_json as away_players_json
        FROM latest_home_snapshot h
        CROSS JOIN latest_away_snapshot a
        WHERE h.rn = 1 AND a.rn = 1
        """
        
        result = self.con.execute(query).fetchdf()
        
        # Anti-leakage assertion
        if not result.empty:
            home_ts = datetime.fromisoformat(result['home_snapshot_timestamp'].iloc[0])
            away_ts = datetime.fromisoformat(result['away_snapshot_timestamp'].iloc[0])
            assert home_ts < kickoff_dt, f"Home leakage: {home_ts} >= {kickoff_dt}"
            assert away_ts < kickoff_dt, f"Away leakage: {away_ts} >= {kickoff_dt}"
        
        return result
    
    def extract_historical_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Extract historical features for time range with rolling window.
        
        Args:
            start_date: Start date (ISO 8601)
            end_date: End date (ISO 8601)
        
        Returns:
            DataFrame with historical features
        """
        query = f"""
        SELECT 
            m.match_id,
            m.kickoff_timestamp,
            m.home_team,
            m.away_team,
            h.fetch_timestamp_utc as home_snapshot_timestamp,
            a.fetch_timestamp_utc as away_snapshot_timestamp
        FROM read_parquet('{self.parquet_dir}/match_features/*.parquet') m
        LEFT JOIN (
            SELECT team_id, fetch_timestamp_utc,
                   ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY fetch_timestamp_utc DESC) as rn
            FROM read_parquet('{self.parquet_dir}/squad_snapshots/*.parquet')
        ) h ON m.home_team = h.team_id AND h.rn = 1
        LEFT JOIN (
            SELECT team_id, fetch_timestamp_utc,
                   ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY fetch_timestamp_utc DESC) as rn
            FROM read_parquet('{self.parquet_dir}/squad_snapshots/*.parquet')
        ) a ON m.away_team = a.team_id AND a.rn = 1
        WHERE m.kickoff_timestamp >= '{start_date}'
          AND m.kickoff_timestamp <= '{end_date}'
          AND h.fetch_timestamp_utc < m.kickoff_timestamp
          AND a.fetch_timestamp_utc < m.kickoff_timestamp
        """
        
        return self.con.execute(query).fetchdf()
```

## MODULE 3: TEMPORAL FEATURE EXTRACTION

**Purpose:** Mathematically defined, causally valid feature extraction.

**Features with Formal Definitions:**

### 1. Continuity Index (Jaccard Similarity)

**Mathematical Definition:**
```
C_t(team) = |S_t ∩ S_{t-1}| / |S_t ∪ S_{t-1}|
```

**Causal Hypothesis:**
- High continuity → tactical cohesion → better performance
- Low continuity → rotation → tactical uncertainty

**Scientific Critique:**
- **Weak:** Continuity not causally guaranteed to improve performance
- **Risk:** Noise if squad changes due to external factors (injuries, suspensions)
- **Validation Required:** Statistical significance, temporal stability

**Implementation:**

```python
"""
temporal_features.py
Mathematically defined temporal features with causal validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class FeatureType(Enum):
    CONTINUITY = "continuity"
    STABILITY = "stability"
    TEMPORAL = "temporal"

@dataclass(frozen=True)
class TemporalFeature:
    """Immutable temporal feature with metadata."""
    name: str
    value: float
    feature_type: FeatureType
    computation_timestamp: str
    source_snapshot_timestamp: str
    causal_hypothesis: str
    scientific_risk: str
    
    def __post_init__(self):
        """Validate feature value is numeric."""
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Feature value must be numeric: {self.value}")

class TemporalFeatureExtractor:
    """
    Extract mathematically defined temporal features from squad snapshots.
    """
    
    @staticmethod
    def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        C = |A ∩ B| / |A ∪ B|
        
        Args:
            set_a: First set of player names
            set_b: Second set of player names
        
        Returns:
            Jaccard similarity [0.0, 1.0]
        """
        if not set_a and not set_b:
            return 1.0  # Both empty → maximum similarity
        if not set_a or not set_b:
            return 0.0  # One empty → zero similarity
        
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        return intersection / union if union > 0 else 0.0
    
    def extract_continuity_index(self, current_players: Set[str],
                                previous_players: Set[str],
                                computation_timestamp: str,
                                source_timestamp: str) -> TemporalFeature:
        """
        Extract continuity index between consecutive squad announcements.
        
        Mathematical Definition:
        C_t = |S_t ∩ S_{t-1}| / |S_t ∪ S_{t-1}|
        
        Causal Hypothesis:
        High continuity → tactical cohesion → better performance
        
        Scientific Risk:
        Weak causality, noise from external factors (injuries, suspensions)
        
        Args:
            current_players: Current squad player names
            previous_players: Previous squad player names
            computation_timestamp: When feature was computed
            source_timestamp: Timestamp of source snapshot
        
        Returns:
            TemporalFeature with continuity index
        """
        continuity = self.jaccard_similarity(current_players, previous_players)
        
        return TemporalFeature(
            name="continuity_index",
            value=continuity,
            feature_type=FeatureType.CONTINUITY,
            computation_timestamp=computation_timestamp,
            source_snapshot_timestamp=source_timestamp,
            causal_hypothesis="High continuity → tactical cohesion → better performance",
            scientific_risk="Weak causality, noise from external factors"
        )
    
    def extract_positional_continuity(self, current_players: List[Dict],
                                     previous_players: List[Dict],
                                     position: str,
                                     computation_timestamp: str,
                                     source_timestamp: str) -> TemporalFeature:
        """
        Extract positional continuity for specific position.
        
        Mathematical Definition:
        C_t^pos = |S_t^pos ∩ S_{t-1}^pos| / |S_t^pos ∪ S_{t-1}^pos|
        
        Causal Hypothesis:
        Positional continuity → positional stability → performance
        
        Scientific Risk:
        Weaker than overall continuity, smaller subsets → higher variance
        
        Args:
            current_players: Current squad players with positions
            previous_players: Previous squad players with positions
            position: Position to analyze (Defensa, Mediocampista, etc.)
            computation_timestamp: When feature was computed
            source_timestamp: Timestamp of source snapshot
        
        Returns:
            TemporalFeature with positional continuity
        """
        current_pos = {p['player'] for p in current_players 
                      if p.get('position') == position}
        previous_pos = {p['player'] for p in previous_players 
                       if p.get('position') == position}
        
        continuity = self.jaccard_similarity(current_pos, previous_pos)
        
        return TemporalFeature(
            name=f"continuity_{position.lower()}",
            value=continuity,
            feature_type=FeatureType.CONTINUITY,
            computation_timestamp=computation_timestamp,
            source_snapshot_timestamp=source_timestamp,
            causal_hypothesis=f"Positional continuity → {position} stability → performance",
            scientific_risk="Weaker than overall continuity, smaller subsets → higher variance"
        )
    
    def extract_squad_size(self, players: List[Dict],
                          computation_timestamp: str,
                          source_timestamp: str) -> TemporalFeature:
        """
        Extract squad size.
        
        Mathematical Definition:
        SS_t = |S_t|
        
        Causal Hypothesis:
        Squad size → organizational capacity
        
        Scientific Risk:
        Very weak causality, likely pure noise
        
        Args:
            players: List of players in squad
            computation_timestamp: When feature was computed
            source_timestamp: Timestamp of source snapshot
        
        Returns:
            TemporalFeature with squad size
        """
        size = len(players)
        
        return TemporalFeature(
            name="squad_size",
            value=float(size),
            feature_type=FeatureType.STABILITY,
            computation_timestamp=computation_timestamp,
            source_snapshot_timestamp=source_timestamp,
            causal_hypothesis="Squad size → organizational capacity",
            scientific_risk="Very weak causality, likely pure noise"
        )
    
    def extract_squad_size_delta(self, current_size: int,
                               previous_size: int,
                               computation_timestamp: str,
                               source_timestamp: str) -> TemporalFeature:
        """
        Extract change in squad size.
        
        Mathematical Definition:
        ΔSS_t = |S_t| - |S_{t-1}|
        
        Causal Hypothesis:
        Size changes → organizational instability
        
        Scientific Risk:
        Very weak causality, likely pure noise
        
        Args:
            current_size: Current squad size
            previous_size: Previous squad size
            computation_timestamp: When feature was computed
            source_timestamp: Timestamp of source snapshot
        
        Returns:
            TemporalFeature with squad size delta
        """
        delta = current_size - previous_size
        
        return TemporalFeature(
            name="squad_size_delta",
            value=float(delta),
            feature_type=FeatureType.STABILITY,
            computation_timestamp=computation_timestamp,
            source_snapshot_timestamp=source_timestamp,
            causal_hypothesis="Size changes → organizational instability",
            scientific_risk="Very weak causality, likely pure noise"
        )
    
    def extract_announcement_lead_time(self, announcement_date: Optional[str],
                                      kickoff_timestamp: str,
                                      computation_timestamp: str,
                                      source_timestamp: str) -> TemporalFeature:
        """
        Extract lead time between squad announcement and kickoff.
        
        Mathematical Definition:
        ALT_t = kickoff_timestamp - announcement_timestamp
        
        Causal Hypothesis:
        Longer lead time → more preparation → better performance
        
        Scientific Risk:
        Moderately weak, confounding with opposition quality
        
        Args:
            announcement_date: Squad announcement date (ISO 8601)
            kickoff_timestamp: Match kickoff timestamp
            computation_timestamp: When feature was computed
            source_timestamp: Timestamp of source snapshot
        
        Returns:
            TemporalFeature with announcement lead time (days)
        """
        if not announcement_date:
            # Missing data → return neutral value
            return TemporalFeature(
                name="announcement_lead_time",
                value=30.0,  # Default 30 days
                feature_type=FeatureType.TEMPORAL,
                computation_timestamp=computation_timestamp,
                source_snapshot_timestamp=source_timestamp,
                causal_hypothesis="Longer lead time → more preparation → better performance",
                scientific_risk="Missing data, using default"
            )
        
        try:
            announcement = datetime.fromisoformat(announcement_date)
            kickoff = datetime.fromisoformat(kickoff_timestamp)
            lead_time = (kickoff - announcement).days
            lead_time = max(0, lead_time)  # Non-negative
        except (ValueError, TypeError) as e:
            # Invalid date → return neutral value
            return TemporalFeature(
                name="announcement_lead_time",
                value=30.0,
                feature_type=FeatureType.TEMPORAL,
                computation_timestamp=computation_timestamp,
                source_snapshot_timestamp=source_timestamp,
                causal_hypothesis="Longer lead time → more preparation → better performance",
                scientific_risk=f"Invalid date format: {e}, using default"
            )
        
        return TemporalFeature(
            name="announcement_lead_time",
            value=float(lead_time),
            feature_type=FeatureType.TEMPORAL,
            computation_timestamp=computation_timestamp,
            source_snapshot_timestamp=source_timestamp,
            causal_hypothesis="Longer lead time → more preparation → better performance",
            scientific_risk="Moderately weak, confounding with opposition quality"
        )
```

## MODULE 4: ANTI-LEAKAGE VALIDATION

**Purpose:** Strict enforcement of temporal causality.

**Implementation:**

```python
"""
anti_leakage_validator.py
Strict anti-leakage validation for temporal features.
"""

import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass(frozen=True)
class LeakageViolation:
    """Immutable record of leakage violation."""
    feature_name: str
    feature_timestamp: str
    kickoff_timestamp: str
    violation_type: str

class AntiLeakageValidator:
    """
    Strict anti-leakage validation for temporal features.
    
    INVARIANT: feature_timestamp < kickoff_timestamp
    """
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.violations: List[LeakageViolation] = []
    
    def validate_feature_timestamp(self, feature_timestamp: str,
                                  kickoff_timestamp: str,
                                  feature_name: str) -> bool:
        """
        Validate that feature timestamp is strictly before kickoff.
        
        Args:
            feature_timestamp: Timestamp when feature was computed
            kickoff_timestamp: Match kickoff timestamp
            feature_name: Name of feature being validated
        
        Returns:
            True if valid, False if leakage detected
        """
        try:
            feature_dt = datetime.fromisoformat(feature_timestamp)
            kickoff_dt = datetime.fromisoformat(kickoff_timestamp)
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format: {e}")
        
        if feature_dt >= kickoff_dt:
            violation = LeakageViolation(
                feature_name=feature_name,
                feature_timestamp=feature_timestamp,
                kickoff_timestamp=kickoff_timestamp,
                violation_type="feature_timestamp >= kickoff_timestamp"
            )
            self.violations.append(violation)
            
            if self.strict_mode:
                raise AssertionError(
                    f"Leakage detected for {feature_name}: "
                    f"{feature_timestamp} >= {kickoff_timestamp}"
                )
            
            return False
        
        return True
    
    def validate_dataframe(self, df: pd.DataFrame,
                         feature_timestamp_col: str,
                         kickoff_timestamp_col: str) -> List[LeakageViolation]:
        """
        Validate entire DataFrame for leakage.
        
        Args:
            df: DataFrame with features
            feature_timestamp_col: Column name for feature timestamps
            kickoff_timestamp_col: Column name for kickoff timestamps
        
        Returns:
            List of leakage violations
        """
        self.violations = []
        
        for idx, row in df.iterrows():
            feature_ts = row[feature_timestamp_col]
            kickoff_ts = row[kickoff_timestamp_col]
            feature_name = row.get('feature_name', f'feature_{idx}')
            
            self.validate_feature_timestamp(feature_ts, kickoff_ts, feature_name)
        
        return self.violations
    
    def get_leakage_report(self) -> Dict:
        """
        Generate leakage report.
        
        Returns:
            Dict with leakage statistics
        """
        return {
            'total_violations': len(self.violations),
            'strict_mode': self.strict_mode,
            'violations': [
                {
                    'feature_name': v.feature_name,
                    'feature_timestamp': v.feature_timestamp,
                    'kickoff_timestamp': v.kickoff_timestamp,
                    'violation_type': v.violation_type
                }
                for v in self.violations
            ]
        }
```

## MODULE 5: STATISTICAL VALIDATION

**Purpose:** Rigorous statistical validation before production.

**Implementation:**

```python
"""
statistical_validation.py
Statistical validation framework for temporal features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.utils import resample
from dataclasses import dataclass
from enum import Enum

class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"

@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result."""
    feature_name: str
    test_name: str
    status: ValidationStatus
    metric_value: float
    threshold: float
    p_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    reason: str

class StatisticalValidator:
    """
    Statistical validation for temporal features.
    
    REQUIRED METRICS:
    - Brier Score improvement >= 0.005
    - LogLoss improvement >= 0.01
    - Calibration slope 0.95-1.05
    - Temporal stability (KS test p-value > 0.05)
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        self.results: List[ValidationResult] = []
    
    def test_brier_improvement(self, baseline_brier: float,
                              model_brier: float,
                              feature_name: str) -> ValidationResult:
        """
        Test if model improves Brier score significantly.
        
        Required: improvement >= 0.005
        
        Args:
            baseline_brier: Baseline model Brier score
            model_brier: Model with feature Brier score
            feature_name: Name of feature being tested
        
        Returns:
            ValidationResult with Brier improvement test
        """
        improvement = baseline_brier - model_brier
        threshold = 0.005
        
        if improvement >= threshold:
            status = ValidationStatus.PASSED
            reason = f"Brier improvement {improvement:.4f} >= threshold {threshold}"
        else:
            status = ValidationStatus.FAILED
            reason = f"Brier improvement {improvement:.4f} < threshold {threshold}"
        
        result = ValidationResult(
            feature_name=feature_name,
            test_name="brier_improvement",
            status=status,
            metric_value=improvement,
            threshold=threshold,
            p_value=None,
            confidence_interval=None,
            reason=reason
        )
        
        self.results.append(result)
        return result
    
    def test_logloss_improvement(self, baseline_logloss: float,
                                model_logloss: float,
                                feature_name: str) -> ValidationResult:
        """
        Test if model improves LogLoss significantly.
        
        Required: improvement >= 0.01
        
        Args:
            baseline_logloss: Baseline model LogLoss
            model_logloss: Model with feature LogLoss
            feature_name: Name of feature being tested
        
        Returns:
            ValidationResult with LogLoss improvement test
        """
        improvement = baseline_logloss - model_logloss
        threshold = 0.01
        
        if improvement >= threshold:
            status = ValidationStatus.PASSED
            reason = f"LogLoss improvement {improvement:.4f} >= threshold {threshold}"
        else:
            status = ValidationStatus.FAILED
            reason = f"LogLoss improvement {improvement:.4f} < threshold {threshold}"
        
        result = ValidationResult(
            feature_name=feature_name,
            test_name="logloss_improvement",
            status=status,
            metric_value=improvement,
            threshold=threshold,
            p_value=None,
            confidence_interval=None,
            reason=reason
        )
        
        self.results.append(result)
        return result
    
    def test_calibration_slope(self, slope: float, feature_name: str) -> ValidationResult:
        """
        Test if calibration slope is within acceptable range.
        
        Required: 0.95 <= slope <= 1.05
        
        Args:
            slope: Calibration slope
            feature_name: Name of feature being tested
        
        Returns:
            ValidationResult with calibration slope test
        """
        lower_bound = 0.95
        upper_bound = 1.05
        
        if lower_bound <= slope <= upper_bound:
            status = ValidationStatus.PASSED
            reason = f"Calibration slope {slope:.3f} within [{lower_bound}, {upper_bound}]"
        else:
            status = ValidationStatus.FAILED
            reason = f"Calibration slope {slope:.3f} outside [{lower_bound}, {upper_bound}]"
        
        result = ValidationResult(
            feature_name=feature_name,
            test_name="calibration_slope",
            status=status,
            metric_value=slope,
            threshold=0.0,  # Not applicable
            p_value=None,
            confidence_interval=None,
            reason=reason
        )
        
        self.results.append(result)
        return result
    
    def test_temporal_stability(self, feature_values: List[float],
                                window_size: int = 30,
                                feature_name: str) -> ValidationResult:
        """
        Test temporal stability via Kolmogorov-Smirnov test.
        
        Required: KS test p-value > 0.05 (no significant drift)
        
        Args:
            feature_values: Feature values ordered temporally
            window_size: Size of temporal window for comparison
            feature_name: Name of feature being tested
        
        Returns:
            ValidationResult with temporal stability test
        """
        if len(feature_values) < window_size * 2:
            result = ValidationResult(
                feature_name=feature_name,
                test_name="temporal_stability",
                status=ValidationStatus.INSUFFICIENT_DATA,
                metric_value=0.0,
                threshold=0.0,
                p_value=None,
                confidence_interval=None,
                reason=f"Insufficient data: {len(feature_values)} < {window_size * 2}"
            )
            self.results.append(result)
            return result
        
        # Split into two temporal windows
        early = feature_values[:window_size]
        late = feature_values[-window_size:]
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(early, late)
        
        if p_value > self.alpha:
            status = ValidationStatus.PASSED
            reason = f"KS test p-value {p_value:.4f} > {self.alpha} (no significant drift)"
        else:
            status = ValidationStatus.FAILED
            reason = f"KS test p-value {p_value:.4f} <= {self.alpha} (significant drift detected)"
        
        result = ValidationResult(
            feature_name=feature_name,
            test_name="temporal_stability",
            status=status,
            metric_value=ks_stat,
            threshold=self.alpha,
            p_value=p_value,
            confidence_interval=None,
            reason=reason
        )
        
        self.results.append(result)
        return result
    
    def bootstrap_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                    feature_idx: int,
                                    n_bootstrap: int = 1000,
                                    feature_name: str = "unknown") -> ValidationResult:
        """
        Bootstrap confidence intervals for feature importance.
        
        Required: 95% CI does not overlap with zero (significant importance)
        
        Args:
            X: Feature matrix
            y: Target
            feature_idx: Index of feature to test
            n_bootstrap: Number of bootstrap iterations
            feature_name: Name of feature
        
        Returns:
            ValidationResult with bootstrap importance test
        """
        from sklearn.ensemble import RandomForestClassifier
        
        importances = []
        
        for _ in range(n_bootstrap):
            X_boot, y_boot = resample(X, y)
            
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X_boot, y_boot)
            
            importances.append(rf.feature_importances_[feature_idx])
        
        mean_importance = np.mean(importances)
        std_importance = np.std(importances)
        ci_lower = np.percentile(importances, 2.5)
        ci_upper = np.percentile(importances, 97.5)
        
        # Check if CI overlaps with zero
        if ci_lower > 0:
            status = ValidationStatus.PASSED
            reason = f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] does not overlap zero"
        else:
            status = ValidationStatus.FAILED
            reason = f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] overlaps zero"
        
        result = ValidationResult(
            feature_name=feature_name,
            test_name="bootstrap_importance",
            status=status,
            metric_value=mean_importance,
            threshold=0.0,
            p_value=None,
            confidence_interval=(ci_lower, ci_upper),
            reason=reason
        )
        
        self.results.append(result)
        return result
    
    def get_validation_summary(self) -> Dict:
        """
        Generate validation summary.
        
        Returns:
            Dict with validation statistics
        """
        passed = sum(1 for r in self.results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == ValidationStatus.FAILED)
        insufficient = sum(1 for r in self.results if r.status == ValidationStatus.INSUFFICIENT_DATA)
        
        return {
            'total_tests': len(self.results),
            'passed': passed,
            'failed': failed,
            'insufficient_data': insufficient,
            'pass_rate': passed / len(self.results) if self.results else 0.0,
            'results': [
                {
                    'feature_name': r.feature_name,
                    'test_name': r.test_name,
                    'status': r.status.value,
                    'metric_value': r.metric_value,
                    'threshold': r.threshold,
                    'p_value': r.p_value,
                    'confidence_interval': r.confidence_interval,
                    'reason': r.reason
                }
                for r in self.results
            ]
        }
```

## MODULE 6: INTEGRATION TESTS

**Purpose:** Concrete anti-leakage tests and validation.

**Implementation:**

```python
"""
test_anti_leakage.py
Concrete anti-leakage tests for temporal squad features.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from anti_leakage_validator import AntiLeakageValidator, LeakageViolation

class TestAntiLeakage:
    """Concrete anti-leakage tests."""
    
    def test_feature_timestamp_before_kickoff(self):
        """Test that feature timestamp is strictly before kickoff."""
        validator = AntiLeakageValidator(strict_mode=True)
        
        feature_ts = "2024-01-01T10:00:00"
        kickoff_ts = "2024-01-01T15:00:00"
        
        result = validator.validate_feature_timestamp(
            feature_ts, kickoff_ts, "test_feature"
        )
        
        assert result is True
    
    def test_feature_timestamp_after_kickoff_fails(self):
        """Test that feature timestamp after kickoff raises assertion."""
        validator = AntiLeakageValidator(strict_mode=True)
        
        feature_ts = "2024-01-01T16:00:00"
        kickoff_ts = "2024-01-01T15:00:00"
        
        with pytest.raises(AssertionError, match="Leakage detected"):
            validator.validate_feature_timestamp(
                feature_ts, kickoff_ts, "test_feature"
            )
    
    def test_feature_timestamp_equal_kickoff_fails(self):
        """Test that feature timestamp equal to kickoff raises assertion."""
        validator = AntiLeakageValidator(strict_mode=True)
        
        feature_ts = "2024-01-01T15:00:00"
        kickoff_ts = "2024-01-01T15:00:00"
        
        with pytest.raises(AssertionError, match="Leakage detected"):
            validator.validate_feature_timestamp(
                feature_ts, kickoff_ts, "test_feature"
            )
    
    def test_dataframe_leakage_detection(self):
        """Test leakage detection in DataFrame."""
        validator = AntiLeakageValidator(strict_mode=False)
        
        df = pd.DataFrame({
            'feature_name': ['f1', 'f2', 'f3'],
            'feature_timestamp': [
                '2024-01-01T10:00:00',
                '2024-01-01T16:00:00',  # Leakage
                '2024-01-01T12:00:00'
            ],
            'kickoff_timestamp': [
                '2024-01-01T15:00:00',
                '2024-01-01T15:00:00',
                '2024-01-01T15:00:00'
            ]
        })
        
        violations = validator.validate_dataframe(
            df, 'feature_timestamp', 'kickoff_timestamp'
        )
        
        assert len(violations) == 1
        assert violations[0].feature_name == 'f2'
    
    def test_invalid_timestamp_format_raises(self):
        """Test that invalid timestamp format raises ValueError."""
        validator = AntiLeakageValidator(strict_mode=True)
        
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            validator.validate_feature_timestamp(
                "invalid-date", "2024-01-01T15:00:00", "test_feature"
            )

class TestTemporalFeatures:
    """Tests for temporal feature extraction."""
    
    def test_jaccard_similarity_identical_sets(self):
        """Test Jaccard similarity for identical sets."""
        from temporal_features import TemporalFeatureExtractor
        
        extractor = TemporalFeatureExtractor()
        set_a = {"player1", "player2", "player3"}
        set_b = {"player1", "player2", "player3"}
        
        similarity = extractor.jaccard_similarity(set_a, set_b)
        
        assert similarity == 1.0
    
    def test_jaccard_similarity_disjoint_sets(self):
        """Test Jaccard similarity for disjoint sets."""
        from temporal_features import TemporalFeatureExtractor
        
        extractor = TemporalFeatureExtractor()
        set_a = {"player1", "player2"}
        set_b = {"player3", "player4"}
        
        similarity = extractor.jaccard_similarity(set_a, set_b)
        
        assert similarity == 0.0
    
    def test_jaccard_similarity_partial_overlap(self):
        """Test Jaccard similarity for partial overlap."""
        from temporal_features import TemporalFeatureExtractor
        
        extractor = TemporalFeatureExtractor()
        set_a = {"player1", "player2", "player3"}
        set_b = {"player2", "player3", "player4"}
        
        similarity = extractor.jaccard_similarity(set_a, set_b)
        
        # Intersection: {player2, player3} = 2
        # Union: {player1, player2, player3, player4} = 4
        # Jaccard: 2/4 = 0.5
        assert similarity == 0.5
    
    def test_continuity_index_value_range(self):
        """Test that continuity index is in [0.0, 1.0]."""
        from temporal_features import TemporalFeatureExtractor
        
        extractor = TemporalFeatureExtractor()
        
        current = {"player1", "player2", "player3"}
        previous = {"player2", "player3", "player4"}
        
        feature = extractor.extract_continuity_index(
            current, previous,
            "2024-01-01T10:00:00",
            "2024-01-01T09:00:00"
        )
        
        assert 0.0 <= feature.value <= 1.0
    
    def test_feature_immutability(self):
        """Test that TemporalFeature is immutable."""
        from temporal_features import TemporalFeature, FeatureType
        
        feature = TemporalFeature(
            name="test",
            value=0.5,
            feature_type=FeatureType.CONTINUITY,
            computation_timestamp="2024-01-01T10:00:00",
            source_snapshot_timestamp="2024-01-01T09:00:00",
            causal_hypothesis="test",
            scientific_risk="test"
        )
        
        with pytest.raises(AttributeError):
            feature.value = 0.6  # Should fail due to frozen=True

class TestStatisticalValidation:
    """Tests for statistical validation framework."""
    
    def test_brier_improvement_pass(self):
        """Test Brier improvement validation passes."""
        from statistical_validation import StatisticalValidator, ValidationStatus
        
        validator = StatisticalValidator()
        
        result = validator.test_brier_improvement(
            baseline_brier=0.25,
            model_brier=0.24,
            feature_name="test_feature"
        )
        
        assert result.status == ValidationStatus.PASSED
        assert result.metric_value == 0.005
    
    def test_brier_improvement_fail(self):
        """Test Brier improvement validation fails."""
        from statistical_validation import StatisticalValidator, ValidationStatus
        
        validator = StatisticalValidator()
        
        result = validator.test_brier_improvement(
            baseline_brier=0.25,
            model_brier=0.248,
            feature_name="test_feature"
        )
        
        assert result.status == ValidationStatus.FAILED
        assert result.metric_value == 0.002
    
    def test_calibration_slope_pass(self):
        """Test calibration slope validation passes."""
        from statistical_validation import StatisticalValidator, ValidationStatus
        
        validator = StatisticalValidator()
        
        result = validator.test_calibration_slope(
            slope=1.0,
            feature_name="test_feature"
        )
        
        assert result.status == ValidationStatus.PASSED
    
    def test_calibration_slope_fail(self):
        """Test calibration slope validation fails."""
        from statistical_validation import StatisticalValidator, ValidationStatus
        
        validator = StatisticalValidator()
        
        result = validator.test_calibration_slope(
            slope=0.9,
            feature_name="test_feature"
        )
        
        assert result.status == ValidationStatus.FAILED
    
    def test_temporal_stability_insufficient_data(self):
        """Test temporal stability with insufficient data."""
        from statistical_validation import StatisticalValidator, ValidationStatus
        
        validator = StatisticalValidator()
        
        result = validator.test_temporal_stability(
            feature_values=[0.5, 0.6, 0.7],
            window_size=30,
            feature_name="test_feature"
        )
        
        assert result.status == ValidationStatus.INSUFFICIENT_DATA
```

## PRODUCTION INTEGRATION STRATEGY

**Go/No-Go Criteria:**

**MUST PASS:**
1. Brier improvement >= 0.005
2. LogLoss improvement >= 0.01
3. Calibration slope 0.95-1.05
4. Temporal stability (KS p-value > 0.05)
5. Zero leakage violations
6. Bootstrap CI does not overlap zero

**IF ANY CRITERION FAILS:**
- Do NOT integrate into production
- Document failure mode
- Consider feature rejection

**Integration Steps:**

1. **Offline Research (Week 1-2):**
   - Implement persistence layer
   - Implement feature extraction
   - Backtest on historical data
   - Run statistical validation

2. **Validation (Week 3):**
   - Purged time-series CV
   - Bootstrap confidence intervals
   - Drift detection
   - Anti-leakage validation

3. **Production Decision (Week 4):**
   - If all criteria pass → integrate
   - If any criterion fails → reject feature

4. **Monitoring (Week 5+):**
   - Continuous drift detection
   - Calibration monitoring
   - Performance degradation alerts
   - Rollback plan

## RISKS AND LIMITATIONS

**Technical Risks:**
1. **Data Sparsity:** Limited historical snapshots → unreliable features
2. **Computational Overhead:** DuckDB queries add latency → caching required
3. **Storage Growth:** Parquet files grow over time → retention policy needed

**Scientific Risks:**
1. **Weak Causality:** Continuity features likely noise → statistical validation required
2. **Confounding:** External factors (injuries, suspensions) not observed → biased features
3. **Overfitting:** Features may not generalize → cross-validation required

**Operational Risks:**
1. **Pipeline Complexity:** Dual storage (SQLite + DuckDB) → operational overhead
2. **Maintenance:** Feature versioning → backward compatibility issues
3. **Monitoring:** Drift detection complexity → alert fatigue

## CONCLUSION

**Recommendation:**

Implement FASE 1-3 (persistence, extraction, validation) before any production integration.

**Critical Principle:**

NO feature reaches production without:
- Statistical uplift validation
- Temporal stability verification
- Zero leakage violations
- Bootstrap confidence intervals

**If validation fails:**

Reject feature. The cost of complexity does not justify marginal or negative uplift.

**Final Decision:**

Features are experimental until proven otherwise. Default to rejection unless evidence is overwhelming.
