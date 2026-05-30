import duckdb
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("temporal_feature_store")

@dataclass(frozen=True)
class TemporalFeatureRecord:
    """
    Immutable representation of a calculated feature.
    Contains full provenance metadata for auditability.
    """
    team_id: int
    feature_name: str
    feature_value: float
    feature_timestamp_utc: datetime
    source_snapshot_id: int
    feature_version: str
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.team_id <= 0:
            raise ValueError("team_id must be > 0")
        if not self.feature_name:
            raise ValueError("feature_name cannot be empty")
        if not isinstance(self.feature_value, (int, float)):
            raise ValueError("feature_value must be numeric")
        self._validate_utc(self.feature_timestamp_utc, "feature_timestamp_utc")
        self._validate_utc(self.created_at_utc, "created_at_utc")

    @staticmethod
    def _validate_utc(dt: datetime, field_name: str):
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt).total_seconds() != 0:
            raise ValueError(f"{field_name} must be a timezone-aware UTC datetime.")

class TemporalFeatureStore:
    """
    Point-in-time correct Feature Store using DuckDB and Parquet.
    Guarantees anti-leakage via ASOF joins and immutable partitioning.
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.features_path = base_path / "features_store"
        self.features_path.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(database=':memory:')
        self._setup_storage()

    def _setup_storage(self):
        """Prepare DuckDB views for parquet scanning."""
        parquet_glob = str(self.features_path / "**" / "*.parquet")
        if any(self.features_path.rglob("*.parquet")):
            self.conn.execute(f"CREATE OR REPLACE VIEW feature_view AS SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=1)")
        else:
            self.conn.execute("""
                CREATE OR REPLACE TABLE feature_view (
                    team_id INTEGER,
                    feature_name TEXT,
                    feature_value DOUBLE,
                    feature_timestamp_utc TIMESTAMP,
                    source_snapshot_id INTEGER,
                    feature_version TEXT,
                    created_at_utc TIMESTAMP,
                    year_month TEXT
                )
            """)
        logger.info(f"Feature Store views initialized at {self.features_path}")

    def save_feature_records(self, records: List[TemporalFeatureRecord]):
        """
        Persists features to partitioned Parquet files.
        """
        if not records:
            return

        data_list = []
        for r in records:
            data_list.append({
                "team_id": r.team_id,
                "feature_name": r.feature_name,
                "feature_value": r.feature_value,
                "feature_timestamp_utc": r.feature_timestamp_utc,
                "source_snapshot_id": r.source_snapshot_id,
                "feature_version": r.feature_version,
                "created_at_utc": r.created_at_utc,
                "year_month": r.feature_timestamp_utc.strftime("%Y-%m")
            })

        self.conn.execute("CREATE OR REPLACE TEMP TABLE batch_features AS SELECT * FROM (SELECT * FROM data_list)")
        
        export_query = f"""
            COPY batch_features TO '{self.features_path}' 
            (FORMAT PARQUET, PARTITION_BY (team_id, year_month), OVERWRITE_OR_IGNORE 1)
        """
        try:
            self.conn.execute(export_query)
            logger.info(f"Saved {len(records)} features to store.")
            self._setup_storage()
        except Exception as e:
            logger.error(f"Failed to persist features: {e}")
            raise

    def get_latest_features_before(self, team_id: int, prediction_timestamp: datetime) -> Dict[str, float]:
        """
        Point-in-time retrieval for a single team.
        """
        if prediction_timestamp.tzinfo is None:
            raise ValueError("prediction_timestamp must be UTC aware")

        query = """
            SELECT feature_name, feature_value
            FROM (
                SELECT feature_name, feature_value, 
                       row_number() OVER (PARTITION BY feature_name ORDER BY feature_timestamp_utc DESC) as rn
                FROM feature_view
                WHERE team_id = ? AND feature_timestamp_utc < ?
            )
            WHERE rn = 1
        """
        try:
            res = self.conn.execute(query, (team_id, prediction_timestamp)).fetchall()
            return {row[0]: row[1] for row in res}
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {}

    def build_point_in_time_dataset(self, events_parquet_path: str, output_path: str, feature_names: List[str]):
        """
        Constructs a training/inference dataset using ASOF JOIN.
        """
        features_list = ", ".join([f"'{f}'" for f in feature_names])
        
        query = f"""
            WITH filtered_features AS (
                SELECT team_id, feature_name, feature_value, feature_timestamp_utc
                FROM feature_view
                WHERE feature_name IN ({features_list})
            )
            SELECT e.*, f.feature_name, f.feature_value, f.feature_timestamp_utc as actual_feature_ts
            FROM read_parquet('{events_parquet_path}') e
            ASOF LEFT JOIN filtered_features f
            ON e.team_id = f.team_id AND e.kickoff_timestamp > f.feature_timestamp_utc
        """
        
        try:
            self.conn.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)")
            logger.info(f"Point-in-time dataset built at {output_path}")
        except Exception as e:
            logger.critical(f"ASOF JOIN failed: {e}")
            raise

    def validate_temporal_consistency(self) -> bool:
        """
        Audits the store for future-dated features.
        """
        now = datetime.now(timezone.utc)
        try:
            future_features = self.conn.execute("SELECT count(*) FROM feature_view WHERE feature_timestamp_utc > ?", (now,)).fetchone()[0]
            if future_features > 0:
                logger.error(f"CRITICAL: Found {future_features} features with future timestamps.")
                return False
            logger.info("Temporal consistency check passed.")
            return True
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            return False

if __name__ == "__main__":
    try:
        store_path = Path("temp_feature_store")
        store = TemporalFeatureStore(store_path)
        logger.info("Feature Store initialized for testing.")
    except Exception as exc:
        logger.error(f"Initialization test failed: {exc}")
