# SQUAD TEMPORAL FEATURES - ARQUITECTURA MÍNIMA Y CIENTÍFICAMENTE VÁLIDA

## ANÁLISIS DE ARQUITECTURA EXISTENTE

**Stack Actual:**
- GBM-based predictor con calibración isotónica
- Sistema ELO para ratings de equipos
- Validación temporal rolling con purged time-series CV
- Métricas de calibración (Brier, ECE, slope)
- Bootstrap determinista para intervalos de confianza
- Detección de drift temporal
- Streamlit UI

**Datos de Squads Disponibles:**
- country, player, position, club, age, jersey_number
- squad_status (Final/Preliminary)
- announcement_date (timestamp de convocatoria)
- group, confederation

**Limitaciones Críticas:**
- NO xG individuales
- NO lesiones
- NO goles históricos por jugador
- NO caps internacionales
- NO valor de mercado
- NO minutos jugados

## ARQUITECTURA PROPUESTA (INCREMENTAL)

### FASE 1: PERSISTENCIA TEMPORAL DE CONVOCATORIAS

**Objetivo:** Crear snapshots históricos auditables sin modificar arquitectura existente.

**Schema SQLite (squad_history.db):**
```sql
CREATE TABLE squad_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id TEXT NOT NULL,
    team_name TEXT NOT NULL,
    fetch_timestamp_utc TEXT NOT NULL,  -- ISO 8601
    players_json TEXT NOT NULL,         -- JSON completo de convocatoria
    squad_status TEXT NOT NULL,          -- Final/Preliminary
    announcement_date TEXT,              -- Si disponible
    INDEX idx_team_timestamp (team_id, fetch_timestamp_utc)
);
```

**Por qué SQLite:**
- Zero-config, compatible con stack existente
- Soporta timestamps y JSON nativamente
- No requiere infraestructura adicional
- Auditabilidad completa (todos los snapshots)

**Riesgos:**
- Crecimiento de DB si snapshots frecuentes
- Mitigación: política de retención (ej: 1 snapshot por día por equipo)

### FASE 2: FEATURE EXTRACTION TEMPORAL

**Features Matemáticamente Definidas:**

#### 1. Continuity Index (Jaccard Similarity)

**Definición:**
```
C_t(team) = |S_t ∩ S_{t-1}| / |S_t ∪ S_{t-1}|
```

Donde:
- S_t: conjunto de jugadores convocados en tiempo t
- S_{t-1}: conjunto de jugadores convocados en tiempo t-1
- t: timestamp de snapshot más reciente antes del partido

**Interpretación:**
- C_t = 1.0: convocatoria idéntica (máxima continuidad)
- C_t = 0.0: convocatoria completamente diferente

**Validación Temporal:**
```
feature_timestamp = fetch_timestamp_utc de S_t
constraint: feature_timestamp < kickoff_timestamp
```

**Hipótesis Causal:**
- Alta continuidad → cohesión táctica → mejor performance
- Baja continuidad → rotación → incertidumbre táctica

**Crítica Científica:**
- Débil: continuidad no causalmente garantiza performance
- Riesgo: ruido si convocatorias cambian por factores externos (lesiones, sanciones)

#### 2. Positional Continuity

**Definición:**
```
C_t^def(team) = |S_t^def ∩ S_{t-1}^def| / |S_t^def ∪ S_{t-1}^def|
C_t^mid(team) = |S_t^mid ∩ S_{t-1}^mid| / |S_t^mid ∪ S_{t-1}^mid|
C_t^fwd(team) = |S_t^fwd ∩ S_{t-1}^fwd| / |S_t^fwd ∪ S_{t-1}^fwd|
```

Donde S_t^pos es el subconjunto de jugadores en posición pos.

**Hipótesis Causal:**
- Continuidad defensiva → estabilidad defensiva
- Continuidad ofensiva → sincronización ofensiva

**Crítica Científica:**
- Más débil aún: subconjuntos más pequeños → mayor varianza
- Riesgo: overfitting a patrones espurios

#### 3. Squad Size Stability

**Definición:**
```
SS_t(team) = |S_t|
ΔSS_t(team) = |S_t| - |S_{t-1}|
```

**Hipótesis Causal:**
- Cambios bruscos en tamaño → inestabilidad organizacional

**Crítica Científica:**
- Muy débil: tamaño de convocatoria no es causalmente fuerte
- Riesgo: ruido puro, probablemente sin señal predictiva

#### 4. Announcement Lead Time

**Definición:**
```
ALT_t(team) = kickoff_timestamp - announcement_timestamp
```

**Hipótesis Causal:**
- Mayor lead time → más preparación → mejor performance

**Crítica Científica:**
- Moderadamente débil: preparación no garantiza performance
- Riesgo: confounding con calidad de oposición

### FASE 3: INTEGRACIÓN EXPERIMENTAL

**Estrategia de Integración:**

NO modificar λ del Poisson directamente.

**Enfoque Recomendado: Meta-calibration Layer**

```
λ_adjusted = λ_base * exp(β * feature)
```

Donde β se aprende históricamente vía validación temporal.

**Alternativa: Feature Engineering para GBM**

Agregar features temporales al input del GBM existente:

```python
features = [
    'elo_home', 'elo_away', 'elo_diff',
    'continuity_home', 'continuity_away',
    'def_continuity_home', 'def_continuity_away',
    'announcement_lead_home', 'announcement_lead_away'
]
```

**Ventaja:**
- Compatible con stack GBM existente
- El modelo aprende pesos automáticamente
- No requiere modificación de lógica de predicción

### FASE 4: VALIDACIÓN ESTADÍSTICA OBLIGATORIA

**Antes de Producción:**

1. **Validación Temporal Rolling:**
   - Purged time-series CV
   - Rolling windows (ej: 6 meses)
   - Gap temporal entre train/test (ej: 7 días)

2. **Métricas Exigidas:**
   - Brier Score: mejora ≥ 0.005 (absoluta)
   - LogLoss: mejora ≥ 0.01 (absoluta)
   - Calibration Slope: 0.95-1.05 rango
   - Bootstrap CI: no overlap con baseline

3. **Estabilidad Temporal de Feature:**
   - Drift detection (KS test, PSI)
   - Varianza temporal aceptable
   - No autocorrelación espuria

4. **Anti-Leakage Validation:**
   - Verificar feature_timestamp < kickoff_timestamp
   - Auditoría de timestamps en pipeline
   - Tests automatizados de causalidad temporal

**Criterio de Rechazo:**
- Feature sin uplift estadístico significativo
- Feature inestable temporalmente
- Feature introduce ruido (degrada métricas)
- Feature con leakage temporal

## IMPLEMENTACIÓN MÍNIMA

### Módulo 1: squad_temporal_persistence.py

```python
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class SquadTemporalPersistence:
    """Persistencia temporal de convocatorias con validación anti-leakage."""
    
    def __init__(self, db_path: str = "data/squad_history.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Inicializa schema SQLite."""
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
                INDEX idx_team_timestamp (team_id, fetch_timestamp_utc)
            )
        """)
        conn.commit()
        conn.close()
    
    def save_snapshot(self, team_id: str, team_name: str, 
                     players: List[Dict], squad_status: str,
                     announcement_date: Optional[str] = None) -> str:
        """
        Guarda snapshot temporal con timestamp UTC.
        
        Args:
            team_id: Identificador único del equipo
            team_name: Nombre del equipo
            players: Lista de jugadores con metadata
            squad_status: Final/Preliminary
            announcement_date: Fecha de anuncio (si disponible)
        
        Returns:
            fetch_timestamp_utc: Timestamp del snapshot
        """
        fetch_timestamp = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO squad_snapshots 
            (team_id, team_name, fetch_timestamp_utc, players_json, 
             squad_status, announcement_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (team_id, team_name, fetch_timestamp, 
              json.dumps(players), squad_status, announcement_date))
        conn.commit()
        conn.close()
        
        return fetch_timestamp
    
    def get_latest_snapshot(self, team_id: str, 
                           before_timestamp: str) -> Optional[Dict]:
        """
        Obtiene snapshot más reciente antes de un timestamp dado.
        
        Validación anti-leakage: snapshot_timestamp < before_timestamp
        
        Args:
            team_id: Identificador del equipo
            before_timestamp: Timestamp límite (kickoff)
        
        Returns:
            Dict con snapshot o None si no existe
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT team_id, team_name, fetch_timestamp_utc, players_json,
                   squad_status, announcement_date
            FROM squad_snapshots
            WHERE team_id = ? AND fetch_timestamp_utc < ?
            ORDER BY fetch_timestamp_utc DESC
            LIMIT 1
        """, (team_id, before_timestamp))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'team_id': row[0],
                'team_name': row[1],
                'fetch_timestamp_utc': row[2],
                'players': json.loads(row[3]),
                'squad_status': row[4],
                'announcement_date': row[5]
            }
        return None
    
    def get_historical_snapshots(self, team_id: str, 
                                limit: int = 10) -> List[Dict]:
        """Obtiene snapshots históricos para análisis de continuidad."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT team_id, team_name, fetch_timestamp_utc, players_json,
                   squad_status, announcement_date
            FROM squad_snapshots
            WHERE team_id = ?
            ORDER BY fetch_timestamp_utc DESC
            LIMIT ?
        """, (team_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'team_id': row[0],
            'team_name': row[1],
            'fetch_timestamp_utc': row[2],
            'players': json.loads(row[3]),
            'squad_status': row[4],
            'announcement_date': row[5]
        } for row in rows]
```

### Módulo 2: squad_temporal_features.py

```python
import numpy as np
from typing import Dict, List, Set, Optional
from datetime import datetime

class SquadTemporalFeatures:
    """Extracción de features temporales desde snapshots de convocatorias."""
    
    @staticmethod
    def jaccard_similarity(set_a: Set, set_b: Set) -> float:
        """
        Calcula similitud Jaccard entre dos conjuntos.
        
        C = |A ∩ B| / |A ∪ B|
        """
        if not set_a and not set_b:
            return 1.0  # Ambos vacíos → máxima similitud
        if not set_a or not set_b:
            return 0.0  # Uno vacío → cero similitud
        
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        return intersection / union if union > 0 else 0.0
    
    def extract_continuity_index(self, current_snapshot: Dict, 
                                previous_snapshot: Dict) -> float:
        """
        Extrae índice de continuidad entre convocatorias consecutivas.
        
        Args:
            current_snapshot: Snapshot más reciente
            previous_snapshot: Snapshot anterior
        
        Returns:
            Continuity index [0.0, 1.0]
        """
        current_players = {p['player'] for p in current_snapshot['players']}
        previous_players = {p['player'] for p in previous_snapshot['players']}
        
        return self.jaccard_similarity(current_players, previous_players)
    
    def extract_positional_continuity(self, current_snapshot: Dict,
                                     previous_snapshot: Dict,
                                     position: str) -> float:
        """
        Extrae continuidad por posición específica.
        
        Args:
            current_snapshot: Snapshot más reciente
            previous_snapshot: Snapshot anterior
            position: Posición (Defensa, Mediocampista, Delantero, Portero)
        
        Returns:
            Positional continuity index [0.0, 1.0]
        """
        current_pos = {p['player'] for p in current_snapshot['players'] 
                      if p.get('position') == position}
        previous_pos = {p['player'] for p in previous_snapshot['players'] 
                       if p.get('position') == position}
        
        return self.jaccard_similarity(current_pos, previous_pos)
    
    def extract_squad_size(self, snapshot: Dict) -> int:
        """Extrae tamaño de convocatoria."""
        return len(snapshot['players'])
    
    def extract_squad_size_delta(self, current_snapshot: Dict,
                                previous_snapshot: Dict) -> int:
        """Extrae cambio en tamaño de convocatoria."""
        return (self.extract_squad_size(current_snapshot) - 
                self.extract_squad_size(previous_snapshot))
    
    def extract_announcement_lead_time(self, snapshot: Dict,
                                      kickoff_timestamp: str) -> Optional[float]:
        """
        Extrae lead time entre anuncio y kickoff.
        
        Args:
            snapshot: Snapshot de convocatoria
            kickoff_timestamp: Timestamp del partido
        
        Returns:
            Lead time en días, o None si announcement_date no disponible
        """
        if not snapshot.get('announcement_date'):
            return None
        
        try:
            announcement = datetime.fromisoformat(snapshot['announcement_date'])
            kickoff = datetime.fromisoformat(kickoff_timestamp)
            lead_time = (kickoff - announcement).days
            return max(0, lead_time)  # Non-negative
        except (ValueError, TypeError):
            return None
    
    def extract_all_features(self, current_snapshot: Dict,
                            previous_snapshot: Optional[Dict],
                            kickoff_timestamp: str) -> Dict[str, float]:
        """
        Extrae todas las features temporales para un partido.
        
        Args:
            current_snapshot: Snapshot más reciente antes del partido
            previous_snapshot: Snapshot anterior (para continuidad)
            kickoff_timestamp: Timestamp del partido
        
        Returns:
            Dict con todas las features
        """
        features = {
            'squad_size': self.extract_squad_size(current_snapshot),
            'announcement_lead_time': self.extract_announcement_lead_time(
                current_snapshot, kickoff_timestamp
            )
        }
        
        if previous_snapshot:
            features['continuity_index'] = self.extract_continuity_index(
                current_snapshot, previous_snapshot
            )
            features['continuity_defense'] = self.extract_positional_continuity(
                current_snapshot, previous_snapshot, 'Defensa'
            )
            features['continuity_midfield'] = self.extract_positional_continuity(
                current_snapshot, previous_snapshot, 'Mediocampista'
            )
            features['continuity_forward'] = self.extract_positional_continuity(
                current_snapshot, previous_snapshot, 'Delantero'
            )
            features['squad_size_delta'] = self.extract_squad_size_delta(
                current_snapshot, previous_snapshot
            )
        else:
            # Valores default si no hay snapshot anterior
            features['continuity_index'] = 1.0  # Máxima continuidad
            features['continuity_defense'] = 1.0
            features['continuity_midfield'] = 1.0
            features['continuity_forward'] = 1.0
            features['squad_size_delta'] = 0
        
        return features
```

### Módulo 3: temporal_validation.py

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

class TemporalValidator:
    """Validación estadística de features temporales."""
    
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
    
    def test_feature_stability(self, feature_values: List[float],
                               window_size: int = 30) -> Dict:
        """
        Prueba estabilidad temporal de feature via KS test.
        
        Args:
            feature_values: Valores de feature ordenados temporalmente
            window_size: Tamaño de ventana para comparación
        
        Returns:
            Dict con resultados de test
        """
        if len(feature_values) < window_size * 2:
            return {'stable': True, 'reason': 'insufficient_data'}
        
        # Dividir en dos ventanas temporales
        early = feature_values[:window_size]
        late = feature_values[-window_size:]
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(early, late)
        
        stable = p_value > self.alpha
        
        return {
            'stable': stable,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'early_mean': np.mean(early),
            'late_mean': np.mean(late),
            'early_std': np.std(early),
            'late_std': np.std(late)
        }
    
    def test_temporal_leakage(self, feature_timestamps: List[str],
                             kickoff_timestamps: List[str]) -> Dict:
        """
        Valida que no exista leakage temporal.
        
        Args:
            feature_timestamps: Timestamps de features
            kickoff_timestamps: Timestamps de kickoffs
        
        Returns:
            Dict con resultados de validación
        """
        violations = []
        
        for feat_ts, kickoff_ts in zip(feature_timestamps, kickoff_timestamps):
            if feat_ts >= kickoff_ts:
                violations.append({
                    'feature_timestamp': feat_ts,
                    'kickoff_timestamp': kickoff_ts
                })
        
        return {
            'leakage_detected': len(violations) > 0,
            'violation_count': len(violations),
            'violations': violations[:10]  # Primeras 10 violaciones
        }
    
    def bootstrap_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                    feature_idx: int,
                                    n_bootstrap: int = 1000) -> Dict:
        """
        Bootstrap intervalos de confianza para importancia de feature.
        
        Args:
            X: Matriz de features
            y: Target
            feature_idx: Índice de feature a evaluar
            n_bootstrap: Número de iteraciones bootstrap
        
        Returns:
            Dict con intervalos de confianza
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils import resample
        
        importances = []
        
        for _ in range(n_bootstrap):
            X_boot, y_boot = resample(X, y)
            
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X_boot, y_boot)
            
            importances.append(rf.feature_importances_[feature_idx])
        
        return {
            'mean_importance': np.mean(importances),
            'std_importance': np.std(importances),
            'ci_95_lower': np.percentile(importances, 2.5),
            'ci_95_upper': np.percentile(importances, 97.5)
        }
```

### Módulo 4: squad_integration.py

```python
import pandas as pd
from typing import Dict, Optional
from squad_temporal_persistence import SquadTemporalPersistence
from squad_temporal_features import SquadTemporalFeatures

class SquadFeatureIntegrator:
    """Integración de features de squad con pipeline existente."""
    
    def __init__(self, db_path: str = "data/squad_history.db"):
        self.persistence = SquadTemporalPersistence(db_path)
        self.features = SquadTemporalFeatures()
    
    def enrich_match_features(self, match_row: pd.Series) -> Dict:
        """
        Enriquece features de un partido con datos de squad.
        
        Args:
            match_row: Fila de DataFrame con partido (debe tener date, home_team, away_team)
        
        Returns:
            Dict con features adicionales de squad
        """
        kickoff_ts = match_row['date'].isoformat()
        
        # Obtener snapshots para home y away
        home_snapshot = self.persistence.get_latest_snapshot(
            match_row['home_team'], kickoff_ts
        )
        away_snapshot = self.persistence.get_latest_snapshot(
            match_row['away_team'], kickoff_ts
        )
        
        if not home_snapshot or not away_snapshot:
            # Si no hay snapshots, retornar features neutrales
            return self._neutral_features()
        
        # Obtener snapshots anteriores para continuidad
        home_history = self.persistence.get_historical_snapshots(
            match_row['home_team'], limit=2
        )
        away_history = self.persistence.get_historical_snapshots(
            match_row['away_team'], limit=2
        )
        
        home_previous = home_history[1] if len(home_history) > 1 else None
        away_previous = away_history[1] if len(away_history) > 1 else None
        
        # Extraer features
        home_features = self.features.extract_all_features(
            home_snapshot, home_previous, kickoff_ts
        )
        away_features = self.features.extract_all_features(
            away_snapshot, away_previous, kickoff_ts
        )
        
        # Combinar features con prefijos
        enriched = {}
        for key, value in home_features.items():
            enriched[f'home_{key}'] = value
        for key, value in away_features.items():
            enriched[f'away_{key}'] = value
        
        # Features derivadas (diferencias)
        enriched['continuity_diff'] = (
            enriched['home_continuity_index'] - 
            enriched['away_continuity_index']
        )
        enriched['squad_size_diff'] = (
            enriched['home_squad_size'] - 
            enriched['away_squad_size']
        )
        
        return enriched
    
    def _neutral_features(self) -> Dict:
        """Retorna features neutrales cuando no hay datos de squad."""
        return {
            'home_squad_size': 23,
            'away_squad_size': 23,
            'home_continuity_index': 1.0,
            'away_continuity_index': 1.0,
            'home_continuity_defense': 1.0,
            'away_continuity_defense': 1.0,
            'home_continuity_midfield': 1.0,
            'away_continuity_midfield': 1.0,
            'home_continuity_forward': 1.0,
            'away_continuity_forward': 1.0,
            'home_squad_size_delta': 0,
            'away_squad_size_delta': 0,
            'home_announcement_lead_time': None,
            'away_announcement_lead_time': None,
            'continuity_diff': 0.0,
            'squad_size_diff': 0
        }
```

## ESTRATEGIA DE VALIDACIÓN

### Fase de Research Offline

1. **Backtesting Histórico:**
   - Usar datos históricos 2018-2024
   - Simular snapshots de convocatorias (si disponibles)
   - Calcular features retrospectivamente
   - Validar uplift en métricas

2. **A/B Testing Temporal:**
   - Baseline: modelo sin features de squad
   - Treatment: modelo con features de squad
   - Validación purged time-series CV
   - Comparación de Brier, LogLoss, calibration

3. **Análisis de Importancia:**
   - SHAP values para features de squad
   - Bootstrap confidence intervals
   - Eliminar features sin importancia significativa

### Criterios de Aceptación

**Uplift Mínimo:**
- Brier Score: Δ ≥ 0.005
- LogLoss: Δ ≥ 0.01
- Calibration Slope: 0.95-1.05

**Estabilidad:**
- KS test p-value > 0.05 (no drift significativo)
- Varianza temporal < 0.1 (coeficiente de variación)

**Anti-Leakage:**
- 0 violaciones de feature_timestamp < kickoff_timestamp
- Auditoría automatizada pasa

## RIESGOS Y LIMITACIONES

### Riesgos Técnicos

1. **Data Sparsity:**
   - Snapshots históricos limitados
   - Features calculadas con pocos datos
   - Mitigación: usar defaults razonables, validar estabilidad

2. **Overfitting:**
   - Features espurias con poco signal
   - Mitigación: validación temporal estricta, bootstrap

3. **Computational Overhead:**
   - Consultas SQLite adicionales
   - Mitigación: caché de snapshots, índices optimizados

### Limitaciones Científicas

1. **Causalidad Débil:**
   - Continuidad de convocatorias no causalmente fuerte
   - Riesgo: features son ruido puro
   - Mitigación: validación estadística rigurosa

2. **Confounding:**
   - Factores externos (lesiones, sanciones) no observados
   - Mitigación: análisis de sensibilidad

3. **Generalización:**
   - Features pueden no generalizar a otros torneos
   - Mitigación: validación cross-tournament

### Limitaciones de Datos

1. **Metadata Limitada:**
   - Sin xG, lesiones, caps, valor de mercado
   - Features restringidas a metadata disponible
   - Impacto: señal predictiva potencialmente débil

2. **Timestamp Quality:**
   - announcement_date puede ser incompleto
   - Mitigación: manejo robusto de missing values

## PLAN DE IMPLEMENTACIÓN INCREMENTAL

### Semana 1: Persistencia Temporal
- Implementar SquadTemporalPersistence
- Crear schema SQLite
- Script de migración de datos existentes
- Tests de anti-leakage

### Semana 2: Feature Extraction
- Implementar SquadTemporalFeatures
- Unit tests para cada feature
- Validación matemática de fórmulas
- Documentación de features

### Semana 3: Integración Offline
- Implementar SquadFeatureIntegrator
- Backtesting histórico
- Validación estadística preliminar
- Análisis de importancia

### Semana 4: Validación Rigurosa
- Purged time-series CV
- Bootstrap confidence intervals
- Drift detection
- Anti-leakage validation

### Semana 5: Integración Producción (Opcional)
- Integración con GBM existente
- A/B testing en producción
- Monitoreo de drift
- Rollback plan

## CONCLUSIÓN

**Recomendación:**

Implementar FASE 1-3 (persistencia, features, validación offline) antes de cualquier integración en producción.

**Criterio de Go/No-Go:**

Solo proceder a producción si:
- Uplift estadísticamente significativo en métricas clave
- Features temporalmente estables
- Zero leakage temporal
- Bootstrap CI no overlap con baseline

**Si criterios no se cumplen:**

Descartar features de squad temporal. El costo de complejidad no justifica el uplift marginal (o nulo).

**Principio Fundamental:**

No introducir features por "completitud". Solo features con señal predictiva validada estadísticamente.
