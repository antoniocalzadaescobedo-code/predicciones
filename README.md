# 🏆 AC 2026 Predictor

Sistema profesional de predicción probabilística para fútbol internacional y Mundial FIFA 2026.

Desarrollado por Antonio Calzada.

---

# 🎯 Descripción

AC 2026 Predictor es una plataforma de predicción futbolística basada en Machine Learning probabilístico, ratings ELO y modelado calibrado de resultados.

El sistema fue diseñado específicamente para escenarios de baja frecuencia y alta varianza como selecciones nacionales, priorizando:

* calibración probabilística,
* estabilidad operacional,
* validación temporal,
* y robustez out-of-sample.

La arquitectura integra pipelines causales auditables, validación anti-leakage y monitoreo de drift en producción.

---

# 🧠 Arquitectura del Modelo

## Modelo Principal

* Gradient Boosting Multiclase (GBM)
* Calibración Isotónica
* Integración Poisson Log-Lineal para ajustes de squad
* Predicción probabilística calibrada (1X2)

## Validación

* 49,288 partidos internacionales históricos
* Temporal Cross Validation
* Purged Time-Series Validation
* Embargo temporal anti-leakage
* Backtesting histórico rolling-window

## Métricas Certificadas

| Métrica           | Resultado |
| ----------------- | --------- |
| Accuracy          | 57.5%     |
| Brier Score       | 0.4158    |
| Log Loss          | 0.5781    |
| ECE               | 0.0185    |
| Calibration Slope | 0.992     |

---

# ⚙️ Features del Modelo

## Features Base

* ELO Rating
* Forma reciente
* Historial H2H
* Campo neutral
* Strength of Schedule (SOS)
* Diferencial ELO
* Contexto competitivo

## Squad Features Certificadas

Tras validación experimental y backtesting histórico:

| Feature                 | Estado       |
| ----------------------- | ------------ |
| continuity_index        | APPROVED     |
| defenders_continuity    | APPROVED     |
| midfielders_continuity  | EXPERIMENTAL |
| forwards_continuity     | REJECTED     |
| announcement_lead_hours | REJECTED     |
| squad_size_delta        | REJECTED     |

Solo las features aprobadas pueden influir en predicciones live.

---

# 🔬 Infraestructura Temporal

El sistema implementa arquitectura point-in-time correct:

* UTC-aware timestamps
* Provenance tracking
* SQLite ACID persistence
* DuckDB + Parquet feature store
* ASOF JOIN temporal
* Validación anti-leakage
* Replay batch processing
* Auditoría causal

Principio operacional:

> ningún partido puede usar información publicada después del kickoff.

---

# 🛡️ Producción y Fiabilidad

## Guardrails Automáticos

El sistema desactiva automáticamente el uplift de squad si detecta:

* PSI > 0.20
* drift de calibración
* features corruptas
* NaN probabilities
* leakage temporal
* degradación estadística

## Fallback Seguro

Ante cualquier inconsistencia:

* el sistema vuelve automáticamente al baseline,
* preservando estabilidad y calibración.

## Observabilidad

Monitoreo live de:

* drift estadístico,
* calibration slope,
* fallback rate,
* latency,
* feature corruption,
* shadow/live divergence,
* incident registry.

---

# 📅 Mundial FIFA 2026

El sistema utiliza el calendario oficial del Mundial 2026:

* 48 selecciones
* 104 partidos
* 16 sedes
* 3 países anfitriones
* 11 junio — 19 julio 2026

Incluye:

* fase de grupos,
* eliminatorias,
* amistosos internacionales,
* análisis inline por partido.

---

# ✨ Funcionalidades

## 🔮 Match Predictor

Predicciones probabilísticas calibradas:

* victoria local,
* empate,
* victoria visitante.

Incluye:

* confidence metrics,
* differential analysis,
* value detection,
* contextual adjustments.

## 📅 Smart Calendar

Calendario integrado con:

* partidos oficiales,
* amistosos,
* análisis inline,
* predicciones rápidas.

## 📊 Probabilistic Analysis

Visualización de:

* calibración,
* probabilidades,
* edge probabilístico,
* comportamiento histórico del modelo.

## 🛡️ Production Monitoring

Dashboard operacional con:

* drift monitoring,
* guardrails,
* incidents,
* calibration tracking,
* system health.

---

# 📂 Estructura del Proyecto

```text
C:\Proyecto_FIFA

├── app_streamlit.py
├── requirements.txt
│
├── src
│   ├── experimental
│   │   ├── poisson_squad_uplift.py
│   │   ├── temporal_cross_validation.py
│   │   ├── feature_ablation.py
│   │   ├── historical_backtest_runner.py
│   │   └── uplift_evaluation.py
│   │
│   ├── production
│   │   ├── squad_uplift_integration.py
│   │   ├── feature_registry.py
│   │   ├── live_observability.py
│   │   ├── live_guardrails.py
│   │   ├── live_drift_analysis.py
│   │   ├── incident_registry.py
│   │   ├── shadow_analyzer.py
│   │   └── operational_dashboard.py
│   │
│   └── squads
│       ├── historical_squad_fetcher.py
│       ├── feature_batch_processor.py
│       ├── feature_dataset_builder.py
│       └── batch_ingestion_pipeline.py
│
├── data
├── reports
└── tests
```

---

# 🚀 Instalación

## Requisitos

* Python 3.9+
* pip

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecutar aplicación

```bash
streamlit run app_streamlit.py
```

---

# 🔒 Principios del Sistema

El proyecto fue construido bajo cinco principios:

1. Probabilidades calibradas > predicciones agresivas
2. Temporal correctness > feature richness
3. System stability > prediction uplift
4. Reproducibilidad > complejidad experimental
5. Fail-safe baseline > modelos frágiles

---

# 📜 Estado del Proyecto

## Certificación

✅ Production Ready
✅ Anti-Leakage Certified
✅ Drift Monitoring Active
✅ Shadow Deployment Validated
✅ Operational Guardrails Enabled

Arquitectura congelada para Mundial FIFA 2026.

---

# 📄 Licencia

MIT License

Copyright (c) Antonio Calzada
