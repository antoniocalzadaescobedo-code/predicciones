# Auditoría del Código Fuente (SRC) - FIFA 2026 Predictor
================================================================

**Fecha:** 2026-05-24
**Objetivo:** Identificar problemas de leakage, splits incorrectos, y validación

---

## 🔴 CRÍTICOS - Requieren atención inmediata

### 1. NO HAY SPLIT TEMPORAL EN train_and_save_model.py
**Archivo:** `train_and_save_model.py` (líneas 7-49)

**Problema:**
```python
# Carga TODOS los datos
df = pd.read_csv(csv_path)
X = df[feature_cols].fillna(0).values
y = df['outcome'].values

# Entrena en TODO sin split temporal
predictor = FIFA2026Predictor(calibrate=True)
predictor.fit(X, y, feature_names=feature_cols)
```

**Impacto:** 
- ❌ **LEAKAGE TEMPORAL CRÍTICO** - El modelo ve el futuro
- ❌ No hay validación temporal
- ❌ Métricas reportadas son optimistas (overfitting al futuro)

**Solución requerida:**
```python
# Split temporal correcto
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

train_cutoff = df['date'].quantile(0.7)  # 70% train temporal
val_cutoff = df['date'].quantile(0.85)    # 15% val temporal

train = df[df['date'] <= train_cutoff]
val = df[(df['date'] > train_cutoff) & (df['date'] <= val_cutoff)]
test = df[df['date'] > val_cutoff]
```

---

### 2. Monte Carlo NO usa el modelo GBM entrenado
**Archivo:** `monte_carlo_wc2026.py` (líneas 138-143)

**Problema:**
```python
def run_monte_carlo(
    n_simulations: int = 1000,
    db: Optional[object] = None,
    predictor: Optional[object] = None # Ignoramos el predictor pesado para velocidad
):
    # Usamos motor rápido ELO, NO el GBM
    winner, g1, g2 = simular_partido_rapido(t1, t2, db, neutral=True)
```

**Impacto:**
- ❌ **INCONSISTENCIA** - Monte Carlo usa ELO simple, no el modelo GBM
- ❌ Probabilidades de Monte Carlo no reflejan el modelo real
- ❌ Hardcoded draw probability (0.26) no basado en datos

**Solución requerida:**
```python
# Usar el predictor real en Monte Carlo
res = predictor.predict_match(t1, t2, features, neutral=True)
probs = res['probabilities']
# Sampling basado en probabilidades del modelo
winner = np.random.choice([t1, 'draw', t2], p=[probs['away_win'], probs['draw'], probs['home_win']])
```

---

### 3. SOS Engine puede tener leakage
**Archivo:** `sos_engine.py` (líneas 18-30)

**Problema:**
```python
def _load_data(self, csv_path: str):
    self.df = pd.read_csv(csv_path)
```

**Impacto:**
- ⚠️ **POTENTIAL LEAKAGE** - Si `data/sos_2026.csv` contiene SOS calculado con información futura
- ⚠️ Delta_ELO puede incluir resultados posteriores al partido
- ⚠️ No hay verificación temporal en la carga

**Solución requerida:**
- Auditoría del archivo `data/sos_2026.csv`
- Verificar que SOS se calcula solo con información previa al partido
- Agregar timestamp checks

---

## 🟡 MEDIO - Requieren revisión

### 4. prepare_dataset.py hace walk-forward CORRECTAMENTE
**Archivo:** `prepare_dataset.py` (líneas 37-82)

**Estado:** ✅ **CORRECTO**
- ELO calculado walk-forward (líneas 46-58)
- Form calculado solo con historial previo (líneas 60-69)
- H2H calculado solo con enfrentamientos previos (líneas 71-79)

**Conclusión:** La preparación de datos es temporalmente correcta, pero el entrenamiento no lo usa.

---

### 5. evaluation.py tiene framework temporal correcto
**Archivo:** `evaluation.py` (líneas 190-199)

**Estado:** ✅ **CORRECTO**
- LiveEloTracker garantiza zero leakage
- Backtesting temporal implementado
- Framework de calibración presente

**Conclusión:** El framework de evaluación es correcto, pero no se usa en el entrenamiento principal.

---

### 6. gbm_production.py tiene ECE calculation correcta
**Archivo:** `gbm_production.py` (líneas 223-235)

**Estado:** ✅ **CORRECTO**
- ECE calculado con bins (10 bins)
- Fórmula estándar de Expected Calibration Error
- Alineación de probabilidades implementada

**Conclusión:** La implementación de ECE es correcta, pero el valor reportado (0.049) puede ser optimista por falta de split temporal.

---

## 📊 RESUMEN DE LEAKAGE

| Componente | Leakage Temporal | Estado |
|-------------|------------------|---------|
| prepare_dataset.py | ❌ NO (walk-forward correcto) | ✅ BIEN |
| train_and_save_model.py | ✅ SÍ (entrena en todo) | 🔴 MAL |
| evaluation.py | ❌ NO (framework correcto) | ✅ BIEN |
| monte_carlo_wc2026.py | ⚠️ POTENCIAL (no usa modelo) | 🟡 REVISAR |
| sos_engine.py | ⚠️ POTENCIAL (CSV no auditado) | 🟡 REVISAR |

---

## 🎯 ACCIONES REQUERIDAS (Prioridad Alta)

1. **Implementar split temporal en train_and_save_model.py**
   - Train: 2014-2022
   - Val: 2023
   - Test: 2024-2025

2. **Integrar predictor GBM en Monte Carlo**
   - Reemplazar `simular_partido_rapido` con `predictor.predict_match`
   - Usar probabilidades del modelo para sampling

3. **Auditar data/sos_2026.csv**
   - Verificar timestamps
   - Confirmar cálculo walk-forward
   - Validar no leakage en Delta_ELO

4. **Ejecutar evaluación con split temporal**
   - Usar framework de evaluation.py
   - Reportar métricas en test set temporal
   - Comparar vs baseline (ELO puro, always-home)

---

## 📈 MÉTRICAS ESPERADAS (Con split temporal correcto)

Para fútbol multinomial serio:

| Métrica | Actual (Reportado) | Esperado (Realista) |
|---------|-------------------|---------------------|
| Accuracy | 59.1% | 50-55% |
| LogLoss | 0.910 | 0.95-1.05 |
| ECE | 0.049 | 0.05-0.08 |
| Brier | ? | 0.20-0.25 |

**Nota:** Las métricas actuales pueden ser optimistas por falta de validación temporal.
