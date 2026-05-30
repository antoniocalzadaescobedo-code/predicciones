# Instrucciones para Descargar Dataset de Kaggle

## 📥 Paso 1: Descargar Dataset de Kaggle

### 1.1 Crear cuenta en Kaggle (si no tienes una)
1. Ve a https://www.kaggle.com/
2. Haz clic en "Sign Up" o "Register"
3. Completa el registro (gratuito)

### 1.2 Descargar el Dataset
1. Ve al dataset: https://www.kaggle.com/datasets/davidcariboo/player-scores
2. O busca: "FIFA World Cup 2026 Match Data"
3. Haz clic en "Download" → "Download All" (ZIP de ~36 KB)
4. Extrae el archivo ZIP

### 1.3 Copiar archivos al proyecto
1. Extrae el contenido del ZIP
2. Copia los siguientes archivos a `C:\Proyecto_FIFA\data\oficial\`:
   - `matches.csv`
   - `teams.csv`
   - `host_cities.csv`
   - `tournament_stages.csv`

## 🌦️ Paso 2: Descargar Datos Climáticos

### 2.1 Ejecutar script de descarga de clima
```bash
cd C:\Proyecto_FIFA
python scripts/descargar_clima.py
```

Esto descargará datos climáticos para las 16 sedes del Mundial 2026 y los guardará en:
- `data/clima/clima_fusionado.csv` (todos los datos combinados)
- `data/clima/clima_[ciudad]_2024-06-01_2024-08-31.csv` (individual por sede)

## 🔗 Paso 3: Fusionar Datasets

### 3.1 Ejecutar script de fusión
```bash
cd C:\Proyecto_FIFA
python scripts/fusionar_datos.py
```

Esto creará:
- `data/dataset_maestro.csv` (DataFrame combinado con todas las features)

## 📊 Paso 4: Re-entrenar el Modelo

### 4.1 Crear script de re-entrenamiento
Crea `scripts/train_model_with_new_features.py` con el código para re-entrenar el GBM con las nuevas features (clima, ranking, stats de jugadores).

### 4.2 Ejecutar re-entrenamiento
```bash
python scripts/train_model_with_new_features.py
```

## ⚠️ Notas Importantes

### Licencias de las Fuentes
| Fuente | Licencia | Actualización | Notas |
|--------|----------|---------------|-------|
| Kaggle WC2026 | CC0 (Público) | Manual | Datos oficiales FIFA |
| FIFA Ranking | FIFA Terms | Mensual | Scraping permitido para uso personal |
| Open-Meteo | CC BY 4.0 | Automático vía API | Sin API key requerida |
| FBref / SoFIFA | Términos del sitio | Semanal | Verificar condiciones de uso |

### Estructura de Carpetas Final
```
C:\Proyecto_FIFA\
│
├── data/
│   ├── oficial/              ← Copia los CSV de Kaggle aquí
│   │   ├── matches.csv
│   │   ├── teams.csv
│   │   ├── host_cities.csv
│   │   └── tournament_stages.csv
│   │
│   ├── historico/           ← Agrega ranking FIFA aquí
│   │   └── fifa_ranking-latest.csv
│   │
│   ├── jugadores/           ← Agrega stats de jugadores aquí
│   │   └── players_stats_2024.csv
│   │
│   ├── clima/               ← Generado automáticamente por descargar_clima.py
│   │   ├── clima_fusionado.csv
│   │   └── clima_[ciudad]_*.csv
│   │
│   └── dataset_maestro.csv  ← Generado por fusionar_datos.py
│
├── scripts/
│   ├── descargar_clima.py   ← Ya creado
│   ├── fusionar_datos.py    ← Ya creado
│   └── train_model_with_new_features.py  ← Por crear
│
└── app_streamlit.py
```

## 🚀 Resumen de Comandos

```bash
# 1. Descargar dataset de Kaggle (manual)
# Ve a Kaggle → Descarga → Extrae a data/oficial/

# 2. Descargar clima
python scripts/descargar_clima.py

# 3. Fusionar datos
python scripts/fusionar_datos.py

# 4. Re-entrenar modelo (por crear)
python scripts/train_model_with_new_features.py
```

## ✅ Verificación

Después de completar los pasos, verifica:

1. **Archivos en data/oficial/**:
   ```bash
   dir data\oficial
   ```
   Deberías ver: matches.csv, teams.csv, host_cities.csv, tournament_stages.csv

2. **Archivos en data/clima/**:
   ```bash
   dir data\clima
   ```
   Deberías ver: clima_fusionado.csv + 16 CSVs individuales

3. **Dataset maestro**:
   ```bash
   dir data
   ```
   Deberías ver: dataset_maestro.csv

## 📞 Soporte

Si encuentras errores:
- Verifica que los archivos de Kaggle estén en la ubicación correcta
- Asegúrate de tener conexión a internet para descargar clima
- Revisa que las dependencias de Python estén instaladas
