# 🌍 FIFA 2026 World Cup Squads Scraper & Analytics Pipeline

Pipeline profesional en Python/Pandas para extraer, limpiar y exportar convocatorias del Mundial 2026.

## ✨ Características

✅ **Scraping automático** desde Wikipedia (estructura estable)  
✅ **Detección de cambios**: solo exporta si hay nuevas convocatorias  
✅ **Limpieza robusta**: nombres, posiciones, edades normalizadas  
✅ **ML-Ready**: datasets sin texto libre, tipos numéricos, sin NaN  
✅ **Exportación triple**: CSV completo, CSV ML, Excel con 5 hojas analíticas  
✅ **Reintentos automáticos**: tenacity con backoff exponencial  
✅ **Fallback CSV**: funciona sin conexión si scraping falla  
✅ **Monitor continuo**: ejecuta en background cada N horas  

## 📦 Estructura de Output

```
data/
├── world_cup_2026_squads.csv # Datos completos (10 columnas)
├── world_cup_2026_players.csv # Versión ML-ready (tipos numéricos)
└── world_cup_2026_squads.xlsx # Excel con 5 hojas analíticas
```

## 🚀 Instalación

```bash
# 1. Clonar o descargar el proyecto
cd wc2026_squads

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar una vez (extracción completa)
python main.py

# 4. O modo monitor (revisa cada 6h y exporta si hay cambios)
python main.py --monitor
# O con intervalo personalizado:
python main.py --monitor --interval 12
```

## 📊 Columnas Generadas

| Columna | Descripción | Tipo |
|---------|-------------|------|
| country | País del jugador | str |
| player | Nombre limpio del jugador | str |
| position | Posición estandarizada (Portero/Defensa/Medio/Delantero) | str |
| club | Club actual (limpio) | str |
| age | Edad en años (numérico, sin NaN) | int |
| jersey_number | Dorsal (0 si no disponible) | int |
| squad_status | Preliminar / Final | str |
| announcement_date | Fecha de anuncio (YYYY-MM-DD) | str |
| group | Grupo del Mundial (A-L) | str |
| confederation | Confederación (UEFA/CONMEBOL/etc.) | str |

## 🔧 Personalización

- **Fuentes**: Editar `config.py` → `SQUAD_SOURCES` para agregar nuevas URLs
- **Mapeos**: Actualizar `POSITION_MAP` o `COUNTRY_NAME_MAP` en `cleaner.py`
- **Intervalo de monitor**: `--interval N` en línea de comandos
- **Fallback CSV**: Crear `data/squads_fallback.csv` con el formato esperado

## ⚠️ Notas Importantes

- FIFA.com usa React/Cloudflare → este proyecto usa Wikipedia como fuente principal (estructura wikitable estable y ética para scraping).
- Si FIFA publica API oficial, solo cambiar `SQUAD_SOURCES` en `config.py` y ajustar parser en `scraper.py`.
- Los datos se guardan en `data/`
