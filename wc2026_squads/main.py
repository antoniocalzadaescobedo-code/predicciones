#!/usr/bin/env python3
# wc2026_squads/main.py

import logging
import sys
import os

# 🔥 DEFINIR LOGGER PRIMERO (antes de cualquier import personalizado)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/squads_scraper.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Fix Unicode encoding for Windows console
if sys.platform == "win32":
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace')
logger = logging.getLogger(__name__)  # ← ESTO ES CRÍTICO

# Imports personalizados (después de configurar logging)
from scraper import scrape_all_squads
from cleaner import clean_and_normalize
from exporter import export_datasets, export_summary_report
from monitor import run_monitor_cycle, start_monitor

def run_full_pipeline():
    """Ejecuta pipeline completo una vez"""
    logger.info("Iniciando pipeline de convocatorias FIFA 2026...")
    
    # 1. Scrapear
    df_raw = scrape_all_squads()
    if df_raw.empty:
        logger.error("No se encontraron datos. Verificar fuentes o fallback CSV.")
        return False
    
    # 2. Limpiar
    df_clean = clean_and_normalize(df_raw)
    
    # 3. Exportar
    export_datasets(df_clean)
    
    # 4. Resumen
    summary = export_summary_report(df_clean)
    logger.info(f"Pipeline completado:\n{summary}")
    
    return True

def main():
    if "--monitor" in sys.argv:
        # Modo monitor continuo
        interval = 6
        if "--interval" in sys.argv:
            try:
                idx = sys.argv.index("--interval")
                interval = int(sys.argv[idx + 1])
            except:
                pass
        start_monitor(check_interval_hours=interval)
    else:
        # Ejecución única
        success = run_full_pipeline()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
