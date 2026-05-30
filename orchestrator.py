#!/usr/bin/env python3
"""
Orchestrator Principal - Nivel Profesional
Automatiza la actualización de Resultados (ELO) y Convocatorias (Squads).
"""

import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd

# Agregar wc2026_squads al path para imports relativos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "wc2026_squads"))

# Asegurar carpetas ANTES de configurar logging
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Configurar Logging Profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/orchestrator.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("[INFO] INICIANDO PIPELINE DE DATOS - " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    success_count = 0
    
    # ---------------------------------------------------------
    # 1. ACTUALIZACIÓN DE RESULTADOS (ELO & LIVE FORM)
    # ---------------------------------------------------------
    try:
        logger.info("[INFO] 1. Actualizando Resultados y ELO...")
        from live_updater import LiveFormUpdater
        
        updater = LiveFormUpdater()
        result = updater.run_auto_update(days_back=7)
        
        if result["status"] in ["success", "cached"]:
            logger.info(f"[OK] ELO actualizado. Estado: {result['status']}")
            success_count += 1
        else:
            logger.warning(f"[WARNING] ELO no actualizado: {result['status']}")
            
    except Exception as e:
        logger.error(f"[ERROR] Error en ELO Update: {e}")

    # ---------------------------------------------------------
    # 2. ACTUALIZACIÓN DE CONVOCATORIAS (SQUADS)
    # ---------------------------------------------------------
    try:
        logger.info("[INFO] 2. Actualizando Convocatorias (Squads)...")
        
        # Cambiar al directorio wc2026_squads para imports relativos
        original_dir = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), "wc2026_squads"))
        
        try:
            from main import run_full_pipeline
            squads_success = run_full_pipeline()
            if squads_success:
                logger.info("[OK] Squads actualizados exitosamente")
                success_count += 1
            else:
                logger.warning("[WARNING] Squads no se actualizaron (posiblemente sin cambios)")
        finally:
            # Volver al directorio original
            os.chdir(original_dir)
            
    except Exception as e:
        logger.error(f"[ERROR] Error en Squads Update: {e}")

    # ---------------------------------------------------------
    # 3. VALIDACIÓN FINAL DE INTEGRIDAD
    # ---------------------------------------------------------
    logger.info("[INFO] 3. Validando integridad de datos...")
    
    elo_file = "elo_actualizado.json"
    squads_file = "wc2026_squads/data/world_cup_2026_players.csv"
    
    checks = {
        "ELO DB": os.path.exists(elo_file),
        "Squads CSV": os.path.exists(squads_file)
    }
    
    for check, status in checks.items():
        logger.info(f"   {'[OK]' if status else '[FAIL]'} {check} presente")
    
    logger.info(f"[INFO] PIPELINE COMPLETADO. Exitos: {success_count}/2")
    return success_count

if __name__ == "__main__":
    # Si se pasa --daemon, se ejecuta infinitamente cada 6 horas
    if "--daemon" in sys.argv:
        logger.info("[INFO] Iniciando modo DAEMON (Automatizacion continua)...")
        while True:
            try:
                run_pipeline()
                logger.info("[INFO] Esperando 6 horas para el siguiente ciclo...")
                time.sleep(6 * 3600)
            except KeyboardInterrupt:
                logger.info("[INFO] Detenido por usuario.")
                break
            except Exception as e:
                logger.error(f"[ERROR] Error fatal en daemon: {e}. Reiniciando en 1 hora.")
                time.sleep(3600)
    else:
        # Ejecución única (ideal para Task Scheduler o GitHub Actions)
        run_pipeline()
