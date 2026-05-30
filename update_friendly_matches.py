#!/usr/bin/env python3
"""
Script para forzar actualización de partidos amistosos
"""
from live_updater import LiveFormUpdater

# Forzar actualización ignorando caché
updater = LiveFormUpdater()
updater.CACHE_HOURS = 0  # Desactivar caché

result = updater.run_auto_update(days_back=30)

print(f"Estado: {result['status']}")
print(f"Amistosos procesados: {result.get('matches_processed', 0)}")
print(f"Última actualización: {result.get('last_update', 'N/A')}")
