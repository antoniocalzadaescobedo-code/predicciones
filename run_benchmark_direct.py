"""
Script para ejecutar el benchmark directamente sin subprocess.
"""
import sys
sys.path.insert(0, 'C:\\Proyecto_FIFA')

print("Ejecutando benchmark directamente...")
with open('C:\\Proyecto_FIFA\\scientific_benchmark.py', 'r', encoding='utf-8') as f:
    exec(f.read())
