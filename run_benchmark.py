"""
Script para ejecutar el benchmark sin usar terminal integrada.
"""
import subprocess
import sys

print("Ejecutando scientific_benchmark.py...")
result = subprocess.run([sys.executable, "scientific_benchmark.py"], cwd="C:\\Proyecto_FIFA", capture_output=True, text=True, timeout=600)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print(f"Exit code: {result.returncode}")
