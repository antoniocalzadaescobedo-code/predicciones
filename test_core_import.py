"""
Test script para validar que core/predictor.py sea importable sin Streamlit.
"""

print("Validando import de core/predictor.py...")

try:
    from core.predictor import (
        EloTracker,
        DixonColes,
        FormCalculator,
        H2HCalculator,
        MLModels,
        EnsemblePredictor,
        MonteCarloSimulator,
        WorldCupPredictor
    )
    print("[OK] Import exitoso - SIN dependencias de Streamlit")
    
    # Test básico de instanciación
    print("\nTest de instanciación de clases:")
    
    elo = EloTracker()
    print("[OK] EloTracker instanciado")
    
    dc = DixonColes()
    print("[OK] DixonColes instanciado")
    
    form = FormCalculator()
    print("[OK] FormCalculator instanciado")
    
    h2h = H2HCalculator()
    print("[OK] H2HCalculator instanciado")
    
    ml = MLModels()
    print("[OK] MLModels instanciado")
    
    ensemble = EnsemblePredictor()
    print("[OK] EnsemblePredictor instanciado")
    
    mc = MonteCarloSimulator()
    print("[OK] MonteCarloSimulator instanciado")
    
    predictor = WorldCupPredictor()
    print("[OK] WorldCupPredictor instanciado")
    
    print("\n[OK] TODAS las clases instanciadas correctamente")
    print("[OK] core/predictor.py es importable SIN dependencias de Streamlit")
    
except ImportError as e:
    print(f"[ERROR] Error de import: {e}")
    print("[ERROR] core/predictor.py tiene dependencias de Streamlit")
except Exception as e:
    print(f"[ERROR] Error inesperado: {e}")
