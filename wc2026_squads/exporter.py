# exporter.py
import pandas as pd
import os
import logging
logger = logging.getLogger(__name__)  # ← Agregar esta línea
from config import OUTPUT_DIR

def export_datasets(df: pd.DataFrame):
    """
    Exporta a 3 formatos:
    - squads.csv: Datos completos
    - players.csv: Versión ML-ready
    - squads.xlsx: Excel con múltiples hojas analíticas
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. squads.csv (datos completos para análisis general)
    squads_path = os.path.join(OUTPUT_DIR, "world_cup_2026_squads.csv")
    df.to_csv(squads_path, index=False, encoding="utf-8-sig")
    print(f"Exportado: {squads_path}")
    
    # 2. players.csv (versión ML-ready: normalizada, sin texto libre)
    from cleaner import prepare_for_ml
    df_ml = prepare_for_ml(df)
    players_path = os.path.join(OUTPUT_DIR, "world_cup_2026_players.csv")
    df_ml.to_csv(players_path, index=False, encoding="utf-8-sig")
    print(f"Exportado: {players_path}")
    
    # 3. squads.xlsx (Excel con múltiples hojas para stakeholders)
    excel_path = os.path.join(OUTPUT_DIR, "world_cup_2026_squads.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Hoja 1: Datos completos
        df.to_excel(writer, sheet_name="Squads", index=False)
        
        # Hoja 2: Resumen por grupo (conteo por posición)
        group_summary = df.groupby(["group", "position"]).size().unstack(fill_value=0)
        group_summary.to_excel(writer, sheet_name="Group_Summary")
        
        # Hoja 3: Distribución de edades por grupo
        age_stats = df.groupby("group")["age"].agg(["mean", "median", "min", "max", "std"])
        age_stats.round(1).to_excel(writer, sheet_name="Age_Distribution")
        
        # Hoja 4: Jugadores por confederación
        conf_summary = df.groupby(["confederation", "position"]).size().unstack(fill_value=0)
        conf_summary.to_excel(writer, sheet_name="Confederation_Summary")
        
        # Hoja 5: Metadatos del dataset
        metadata = pd.DataFrame({
            "metric": ["Total jugadores", "Países", "Grupos", "Confederaciones", "Fecha exportación"],
            "value": [len(df), df["country"].nunique(), df["group"].nunique(), 
                     df["confederation"].nunique(), pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")]
        })
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
    
    print(f"Exportado: {excel_path}")
    print(f"Archivos guardados en: {os.path.abspath(OUTPUT_DIR)}")
    
    return True

def export_summary_report(df: pd.DataFrame) -> str:
    """Genera resumen textual para logging/notificaciones"""
    summary = [
        f"Resumen de Convocatorias FIFA 2026",
        f"• Total jugadores: {len(df)}",
        f"• Países: {df['country'].nunique()}",
        f"• Por posición: {df['position'].value_counts().to_dict()}",
        f"• Edad promedio: {df['age'].mean():.1f} años",
        f"• Convocatorias finales: {(df['squad_status']=='Final').sum()}/{len(df)}",
    ]
    return "\n".join(summary)
