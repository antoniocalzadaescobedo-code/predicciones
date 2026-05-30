#!/usr/bin/env python3
"""
Actualizar CSV de amistosos con nuevos partidos
"""

import pandas as pd

# Leer CSV existente
import os
file_path = os.path.join("data", "amistosos_reales.csv")
df_existing = pd.read_csv(file_path)
print(f"CSV existente: {len(df_existing)} partidos")

# Nuevos partidos proporcionados
new_matches = [
    ["2026-05-27","Cyprus","Greece"],
    ["2026-05-27","Greece","Qatar"],
    ["2026-05-27","Jamaica","India"],
    ["2026-05-28","Greece","England"],
    ["2026-05-28","Egypt","Russia"],
    ["2026-05-28","Republic of Ireland","Qatar"],
    ["2026-05-29","Iran","Gambia"],
    ["2026-05-29","South Africa","Nicaragua"],
    ["2026-05-29","Andorra","Iraq"],
    ["2026-05-29","Bosnia and Herzegovina","North Macedonia"],
    ["2026-05-30","Scotland","Curacao"],
    ["2026-05-30","Croatia","Greece"],
    ["2026-05-30","Ecuador","Saudi Arabia"],
    ["2026-05-30","South Korea","Trinidad and Tobago"],
    ["2026-05-30","Mexico","Australia"],
    ["2026-05-31","Japan","Iceland"],
    ["2026-05-31","Singapore","Mongolia"],
    ["2026-05-31","Switzerland","Jordan"],
    ["2026-05-31","Cape Verde","Serbia"],
    ["2026-05-31","Czech Republic","Kosovo"],
    ["2026-05-31","Cyprus","England"],
    ["2026-05-31","Mexico","England"],
    ["2026-05-31","Poland","Ukraine"],
    ["2026-05-31","Germany","Finland"],
    ["2026-05-31","United States","Senegal"],
    ["2026-05-31","Brazil","Panama"],
    ["2026-06-01","Tajikistan","Palestine"],
    ["2026-06-01","Slovakia","Malta"],
    ["2026-06-01","Bulgaria","Montenegro"],
    ["2026-06-01","Turkey","North Macedonia"],
    ["2026-06-01","Norway","Sweden"],
    ["2026-06-01","Austria","Tunisia"],
    ["2026-06-01","Colombia","Costa Rica"],
    ["2026-06-01","Canada","Uzbekistan"],
    ["2026-06-02","Croatia","Belgium"],
    ["2026-06-02","Georgia","Romania"],
    ["2026-06-02","Morocco","Madagascar"],
    ["2026-06-02","Wales","Ghana"],
    ["2026-06-02","Haiti","New Zealand"],
    ["2026-06-03","Philippines","Guam"],
    ["2026-06-03","Kyrgyzstan","Kenya"],
    ["2026-06-03","Croatia","Qatar"],
    ["2026-06-03","Gibraltar","British Virgin Islands"],
    ["2026-06-03","Denmark","DR Congo"],
    ["2026-06-03","Albania","Israel"],
    ["2026-06-03","Portugal","Northern Ireland"],
    ["2026-06-03","Luxembourg","Italy"],
    ["2026-06-03","Poland","Nigeria"],
    ["2026-06-03","Netherlands","Algeria"],
    ["2026-06-03","Panama","Dominican Republic"],
    ["2026-06-03","South Korea","El Salvador"],
    ["2026-06-04","Cambodia","Bhutan"],
    ["2026-06-04","Slovenia","Cyprus"],
    ["2026-06-04","Moldova","Malta"],
    ["2026-06-04","Burundi","Equatorial Guinea"],
    ["2026-06-04","Northern Ireland","Guinea"],
    ["2026-06-04","Sweden","Greece"],
    ["2026-06-04","Andorra","Liechtenstein"],
    ["2026-06-04","Spain","Iraq"],
    ["2026-06-04","France","Ivory Coast"],
    ["2026-06-04","Guatemala","Czech Republic"],
    ["2026-06-04","Mexico","Serbia"],
    ["2026-06-05","Singapore","China"],
    ["2026-06-05","Angola","Botswana"],
    ["2026-06-05","Tanzania","Uganda"],
    ["2026-06-05","Niger","Mauritania"],
    ["2026-06-05","Hong Kong","Mongolia"],
    ["2026-06-05","Central African Republic","Togo"],
    ["2026-06-05","Thailand","Kuwait"],
    ["2026-06-05","Indonesia","Oman"],
    ["2026-06-05","Georgia","Bahrain"],
    ["2026-06-05","Belarus","Syria"],
    ["2026-06-05","Slovakia","Montenegro"],
    ["2026-06-05","Moldova","Bulgaria"],
    ["2026-06-05","Russia","Burkina Faso"],
    ["2026-06-05","San Marino","Bangladesh"],
    ["2026-06-05","Hungary","Finland"],
    ["2026-06-05","Azerbaijan","Malta"],
    ["2026-06-05","Paraguay","Nicaragua"],
    ["2026-06-05","Puerto Rico","Saudi Arabia"],
    ["2026-06-05","Canada","Republic of Ireland"],
    ["2026-06-05","Haiti","Peru"],
    ["2026-06-06","Thailand","South Korea"],
    ["2026-06-06","Myanmar","Guam"],
    ["2026-06-06","Ethiopia","Malawi"],
    ["2026-06-06","Belgium","Tunisia"],
    ["2026-06-06","Palestine","Kenya"],
    ["2026-06-06","Croatia","Republic of Ireland"],
    ["2026-06-06","Comoros","Rwanda"],
    ["2026-06-06","Gibraltar","Cayman Islands"],
    ["2026-06-06","Romania","Wales"],
    ["2026-06-06","Portugal","Chile"],
    ["2026-06-06","Albania","Luxembourg"],
    ["2026-06-06","United States","Germany"],
    ["2026-06-06","Australia","Switzerland"],
    ["2026-06-06","Panama","Bosnia and Herzegovina"],
    ["2026-06-06","Bolivia","Scotland"],
    ["2026-06-06","England","New Zealand"],
    ["2026-06-06","Brazil","Egypt"],
    ["2026-06-06","Cape Verde","Bermuda"],
    ["2026-06-06","Qatar","El Salvador"],
    ["2026-06-06","Venezuela","Turkey"],
    ["2026-06-06","Curacao","Aruba"],
    ["2026-06-06","Argentina","Honduras"],
    ["2026-06-07","Liechtenstein","Cyprus"],
    ["2026-06-07","Denmark","Ukraine"],
    ["2026-06-07","Kosovo","Andorra"],
    ["2026-06-07","Croatia","Slovenia"],
    ["2026-06-07","Greece","Italy"],
    ["2026-06-07","Morocco","Norway"],
    ["2026-06-07","Ecuador","Guatemala"],
    ["2026-06-07","Colombia","Jordan"],
    ["2026-06-08","Uganda","Madagascar"],
    ["2026-06-08","Netherlands","Uzbekistan"],
    ["2026-06-08","France","Northern Ireland"],
    ["2026-06-08","Peru","Spain"],
    ["2026-06-09","Cambodia","Hong Kong"],
    ["2026-06-09","China","Thailand"],
    ["2026-06-09","Thailand","United Arab Emirates"],
    ["2026-06-09","Philippines","Myanmar"],
    ["2026-06-09","Oman","Kuwait"],
    ["2026-06-09","Angola","Central African Republic"],
    ["2026-06-09","Ethiopia","Malawi"],
    ["2026-06-09","Mauritania","Liberia"],
    ["2026-06-09","Botswana","Niger"],
    ["2026-06-09","Togo","Benin"],
    ["2026-06-09","Indonesia","Mozambique"],
    ["2026-06-09","Bahrain","Syria"],
    ["2026-06-09","Kyrgyzstan","Palestine"],
    ["2026-06-09","Equatorial Guinea","Comoros"],
    ["2026-06-09","Russia","Trinidad and Tobago"],
    ["2026-06-09","DR Congo","Chile"],
    ["2026-06-09","Belarus","Burkina Faso"],
    ["2026-06-09","Hungary","Kazakhstan"],
    ["2026-06-09","San Marino","Azerbaijan"],
    ["2026-06-09","Iraq","Venezuela"],
    ["2026-06-09","Saudi Arabia","Senegal"],
    ["2026-06-09","Iceland","Argentina"],
    ["2026-06-10","Portugal","Nigeria"],
    ["2026-06-10","England","Costa Rica"],
    ["2026-06-10","Austria","Guatemala"]
]

df_new = pd.DataFrame(new_matches, columns=["date","home_team","away_team"])
df_new["competition"] = "International Friendly"

print(f"Nuevos partidos: {len(df_new)}")

# Combinar y eliminar duplicados
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=["date","home_team","away_team"], keep="first")
df_combined = df_combined.sort_values("date").reset_index(drop=True)

print(f"Total después de eliminar duplicados: {len(df_combined)}")

# Guardar
import os
file_path = os.path.join("data", "amistosos_reales.csv")
df_combined.to_csv(file_path, index=False, encoding="utf-8-sig")
print("✅ CSV actualizado guardado")

print(f"\nPrimeros 10 partidos:")
print(df_combined.head(10))
