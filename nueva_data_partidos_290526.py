import pandas as pd

# URL directa al archivo CSV
url = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"

# Cargar el dataset
df = pd.read_csv(url)

# Ver las primeras filas para explorar la estructura
print(df.head())

# Ver la información del dataset (tipos de datos, valores nulos)
print(df.info())