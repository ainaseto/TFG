import os
import numpy as np
import pandas as pd

carpeta_entrada = "/Users/aina/Desktop/TFG/codi/data/RS"
carpeta = "/Users/aina/Desktop/TFG/codi/data"
carpeta_salida = os.path.join(carpeta, "RS_definitiu")
os.makedirs(carpeta_salida, exist_ok=True)

for nombre_archivo in os.listdir(carpeta_entrada):
    if nombre_archivo.endswith(".csv"):
        ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
        df = pd.read_csv(ruta_entrada, header=None)
        matriz = df.values
        np.fill_diagonal(matriz, 0)
        matriz = np.abs(matriz)
        ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
        pd.DataFrame(matriz).to_csv(ruta_salida, header=False, index=False)
        print(f"Guardado en rs_definitiu: {nombre_archivo}")
