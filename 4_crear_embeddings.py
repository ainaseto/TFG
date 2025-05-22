
import numpy as np
import pandas as pd
import os

csv_files = [
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/Degree.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/Strength.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/ClosenessCentrality.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/LocalEfficiency.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/BetweennessCentrality.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/PageRank.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/Clustering.csv"
]

metrics = []
for file in csv_files:
    df = pd.read_csv(file, header=None)
    assert df.shape == (270, 76), f"{file} té forma incorrecta: {df.shape}"
    metrics.append(df.to_numpy())

x_all = np.stack(metrics, axis=-1)

# Guardar
np.save('/Users/aina/Desktop/TFG/codi/embeddings/embeddings_multiplex', x_all)
