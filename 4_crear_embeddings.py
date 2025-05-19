'''
import pandas as pd
import os
import json


def embeddings(carpeta):
    degree_df = pd.read_csv(os.path.join(carpeta, "Degree.csv"), header=None)
    strength_df = pd.read_csv(os.path.join(carpeta, 'Strength.csv'), header=None)
    closeness_df = pd.read_csv(os.path.join(carpeta, 'ClosenessCentrality.csv'), header=None)
    local_eff_df = pd.read_csv(os.path.join(carpeta, 'LocalEfficiency.csv'), header=None)
    betweenness_df = pd.read_csv(os.path.join(carpeta, "BetweennessCentrality.csv"), header=None)
    pagerank_df = pd.read_csv(os.path.join(carpeta, 'PageRank.csv'), header=None)
    clustering_df = pd.read_csv(os.path.join(carpeta, 'Clustering.csv'), header=None)

    degree_mean = degree_df.mean(axis=0) 
    strength_mean = strength_df.mean(axis=0)
    closeness_mean = closeness_df.mean(axis=0)
    local_eff_mean = local_eff_df.mean(axis=0)
    betweenness_mean = betweenness_df.mean(axis=0) 
    pagerank_mean = pagerank_df.mean(axis=0)
    clustering_mean = clustering_df.mean(axis=0)

    embedding = pd.DataFrame({
        'degree': degree_mean,
        'strength': strength_mean,
        'closeness_centrality': closeness_mean,
        'local_efficiency': local_eff_mean,
        'betweenness_centrality': betweenness_mean,
        'pagerank': pagerank_mean,
        'clustering': clustering_mean
    })
    return embedding

embeddings_multilayer = embeddings('/Users/aina/Desktop/TFG/codi/metriques/multilayer')
embeddings_multiplex = embeddings('/Users/aina/Desktop/TFG/codi/metriques/multiplex')
embeddings_monoplex = embeddings('/Users/aina/Desktop/TFG/codi/metriques/monoplex')

embeddings_multilayer.to_csv('embeddings/embeddings_multilayer.csv', index=False)
embeddings_multiplex.to_csv('embeddings/embeddings_multiplex.csv', index=False)
embeddings_monoplex.to_csv('embeddings/embeddings_monoplex.csv', index=False)
'''

import numpy as np
import pandas as pd
import os

# Ruta als 7 arxius CSV (ordre: una metrica per fitxer)
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
    metrics.append(df.to_numpy())  # Shape: [270, 76]

# Stack por la última dimensión: resultado [270, 76, 7]
x_all = np.stack(metrics, axis=-1)

# Guardar
np.save('/Users/aina/Desktop/TFG/codi/embeddings/embeddings_multiplex', x_all)