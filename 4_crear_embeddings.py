
import numpy as np
import pandas as pd
import os

csv_files = [[
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/Degree.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/Strength.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/ClosenessCentrality.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/LocalEfficiency.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/BetweennessCentrality.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/PageRank.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multilayer/Clustering.csv"],
    ["/Users/aina/Desktop/TFG/codi/metriques/multiplex/Degree.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multiplex/Strength.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multiplex/ClosenessCentrality.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multiplex/LocalEfficiency.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multiplex/BetweennessCentrality.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multiplex/PageRank.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/multiplex/Clustering.csv"],
    ["/Users/aina/Desktop/TFG/codi/metriques/monoplex/Degree.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/monoplex/Strength.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/monoplex/ClosenessCentrality.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/monoplex/LocalEfficiency.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/monoplex/BetweennessCentrality.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/monoplex/PageRank.csv",
    "/Users/aina/Desktop/TFG/codi/metriques/monoplex/Clustering.csv"]]


embeddings=[]
for fitxer in csv_files:
    metriques = []
    for file in fitxer:
        df = pd.read_csv(file, header=None)
        metriques.append(df.to_numpy())
    x_all = np.stack(metriques, axis=-1)
    embeddings.append(x_all)

np.save('/Users/aina/Desktop/TFG/codi/embeddings/embeddings_multilayer', embeddings[0])
np.save('/Users/aina/Desktop/TFG/codi/embeddings/embeddings_multiplex', embeddings[1])
np.save('/Users/aina/Desktop/TFG/codi/embeddings/embeddings_monoplex', embeddings[2])
