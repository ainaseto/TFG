import numpy as np
import pandas as pd

pacients = pd.read_csv('/Users/aina/Desktop/TFG/codi/data/demographics.csv')
etiquetes = pacients['mstype'].to_numpy()
etiquetes_resultat = np.where(etiquetes > -1, 1, etiquetes)
vector_resultat = np.where(etiquetes_resultat == -1, 0, etiquetes_resultado)
print(vector_resultat.shape)

np.save("etiquetes/etiquetes.npy", vector_resultat)
