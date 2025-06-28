import numpy as np
import random
import metriques_multiplex as multiplex

adj_all = np.load('/Users/aina/Desktop/TFG/codi/xarxes/multiplex.npy') # (270, 228, 228)
y_all = np.load("/Users/aina/Desktop/TFG/codi/etiquetes/etiquetes.npy") # (270,)

num_classe0 = np.where(y_all == 0)[0]
num_sintetics = 199 - len(num_classe0)

print(f"Actuals de classe 0: {len(num_classe0)}")
print(f"Calen {num_sintetics} gràfics nous")

matrius_adj = []
classes = []

for _ in range(num_sintetics):
    i1, i2 = random.sample(list(num_classe0), 2)
    alpha = random.uniform(0, 1)
    new_adj = adj_all[i1] * (1 - alpha) + adj_all[i2] * alpha
    matrius_adj.append(new_adj)
    classes.append(0)

matrius_adj = np.stack(matrius_adj)  
classes = np.array(classes)  
print(f"Nou shape de adj: {matrius_adj.shape}")
print(f"Nou shape de y: {classes.shape}")

lst_metrics = ["Strength", "Degree", "Clustering", "ClosenessCentrality", "BetweennessCentrality", "PageRank", "LocalEfficiency"]
for metrica in lst_metrics:
    multiplex.get_metric_values(matrius_adj,  metrics_folder = "/Users/aina/Desktop/TFG/codi/metriques/smote/multiplex", METRIC=metrica)

np.save("xarxes/smote/multiplex.npy", matrius_adj)
np.save("etiquetes/etiquetes_finals.npy", classes)