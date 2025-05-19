
import networkx as nx
import numpy as np
import Metrics as Metrics
import Functions as Functions
import os

def get_metric_values(em, METRIC=None):
    num_subjs = em.shape[0]
    num_nodes = 76
    print("Computing values for metric '{}'...".format(METRIC))
    values = np.zeros((num_subjs, num_nodes), dtype=float)
    for i in range(num_subjs):
        print("Processing subject {}...".format(i))
        A = em[i,:,:] # data 
        G = Functions.create_graph_from_AM(A) # create graph
        # compute metric values
        try:
            if METRIC=='Degree':
                temp = np.array(list(nx.degree_centrality(G).values()))

            elif METRIC=='Strength':
                temp = np.sum(A, axis=0)

            elif METRIC=='Clustering':
                temp = np.array(list(nx.clustering(G, weight='weight').values()))

            elif METRIC=='BetweennessCentrality':
                G = Functions.create_distance_graph_from_AM(A) # Use 'distance'
                Functions.report_graph_basics(G) # debug only
                temp = np.array(list(nx.betweenness_centrality(G, k=None, normalized=True, weight='weight').values()))

            elif METRIC=='ClosenessCentrality':
                G = Functions.create_distance_graph_from_AM(A)  # Use 'distance'
                temp = np.array(list(nx.closeness_centrality(G, distance='weight').values()))

            elif METRIC=='PageRank':
                temp = np.array(list(nx.pagerank(G, max_iter=1000, weight='weight').values()))

            elif METRIC=='LocalEfficiency':
                G = Functions.create_distance_graph_from_AM(A) # Use 'distance'ç
                temp = Metrics.compute_LE_SL(G) # Implementation of LE on single layer networks
            
            else:
                raise Exception("ERROR: Incorrect metric value! (METRIC is {})".format(METRIC))
        except Exception as e:
                print(f"Error en sujeto {i}: {e}")
                return None
        values[i,:] = temp

    metrics_folder = "/Users/aina/Desktop/TFG/codi/metriques/monoplex"
    file_metric = os.path.join(metrics_folder, METRIC + ".csv")
    np.savetxt(file_metric, values, delimiter=",", fmt='%1.8f') # export
    return(values)

multilayer = np.load("xarxes/monoplex.npy")
lst_metrics = ["Strength", "Degree", "Clustering", "ClosenessCentrality", "BetweennessCentrality", "PageRank", "LocalEfficiency"]
for metrica in lst_metrics:
    get_metric_values(multilayer, METRIC=metrica)
    
print("    Process finished successfully!")
