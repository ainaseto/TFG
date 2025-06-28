
import networkx as nx
import numpy as np
import Metrics as Metrics
import Functions as Functions
import os

def get_metric_values(em, METRIC=None):
    num_subjs = em.shape[0]
    num_nodes = 76
    metrics_folder = "/Users/aina/Desktop/TFG/codi/metriques/multiplex"
    file_metric = os.path.join(metrics_folder, METRIC + ".csv")
    if os.path.exists(file_metric):
        print(f"Metric '{METRIC}' was previously computed and stored...")
    else:
        print(f"Computing values for metric '{METRIC}'...")
    values = np.zeros((num_subjs, num_nodes), dtype=float)
    for i in range(num_subjs):
        print("Processing subject {}...".format(i))
        A = em[i,:,:] # get data 
        bool_reshape = True
        # compute the specific metric
        try:
            if METRIC=='Degree':
                temp = np.count_nonzero(A > 0, axis=0)

            elif METRIC=='Strength':
                temp = np.sum(A, axis=0)

            elif METRIC=='LocalEfficiency':
                A_inv = Functions.create_distance_A_from_A(A) # Use 'distance'            
                #plot_ml_matrix(fs, A, i) # debug only                
                A_min = Functions.compute_A_min_multiplex(A_inv) # Compute A min               
                G = Functions.create_graph_from_AM(A_min) # create G
                temp = Metrics.compute_LE_SL(G) # compute LE as a single layer                
                bool_reshape = False # Do not reshape 

            elif METRIC=='ClosenessCentrality':
                A_inv = Functions.create_distance_A_from_A(A) # Use 'distance'
                #plot_ml_matrix(fs, A, i) # debug only
                A_min = Functions.compute_A_min_multiplex(A_inv) # Compute A min
                G = Functions.create_graph_from_AM(A_min) # create G
                temp = np.array(list(nx.closeness_centrality(G, distance='weight').values())) # compute LE as a single layer
                bool_reshape = False # Do not reshape 

            elif METRIC=='BetweennessCentrality':
                A_inv = Functions.create_distance_A_from_A(A) # Use 'distance'
                #plot_ml_matrix(fs, A, i) # debug only
                A_min = Functions.compute_A_min_multiplex(A_inv) # Compute A min
                G = Functions.create_graph_from_AM(A_min) # create G
                #print(f"Nodes: {G.nodes()}")
                #print(f"Arestes: {G.edges()}")
                temp = np.array(list(nx.betweenness_centrality(G, k=None, normalized=True, weight='weight').values())) # compute LE as a single layer
                bool_reshape = False # Do not reshape 

            elif METRIC == 'PageRank':
                A_inv = Functions.create_distance_A_from_A(A)  
                A_min = Functions.compute_A_min_multiplex(A_inv) 
                G = Functions.create_graph_from_AM(A_min) 
                temp = np.array(list(nx.pagerank(G, weight='weight').values())) 
                bool_reshape = False 

            elif METRIC == 'Clustering':
                A_inv = Functions.create_distance_A_from_A(A)  
                A_min = Functions.compute_A_min_multiplex(A_inv)  
                G = Functions.create_graph_from_AM(A_min)  
                temp = np.array(list(nx.clustering(G, weight='weight').values()))  
                bool_reshape = False  

            else:
                raise Exception("ERROR: Incorrect metric value! (METRIC is {})".format(METRIC))
        except Exception as e:
                return None

        if bool_reshape:
            temp2 = temp.reshape((76,3), order='F')
            temp3 = np.sum(temp2, axis=1) # sum values
        else:
            temp3 = temp
        values[i,:] = temp3 # store the results for each node
    np.savetxt(file_metric, values, delimiter=",", fmt='%1.8f') # export
    return(values)


multiplex = np.load("xarxes/multiplex.npy")
lst_metrics = ["Strength", "Degree", "Clustering", "ClosenessCentrality", "BetweennessCentrality", "PageRank", "LocalEfficiency"]
for metrica in lst_metrics:
    get_metric_values(multiplex, METRIC=metrica)
    
print("    Process finished successfully!")
