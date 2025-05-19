
import networkx as nx
import numpy as np
import sys
 

# Print some basics properties of G
def report_graph_basics(G):
    A = nx.adjacency_matrix(G).todense()
    v_min = np.min(A)
    v_max = np.max(A)
    n_zero = np.count_nonzero(A==0)
    n_non_zero = np.count_nonzero(A)
    print("   Min={:.2f}, Max={:.2f}, # Zero={}, # Non-Zero={}".format(v_min, v_max, n_zero, n_non_zero))


# Create graph G from adjacency matrix A
def create_graph_from_AM(A):
    G = nx.from_numpy_array(A=A, parallel_edges=False) # create undirected and weighted graph from adjacency matrix
    return G


# Compute the 1/w to obtain the "distance" between nodes 
# higher values mean shorter distance
def create_distance_A_from_A(A):
    A_inv = np.zeros(A.shape, dtype=float)
    for i in range(A.shape[0]): # Compte A=1/w 
        for j in range(A.shape[1]):
            if not A[i,j] == 0:
                A_inv[i,j] = 1 / A[i,j]
    return A_inv


# some measures, like betweenness and closeness centrality, are based on distance, 
# so we need to compute 1/weights in order to get the shortest paths
def create_distance_graph_from_AM(A):
    A_inv = create_distance_A_from_A(A) # Compte A=1/w 
    G = create_graph_from_AM(A_inv) # create undirected and weighted graph from adjacency matrix
    return G


def compute_A_min(A):
    """
    Compute the matrix of minimum distances between nodes considering the multilayer network

    Parameters
    ----------
    A : ndarray
        Super-Adjacency matrix (152x152)
    Returns
    -------
    ndarray
        Adjacency matrix (76x76)
    """
    num_nodes_SG = A.shape[0] # number of nodes in supergraph
    if (num_nodes_SG % 2) == 0: # number of nodes in single graph
        num_nodes_G = int(num_nodes_SG / 2)
    else:
        print("The number of nodes in super graph is not EVEN! ({})".format(num_nodes_SG))
        sys.exit(0)

    ###
    # Step 1: create A_min
    # A_min[i,j]: minimum path from i to j through any layer
    A_min = np.zeros((num_nodes_G, num_nodes_G), dtype=float)

    ### DEBUG
    count_min = np.zeros(4, dtype=int)
    for source in range(num_nodes_G):  # for each node in G (i.e. [1..76])
        for target in range(num_nodes_G):
            # distance value through different layers
            tmp = np.asarray((A[source, target] # source RS --> target RS
                , A[source, (target+num_nodes_G)] # source RS --> target GM (through FA)
                , A[source+num_nodes_G, target] # source GM --> target RS (through FA)
                , A[(source+num_nodes_G), (target+num_nodes_G)])) # source GM --> target GM
            if np.count_nonzero(tmp):
                min_value = np.min(tmp[np.nonzero(tmp)])
                A_min[source, target] = min_value
                ### DEBUG 
                ind = np.where(tmp==min_value)[0][0]
                count_min[ind] = count_min[ind] + 1
            else:
                A_min[source, target] = 0

    ### DEBUG
    """
    count_min_total = np.sum(count_min)
    count_min_rs = count_min[0]
    count_min_fa = count_min[1] + count_min[2] # FA includes: RS -> GM and GM -> RS
    count_min_gm = count_min[3]
    print("RS={} ({:.2f} %), FA={} ({:.2f} %), GM={} ({:.2f} %), Total={}".format(
        count_min_rs, (count_min_rs / count_min_total) * 100
        , count_min_fa, (count_min_fa / count_min_total) * 100
        , count_min_gm, (count_min_gm / count_min_total) * 100
        , count_min_total))
    """
    return A_min


def compute_A_min_multiplex(A):
    """
    Compute the matrix of minimum distances between nodes considering the multilayer network

    Parameters
    ----------
    A : ndarray
        Super-Adjacency matrix (228x228) for 3-layer multiplex network (3 x 76 nodes)

    Returns
    -------
    ndarray
        Adjacency matrix (76x76) with the minimum distance between nodes across all layers
    """
    num_nodes_SG = A.shape[0]  # número total de nodes (228)
    if (num_nodes_SG % 76) != 0:
        print("Error: el número total de nodes no és múltiple de 76!")
        sys.exit(0)

    num_layers = num_nodes_SG // 76  # ara hauria de donar 3
    num_nodes = 76

    # Inicialitza la matriu de mínimes distàncies
    A_min = np.zeros((num_nodes, num_nodes), dtype=float)

    for i in range(num_nodes):
        for j in range(num_nodes):
            # Agafem totes les combinacions de capes per i i j
            dist_values = []
            for l1 in range(num_layers):
                for l2 in range(num_layers):
                    idx_i = i + l1 * num_nodes
                    idx_j = j + l2 * num_nodes
                    value = A[idx_i, idx_j]
                    if value != 0:
                        dist_values.append(value)
            if dist_values:
                A_min[i, j] = np.min(dist_values)
            else:
                A_min[i, j] = 0  # o np.inf si vols representar desconnexions

    return A_min
