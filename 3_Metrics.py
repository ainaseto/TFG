import networkx as nx
import numpy as np

def compute_CC_ML(G):
    """
    Compute the closeness centrality (CC) in multilayer networks
    PARAMETERS
    ----------
    G: Networkx graph
        Adjacency matrix of minimum distances between nodes (76x76)
    """
    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    values = np.zeros(num_nodes, dtype=float)  # array to store the scores
    sp = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))  # compute SP for all nodes
    for node in nodes:  # for each node, compute the CC
        values[node_to_index[node]] = (num_nodes - 1) / sum(sp[node].values())  # CC(x) = (N-1) / sum_y d(x,y)
    return values


def compute_LE_SL(G):
    """
    Local Efficiency (LE) for single layer networks   
    PARAMETERS
    ----------
    G: Networkx graph
        Adjacency matrix of minimum distances between nodes (76x76) 
    """
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    values = np.zeros(num_nodes, dtype=float)  # array to store the scores
    for node in nodes:  # for each node in G
        subG = nx.ego_graph(G, n=node, radius=1, center=False, undirected=True)  # get the ego network
        #print("Creating subgraph of 1-neighbourhood of node {}... [n={} and e={}]".format(node, subG.number_of_nodes(), subG.number_of_edges()))
        eff_node = 0  # node efficiency
        if subG.number_of_nodes() > 1:  # check if subgraph is not null
            if not nx.is_connected(subG):  # check if it is connected
                subG = subG.subgraph(max(nx.connected_components(subG), key=len))
                #print("   New subgraph of node {} has n={} and e={}".format(node, subG.number_of_nodes(), subG.number_of_edges()))
            sp = dict(nx.all_pairs_dijkstra_path_length(subG, weight='weight'))  # compute SP for all nodes in ego network
            subnodes = list(subG.nodes())
            num_subnodes = subG.number_of_nodes()
            for node_source in subnodes:
                for node_target in subnodes:
                    if node_source != node_target:
                        eff_node += (1 / sp[node_source][node_target])
            eff_node /= (num_subnodes * (num_subnodes - 1))
        values[node_to_index[node]] = eff_node  # store node efficiency
    return values
