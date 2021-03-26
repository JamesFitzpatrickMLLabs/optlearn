import itertools

import networkx as nx
import numpy as np


def symmetrise_weights(digraph, min_weight, max_weight):
    """ Given a digraph, compute the updated weight matrix needed to symmetrise the graph """
    
    offset = 3 * max_weight - 4 * min_weight 
    return nx.linalg.adjacency_matrix(graph).toarray() + offset
    
    

def symmetrise_digraph(graph):
    """ Make a symmetric graph of order 2n from an assymetric digraph of order n """

    new_graph = nx.Graph()
    array = nx.linalg.adjacency_matrix(graph).toarray()
    array[np.eye(len(array)).astype(bool)] = 2 * np.max(array) * len(array)
    min_weight = array.min()
    array[np.eye(len(array)).astype(bool)] = 0
    max_weight = nx.linalg.adjacency_matrix(graph).max()
    del(array)

    if (4 * min_weight - 3 * max_weight) > 0:
        weights = symmetrise_weights(digraph, min_weight, max_weight)
    else:
        weights = nx.linalg.adjacency_matrix(graph).toarray()

    edges = itertools.product(list(graph.nodes), map(str, list(graph.nodes)))
    tuples = ((*edge, weight) for (edge, weight) in zip(edges, weights.flatten()))
    new_graph.add_weighted_edges_from(tuples)
    tuples = ((node, str(node), 0) for node in graph.nodes)
    new_graph.add_weighted_edges_from(tuples)
    
    return new_graph


def fix_degrees(graph, tree):
    """ Make sure the number of real and virtual odd degrees match! """

    real_odd_nodes = [node for node in graph.nodes if tree.degree[node] % 2 == 1]
    virtual_odd_nodes = [node for node in graph.nodes if tree.degree[str(node)] % 2 == 1]

    difference = np.abs(len(real_odd_nodes) - len(virtual_odd_nodes))
    
    if difference == 0:
        return tree
    else:
        if len(real_odd_nodes) > len(virtual_odd_nodes):
            count = int((len(real_odd_nodes) - len(virtual_odd_nodes)) / 2)
            differents = [node for node in real_odd_nodes if node not in virtual_odd_nodes]
            addables = [node for node in differents if (node, str(node)) not in tree.edges]
            removables = [node for node in differents if (node, str(node)) in tree.edges]
            adds = len(addables)
            removes = count - adds
            tree.add_weighted_edges_from([(node, str(node), 0) for node in addables[:adds]])
            tree.remove_edges_from([(node, str(node)) for node in removables[:removes]])
        else:
            count = int((len(virtual_odd_nodes) - len(real_odd_nodes)) / 2)
            differents = [int(node) for node in virtual_odd_nodes if node not in real_odd_nodes]
            addables = [node for node in differents if (node, str(node)) not in tree.edges]
            removables = [node for node in differents if (node, str(node)) in tree.edges]
            adds = len(addables)
            removes = count - adds
            tree.add_weighted_edges_from([(node, str(node), 0) for node in addables[:adds]])
            tree.remove_edges_from([(node, str(node)) for node in removables[:removes]])
    return tree
