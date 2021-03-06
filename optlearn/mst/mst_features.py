import numpy as np

from optlearn import graph_utils

from optlearn.mst import mst_model


def extract_edges(model, graph, iterations="auto"):
    """ Use the MWST sparsifier to extract edges from the graph """

    if iterations == "auto":
        order = graph_utils.get_order(graph)
        iterations = int(np.ceil(np.log2(order)))
    if type(iterations) is not int:
        raise ValueError("Expected auto or int for mst pruning iterations!")
    
    pruned_edges = model.run_sparsify(graph, iterations=iterations)
    return [edge for edge_list in pruned_edges for edge in edge_list]


def build_prune_indicators(model, graph, iterations="auto"):
    """ Build indicator features for each edge in the graph """

    pruned_edges = extract_edges(model, graph, iterations)
    return graph_utils.compute_indicator_vector(graph, pruned_edges)

    
