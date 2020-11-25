import numpy as np

from optlearn import graph_utils

from optlearn.fix import fix_utils
from optlearn.mst import mst_model


def ensure_minimum_degree(original_graph, pruned_graph, threshold=None):
    """ Make sure that each vertex has minium degree of threshold in the pruned graph """

    threshold = threshold or graph_utils.logceil(pruned_graph)

    if fix_utils.is_weakly_connected(pruned_graph):
        new_edges = fix_utils.get_all_k_min_strenghtheners(original_graph,
                                                           pruned_graph,
                                                           threshold
        )
    else:
        return pruned_graph
    
    return fix_utils.migrate_edges(original_graph, pruned_graph, new_edges)


def ensure_connected(original_graph, pruned_graph, threshold=None):
    """ Make sure that the graph is connected with threshold edges between components """

    threshold = threshold or graph_utils.logceil(pruned_graph)

    if not fix_utils.is_connected(pruned_graph):
        new_edges = fix_utils.get_all_k_min_connectors(original_graph,
                                                       pruned_graph,
                                                       threshold,
        )
    else:
        return pruned_graph
        
    return fix_utils.migrate_edges(original_graph, pruned_graph, new_edges)


def ensure_doubletree(original_graph, pruned_graph, iterations=None):
    """ Make sure that the graph at least has two doubletree edge sets """


    if iterations is None:
        iterations = int(np.ceil(np.log2(len(original_graph.nodes))))
        
    model = mst_model.doubleTreeSparsifier()
    new_edge_sets = model.run_sparsify(original_graph, iterations)

    for new_edges in new_edge_sets:
        pruned_graph = fix_utils.migrate_edges(original_graph, pruned_graph, new_edges)
        
    return pruned_graph 
