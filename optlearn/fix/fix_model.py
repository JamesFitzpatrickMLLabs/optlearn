import numpy as np

from optlearn import graph_utils

from optlearn.fix import fix_utils


def minimum_degree(original_graph, pruned_graph, threshold=None):
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


def connected(original_graph, pruned_graph, threshold=None):
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
