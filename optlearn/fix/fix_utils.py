import itertools

import numpy as np
import networkx as nx

from optlearn import graph_utils


def is_connected(pruned_graph):
    """ Check if the graph is connected """

    return nx.number_connected_components(pruned_graph) == 1


def is_weakly_connected(graph, threshold=None):
    """ Check if the graph is connected """

    threshold = threshold or graph_utils.logceil(graph)

    return any([len(graph[node]) < threshold for node in graph.nodes])


def get_weak_nodes(graph, threshold=None):
    """ Get the weakly connected nodes """

    threshold = threshold or graph_utils.logceil(graph)

    return [node for node in graph.nodes if len(graph[node]) < threshold]


def get_ghost_edges(original_graph, pruned_graph, node, threshold=None):
    """ Find the neighbours of the original graph not in the pruned_graph """

    original_neighbours = list(original_graph[node])
    pruned_neighbours = list(pruned_graph[node])
    return [(node, other) for other in original_neighbours if other not in pruned_neighbours]


def get_all_ghost_edges(original_graph, pruned_graph, threshold=None):
    """ Find the neighbours of the original graph not in the pruned_graph """

    ghost_edges = []

    for node in pruned_graph.nodes:
        ghost_edges += get_ghost_edges(original_graph, pruned_graph, node, threshold)
    return ghost_edges


def migrate_edges(original_graph, pruned_graph, edges):
    """ Copy the given edges to the pruned graph, retaining their attributes """

    for edge in edges:
        pruned_graph.add_edge(*edge, **original_graph[edge[0]][edge[1]])
    return pruned_graph


def get_k_min_edges(graph, edges, k, weight="weight"):
    """ Given some edges in the graph, find the k with the least weight """

    weights = graph_utils.get_edges_weights(graph, edges)
    return [edges[index] for index in np.argsort(weights)[:k]]


def get_k_min_strenghtheners(original_graph, pruned_graph, node, k, weight="weight"):
    """ For a given node get the k least-weight ghost edges from the original graph """

    ghost_edges = get_ghost_edges(original_graph, pruned_graph, node)
    return get_k_min_edges(original_graph, ghost_edges, k, weight=weight)


def get_all_k_min_strenghtheners(original_graph, pruned_graph, k,  weight="weight"):
    """ For each node get the k least-weight ghost edges from the original graph """

    strengtheners = []
    
    for node in get_weak_nodes(pruned_graph):
        degree = len(pruned_graph[node])
        k = graph_utils.logceil(pruned_graph) - degree
        strengtheners += get_k_min_strenghtheners(original_graph,
                                                  pruned_graph,
                                                  weight=weight,
                                                  node=node,
                                                  k=k, 
        )

    return strengtheners

    
def get_components(pruned_graph):
    """ Get the components of the graph """

    return list(nx.connected_components(pruned_graph))


def get_ghost_connectors(original_graph, component_a, component_b):
    """ Get the edges that connect from the original graph that connect two components """

    return [edge for edge in original_graph.edges if
            (edge[0] in component_a and edge[1] in component_b) or
            (edge[0] in component_b and edge[1] in component_a)
            ]


def get_k_min_connectors(original_graph, component_a, component_b, k, weight="weight"):
    """ Get the smallest k edges that connect the two components in the old graph """

    edges = get_ghost_connectors(original_graph, component_a, component_b)
    return get_k_min_edges(original_graph, edges, k=k, weight=weight)


def get_all_k_min_connectors(original_graph, pruned_graph, k, weight="weight"):
    """ Get for each component the smallest k edges to reconnect the new graph """

    components = get_components(pruned_graph)
    component_pairs = list(itertools.combinations(components, 2))
    connecting_edges = [get_k_min_connectors(original_graph, *pair, k, weight)
                        for pair in component_pairs]
    return [item for sublist in connecting_edges for item in sublist]


def get_bridges(graph):
    """ Find all of the bridges in the graph """

    return list(nx.algorithms.bridges(graph))


