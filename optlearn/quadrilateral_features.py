import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from optlearn import io_utils
from optlearn import graph_utils


def get_path_edges(path):
    """ Get an array of edges for a given path """

    return [tuple([path[num], path[num+1]]) for num in range(len(path)-1)]


def get_path_lengths(graph, path):
    """ Get an array of edge lengths for a path """

    edges = get_path_edges(path)
    return np.array([graph_utils.get_edge_weight(graph, *edge) for edge in edges])

    
def compute_path_length(graph, path):
    """ Compute the length of a given path """

    return get_path_lengths(graph, path).sum()


def compute_paths_lengths(graph, paths):
    """ Compute the lengths of given paths """

    return [compute_path_length(graph, path) for path in paths]


def compute_paths_pairs_lengths(graph, paths):
    """ Compute the lengths of given pairs of paths """

    return [compute_paths_lengths(graph, pair) for pair in paths]


def get_symmetric_endpoints(vertices):
    """ Get six path endpoint pairs, assuming symmetry in edges """

    return [(vertices[0], vertices[1]),
            (vertices[0], vertices[2]),
            (vertices[0], vertices[3]),
            (vertices[1], vertices[2]),
            (vertices[1], vertices[3]),
            (vertices[2], vertices[3]),
    ]


def get_asymmetric_endpoints(vertices):
    """ Get twelve path endpoint pairs, assuming asymmetry in edges """

    return [(vertices[0], vertices[1]),
            (vertices[0], vertices[2]),
            (vertices[0], vertices[3]),
            (vertices[1], vertices[0]),
            (vertices[1], vertices[2]),
            (vertices[1], vertices[3]),
            (vertices[2], vertices[0]),
            (vertices[2], vertices[1]),
            (vertices[2], vertices[3]),
            (vertices[3], vertices[0]),
            (vertices[3], vertices[1]),
            (vertices[3], vertices[2]),
    ]


def get_paths_endpoints(vertices, symmetric=False):
    """ Return each pair of endpoints for each possible four-path """

    if symmetric:
        return get_symmetric_endpoints(vertices)
    else:
        return get_asymmetric_endpoints(vertices)


def get_four_path_pair(endpoint, vertices):
    """ given a pair of path endpoints, compute both possible four-vertex paths """

    others = [item for item in vertices if item not in endpoint]
    return [tuple([endpoint[0], others[0], others[1], endpoint[1]]),
            tuple([endpoint[0], others[1], others[0], endpoint[1]]),
    ]


def get_four_path_pairs(endpoints, vertices):
    """ given pairs of path endpoints, compute all pairs of four-vertex paths """

    return [get_four_path_pair(endpoint, vertices) for endpoint in endpoints]


def get_shortest_path(graph, paths):
    """ From a bunch of specified paths, determine the shortest """

    lengths = compute_paths_lengths(graph, paths)
    return paths[np.argmin(lengths)]


def get_shortest_paths(graph, paths):
    """ From a bunch of sets of specified paths, determine the shortest in each set """

    return [get_shortest_path(graph, path) for path in paths]


def get_paths_edges(paths):
    """ Get all edges from a set of paths """

    edges = [get_path_edges(path) for path in paths]
    return [item for edge in edges for item in edge]


def get_all_edges_complete(graph):
    """ Get all edges in a complete graph"""

    itertools.permutations(graph.nodes, 2)


def compute_frequencies(edge_hash, hashes):
    """ Compute the frequency of given edge hash in list of hashes """

    return hashes.count(edge_hash)


def generate_quadrilateral(graph, edge):
    """ Generate vertices for a random quadrilateral, given an edge """

    selection_nodes = [node for node in graph.nodes if node not in edge]
    new_vertices = random.sample(selection_nodes, 2)
    return list(edge) + new_vertices


def compute_edge_quadrilateral_frequency(graph, edge):
    """ Compute frequency for a single edge in a single quadrilateral """

    vertices = generate_quadrilateral(graph, edge)
    endpoints = get_asymmetric_endpoints(vertices)
    four_paths = get_four_path_pairs(endpoints, vertices)
    shortest_paths = get_shortest_paths(graph, four_paths)
    paths_edges = get_paths_edges(shortest_paths)
    return compute_frequencies(hash(edge), graph_utils.hash_edges(paths_edges))


def compute_edge_quadrilateral_frequencies(graph, edge, N=10):
    """ Compute the mean edge frequency for N quadrilaterals """

    if len(np.unique(edge)) < 2:
        return -1
    else:
        frequencies = [compute_edge_quadrilateral_frequency(graph, edge) for i in range(N)]
        return np.mean(frequencies)


def compute_f7_edges(graph, N=100):
    """ Compute the edge frequency features for all edges (slow implementation) """
    
    edges = graph_utils.get_edges(graph)
    return [compute_edge_quadrilateral_frequencies(graph, edge, N=N) for edge in edges]
