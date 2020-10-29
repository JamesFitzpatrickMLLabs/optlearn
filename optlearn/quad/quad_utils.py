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


def symmetrise_edge(edge):
    """ Return a symmetrised form of the given edge """

    a, b = edge
    if a > b:
        return (b, a)
    else:
        return (a, b)
    

def get_symmetric_endpoints(vertices):
    """ Get six path endpoint pairs, assuming symmetry in edges """

    return [symmetrise_edge((vertices[0], vertices[1])),
            symmetrise_edge((vertices[2], vertices[3])),
            symmetrise_edge((vertices[0], vertices[2])),
            symmetrise_edge((vertices[1], vertices[3])),
            symmetrise_edge((vertices[0], vertices[3])),
            symmetrise_edge((vertices[1], vertices[2])),
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


def get_paths_endpoints(vertices, symmetric=True):
    """ Return each pair of endpoints for each possible four-path """

    if symmetric:
        return get_symmetric_endpoints(vertices)
    else:
        raise Exception("Asymmetric implementation should not be used!")
        return get_asymmetric_endpoints(vertices)


def get_opposite_edge(edge, edges):
    """ Get the opposite edge of the given edge """

    for item in edges:
        a, b = edge
        if a not in item and b not in item:
            return item
    raise ValueError("No opposite edge found!")


def get_opposite_edge_pairs(edges):
    """ Get the opposite edge pairs """

    poplist, pairlist = [item for item in edges], []
    while poplist:
        edge_a = poplist.pop(0)
        edge_b = get_opposite_edge(edge_a, poplist)
        edge_b = poplist.pop(poplist.index(edge_b))
        pairlist.append([edge_a, edge_b])
    return pairlist


def compute_pair_sum(graph, edge_pair):
    """ Compute the sum of the weights of a pair of edges """

    weights = graph_utils.get_edges_weights(graph, edge_pair)
    return np.sum(weights)


def compute_pair_sums(graph, edge_pairs):
    """ Compute the sum of the weights for each given edge pair """

    return [compute_pair_sum(graph, edge_pair) for edge_pair in edge_pairs]


def normalised_frequencies(pair_sums):
    """ Compute the normalised frequencies, given the pair sums """

    frequencies =  np.argsort(pair_sums)
    frequencies[frequencies.copy()] = np.arange(len(frequencies))
    return frequencies / frequencies.max()


def get_edge_frequency(edge, edge_pairs, pair_freqs):
    """ Get the frequency of a specific edge """

    for num, edge_pair in enumerate(edge_pairs):
        if edge in edge_pair:
            return pair_freqs[num]
    raise ValueError("Given edge not found in the edge pairs!")


def get_edge_frequencies(edges, edge_pairs, pair_freqs):
    """ Get the frequency of the given edges """

    return [get_edge_frequency(edge, edge_pairs, pair_freqs) for edge in edges]     


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
