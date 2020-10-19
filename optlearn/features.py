from optlearn import graph_utils
from optlearn import quadrilateral_features

from optlearn.mip import mip_model

import numpy as np


def compute_f1_edge(graph, vertex_a, vertex_b, self_max=False):
    """ Compute feature f1 of Sun et al. for a specific edge"""
    """ If the self-weight is the largest weight, set self_max true """

    out_weights = graph_utils.get_edge_weights(graph, vertex_a)
    edge_weight = graph_utils.get_edge_weight(graph, vertex_a, vertex_b)
    numerator = edge_weight - np.min(out_weights)
    if not self_max:
        denominator = np.max(out_weights) - np.min(out_weights)
    else:
        denominator = np.partition(out_weights, -2)[-2] - np.min(out_weights)
    return numerator / denominator


def compute_f1_vertex(graph, vertex, self_max=False):
    """ Compute feature f1 of Sun et al. for a specific vertex """
    """ If the self-weight is the largest weight, set self_max true """

    out_weights = graph_utils.get_edge_weights(graph, vertex)
    numerator = out_weights - np.min(out_weights)
    if not self_max:
        denominator = np.max(out_weights) - np.min(out_weights)
    else:
        denominator = np.partition(out_weights, -2)[-2] - np.min(out_weights)
    return numerator / denominator


def compute_f2_edge(graph, vertex_a, vertex_b, self_max=False):
    """ Compute feature f2 of Sun et al. for a specific edge"""

    out_weights = graph_utils.get_edge_weights(graph, vertex_a)
    in_weights = graph_utils.get_edge_weights(graph, vertex_b, out=False)
    edge_weight = graph_utils.get_edge_weight(graph, vertex_a, vertex_b)
    numerator = edge_weight - np.min(out_weights)
    if not self_max:
        denominator = np.max(in_weights) - np.min(in_weights)
    else:
        denominator = np.partition(in_weights, -2)[-2] - np.min(in_weights)
    return numerator / denominator


def compute_f2_vertex(graph, vertex, self_max=False):
    """ Compute feature f2 of Sun et al. for a specific vertex """
    """ If the self-weight is the largest weight, set self_max true """
    """ NOTE that this is the transpose of the usual order """
    
    in_weights = graph_utils.get_edge_weights(graph, vertex, out=False)
    minimum = np.min(in_weights)
    numerator = in_weights - minimum
    if not self_max:
        denominator = np.max(in_weights) - np.min(in_weights)
    else:
        denominator = np.partition(in_weights, -2)[-2] - minimum
    return numerator / denominator


def compute_f3_edge(graph, vertex_a, vertex_b, self_max=False):
    """ Compute feature f3 of Sun et al. for a specific edge"""

    out_weights = graph_utils.get_edge_weights(graph, vertex_a)
    edge_weight = graph_utils.get_edge_weight(graph, vertex_a, vertex_b)
    if not self_max:
        numerator = edge_weight - np.mean(out_weights)
        denominator = np.max(out_weights) - np.min(out_weights)
    else:
        numerator = edge_weight - np.partition(out_weights, -2)[:-1].mean()
        denominator = np.partition(out_weights, -2)[-2] - np.min(out_weights)
    return numerator / denominator


def compute_f3_vertex(graph, vertex, self_max=False):
    """ Compute feature f3 of Sun et al. for a specific vertex """
    """ If the self-weight is the largest weight, set self_max true """

    out_weights = graph_utils.get_edge_weights(graph, vertex)
    if not self_max:
        numerator = out_weights - np.mean(out_weights)
        denominator = np.max(out_weights) - np.min(out_weights)
    else:
        numerator = out_weights - np.partition(out_weights, -2)[:-1].mean()
        denominator = np.partition(out_weights, -2)[-2] - np.min(out_weights)
    return numerator / denominator


def compute_f4_edge(graph, vertex_a, vertex_b, self_max=False):
    """ Compute feature f4 of Sun et al. for a specific edge"""

    out_weights = graph_utils.get_edge_weights(graph, vertex_a)
    in_weights = graph_utils.get_edge_weights(graph, vertex_b, out=False)
    edge_weight = graph_utils.get_edge_weight(graph, vertex_a, vertex_b)
    if not self_max:
        numerator = edge_weight - np.mean(in_weights)
        denominator = np.max(in_weights) - np.min(in_weights)
    else:
        numerator = edge_weight - np.partition(in_weights, -2)[:-1].mean()
        denominator = np.partition(in_weights, -2)[-2] - np.min(in_weights)
    return numerator / denominator


def compute_f4_vertex(graph, vertex, self_max=False):
    """ Compute feature f4 of Sun et al. for a specific vertex """
    """ If the self-weight is the largest weight, set self_max true """
    """ NOTE that this is the transpose of the usual order """
    
    in_weights = graph_utils.get_edge_weights(graph, vertex, out=False)
    minimum = np.min(in_weights)
    if not self_max:
        numerator = in_weights - np.mean(in_weights)
        denominator = np.max(in_weights) - minimum
    else:
        numerator = in_weights - np.partition(in_weights, -2)[:-1].mean()
        denominator = np.partition(in_weights, -2)[-2] - minimum 
    return numerator / denominator


def compute_f1_edges(graph, self_max=False):
    """ Compute feature f1 of Sun et al. """

    edges = graph_utils.get_edges(graph)
    return np.array([compute_f1_edge(graph, *edge, self_max=self_max) for edge in edges])


def compute_f1_vertices(graph, self_max=False):
    """ Compute feature f1 of Sun et al. """

    vertices = graph_utils.get_vertices(graph)
    return np.array([compute_f1_vertex(graph, vertex, self_max=self_max)
                     for vertex in vertices]).flatten()


def compute_f2_edges(graph, self_max=False):
    """ Compute feature f2 of Sun et al. """

    edges = graph_utils.get_edges(graph)
    return np.array([compute_f2_edge(graph, *edge, self_max=self_max) for edge in edges])


def compute_f2_vertices(graph, self_max=False):
    """ Compute feature f2 of Sun et al. """

    vertices = graph_utils.get_vertices(graph)
    return np.array([compute_f2_vertex(graph, vertex, self_max=self_max)
                     for vertex in vertices]).T.flatten()


def compute_f3_edges(graph, self_max=False):
    """ Compute feature f3 of Sun et al. """

    edges = graph_utils.get_edges(graph)
    return np.array([compute_f3_edge(graph, *edge, self_max=self_max) for edge in edges])


def compute_f3_vertices(graph, self_max=False):
    """ Compute feature f3 of Sun et al. """

    vertices = graph_utils.get_vertices(graph)
    return np.array([compute_f3_vertex(graph, vertex, self_max=self_max)
                     for vertex in vertices]).flatten()


def compute_f4_edges(graph, self_max=False):
    """ Compute feature f4 of Sun et al. """

    edges = graph_utils.get_edges(graph)
    return np.array([compute_f4_edge(graph, *edge, self_max=self_max) for edge in edges])


def compute_f4_vertices(graph, self_max=False):
    """ Compute feature f4 of Sun et al. """

    vertices = graph_utils.get_vertices(graph)
    return np.array([compute_f4_vertex(graph, vertex, self_max=self_max)
                     for vertex in vertices]).T.flatten()


def compute_f5_edges(graph, tours, edges_in, sort=False):
    """ Compute feature f5 of Sun et al. """
    """ Expects a set of tours and an array indicating if each edge 
        is in each tour """

    if not sort:
        tours = graph_utils.sort_tours_by_length(graph, tours - 1) + 1
    tour_ranks = np.expand_dims(np.arange(len(tours)) + 1, 1)
    scores = np.sum(edges_in / tour_ranks, axis=0)
    return scores / np.max(scores)
    

def compute_f6_edges(graph, tours, edges_in, sort=False):
    """ Compute feature f6 of Sun et al. """
    """ Expects a set of tours and an array indicating if each edge 
        is in each tour """

    if not sort:
        tours = graph_utils.sort_tours_by_length(graph, tours - 1) + 1
    tour_lengths = graph_utils.compute_tour_lengths(graph, tours - 1)
    numerator_a = (edges_in - edges_in.mean(axis=0))
    numerator_b = np.expand_dims(tour_lengths - tour_lengths.mean(), 1)
    numerator = np.sum(numerator_a * numerator_b, axis=0)
    denominator_a = np.power(np.sum(np.power(numerator_a, 2), axis=0), 0.5)
    denominator_b = np.power(np.sum(np.power(numerator_b,2)), 0.5)
    denominator = denominator_a * denominator_b
    scores =  np.nan_to_num(numerator / denominator, 0)
    return scores / np.max(scores)


def compute_f7_edges(graph):
    """ Compute quadilateral frequencies for each edge """

    return quadrilateral_features.compute_f7_edges(graph, 100)


def compute_f8_edges(graph):
    """ Compute the bet-and-run features for each edge """

    problem = mip_model.tspIntegerModel(graph=graph,
                                        formulation="dantzig",
                                        var_type="binary",
                                        solver="coinor")
    problem.solve_dantzig()
    return problem.get_varvals()

    
def compute_f9_edges(graph):
    """ Compare the edges to the max edge value """

    weights = np.array(graph_utils.get_weights(graph))
    return weights.flatten() / np.max(weights)


def compute_f10_edges(graph):
    """ Compare the edges to the min edge value """

    weights = np.array(graph_utils.get_weights(graph))
    return (weights.flatten() - np.min(weights)) / (np.max(weights) - np.min(weights))


def compute_f11_edges(graph):
    """ Compare the edges to the mean edge value """

    weights = np.array(graph_utils.get_weights(graph))
    return weights.flatten() / (np.max(weights) - np.min(weights))



functions = {
    "compute_f1_edges": compute_f1_edges,
    "compute_f2_edges": compute_f2_edges,
    "compute_f3_edges": compute_f3_edges,
    "compute_f4_edges": compute_f4_edges,
    "compute_f5_edges": compute_f5_edges,
    "compute_f6_edges": compute_f6_edges,
    "compute_f7_edges": compute_f7_edges,
    "compute_f8_edges": compute_f8_edges,
    "compute_f9_edges": compute_f9_edges,
    "compute_f10_edges": compute_f10_edges,
    "compute_f11_edges": compute_f11_edges,
    "compute_f1_vertices": compute_f1_vertices,
    "compute_f2_vertices": compute_f2_vertices,
    "compute_f3_vertices": compute_f3_vertices,
    "compute_f4_vertices": compute_f4_vertices,
    }

