import random

import numpy as np
import networkx as nx

from optlearn import graph_utils

from optlearn.quad import quad_features
from optlearn.mst import mst_features
from optlearn.mst import mst_model
from optlearn.mip import mip_model


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


def compute_f5_edges(graph, sort=False):
    """ Compute feature f5 of Sun et al. """
    """ Expects a set of tours and an array indicating if each edge 
        is in each tour """

    tours = np.array([random.sample(graph.nodes, len(graph.nodes)) for i in range(10000)])
    edges_in = graph_utils.hash_edges_in_tours(graph.edges, tours)
    tours = graph_utils.sort_tours_by_length(graph, tours)
    tour_ranks = np.expand_dims(np.arange(len(tours)) + 1, 1)
    scores = np.sum(edges_in / tour_ranks, axis=0)
    return scores / np.max(scores)
    

def compute_f6_edges(graph):
    """ Compute feature f6 of Sun et al. """
    """ Expects a set of tours and an array indicating if each edge 
        is in each tour """

    tours = np.array([random.sample(graph.nodes, len(graph.nodes)) for i in range(10000)])
    edges_in = graph_utils.hash_edges_in_tours(graph.edges, tours)
    tours = graph_utils.sort_tours_by_length(graph, tours)
    tour_lengths = graph_utils.compute_tour_lengths(graph, tours)
    numerator_a = (edges_in - edges_in.mean(axis=0))
    numerator_b = np.expand_dims(tour_lengths - tour_lengths.mean(), 1)
    numerator = np.sum(numerator_a * numerator_b, axis=0)
    denominator_a = np.power(np.sum(np.power(numerator_a, 2), axis=0), 0.5)
    denominator_b = np.power(np.sum(np.power(numerator_b,2)), 0.5)
    denominator = denominator_a * denominator_b
    scores =  np.nan_to_num(numerator / denominator, 0)
    return scores / np.min(scores)


def compute_f7_edges(graph, iterations="auto"):
    """ Compute quadilateral frequencies for each edge """

    if iterations == "auto":
        iterations = graph_utils.get_order(graph)
    model = mst_model.mstSparsifier()
    edges = mst_features.extract_edges(model, graph)
    return quad_features.fast_quadrilateral_frequencies(graph, edges, iterations)


def compute_f8_edges(graph):
    """ Compute the root relxation features for each edge """

    problem = mip_model.tspProblem(
        var_type="binary",
        graph=graph,
        verbose=False
    )

    problem.perform_relaxation()
    return problem.get_varvals()


def compute_f9_edges(graph):
    """ Indicator features from the MWST extraction method  """

    model = mst_model.mstSparsifier()
    return mst_features.build_prune_indicators(model, graph)

    
def compute_f10_edges(graph):
    """ Compare the edges to the max edge value """

    weights = np.array(graph_utils.get_weights(graph))
    return weights.flatten() / np.max(weights)


def compute_f11_edges(graph):
    """ Compare the edges to the min edge value """

    weights = np.array(graph_utils.get_weights(graph))
    return (weights.flatten() - np.min(weights)) / (np.max(weights) - np.min(weights))


def compute_f12_edges(graph):
    """ Compare the edges to the mean edge value """

    weights = np.array(graph_utils.get_weights(graph))
    return weights.flatten() / (np.max(weights) - np.min(weights))


def compute_f13_edges(graph):
    """ Include a feature that informs the order of the graph """

    order = graph_utils.get_order(graph)
    return np.log(10) / np.log(order)


def compute_fa_edges(graph):
    """ Compute the weight divided by the global max """

    weights = np.array(graph_utils.get_weights(graph))
    global_max = weights.max()
    return  (1 + weights) / (1 + global_max)


def compute_fb_edges(graph):
    """ Compute the weight divided by the max left neighbour weight """

    min_vertex = np.min(graph.nodes)
    weights = 1 + np.array(graph_utils.get_weights(graph)) 
    maxes = [1 + np.max(graph_utils.get_edge_weights(graph, node)) for node in graph.nodes]
    return np.array([weight / maxes[edge[0] - min_vertex]
                     for weight, edge in zip(weights, graph.edges)])


def compute_fc_edges(graph):
    """ Compute the weight divided by the max right neighbour weight """

    min_vertex = np.min(graph.nodes)
    weights = 1 + np.array(graph_utils.get_weights(graph)) 
    maxes = [1 + np.max(graph_utils.get_edge_weights(graph, node, out=False)) for
             node in graph.nodes]
    return np.array([weight / maxes[edge[1] - min_vertex] for
                     weight, edge in zip(weights, list(graph.edges))])


def compute_fd_edges(graph):
    """ Compute the weight divided by the global max """

    weights = np.array(graph_utils.get_weights(graph))
    global_min = weights.min()
    return  (1 + global_min) / (1 + weights) 


def compute_fe_edges(graph):
    """ Compute the weight divided by the min left neighbour weight """

    min_vertex = np.min(graph.nodes)
    weights = 1 + np.array(graph_utils.get_weights(graph)) 
    mins = [1 + np.min(graph_utils.get_edge_weights(graph, node)) for
            node in graph.nodes]
    return np.array([mins[edge[0] - min_vertex] / weight for
                     weight, edge in zip(weights, graph.edges)])


def compute_ff_edges(graph):
    """ Compute the weight divided by the min right neighbour weight """

    min_vertex = np.min(graph.nodes)
    weights = 1 + np.array(graph_utils.get_weights(graph)) 
    mins = [1 + np.min(graph_utils.get_edge_weights(graph, node, out=False)) for
            node in graph.nodes]
    return np.array([mins[edge[1]-min_vertex] / weight for
                     weight, edge in zip(weights, graph.edges)])


def compute_fg_edges(graph):
    """ Continuous features from the MWST extraction method  """

    model = mst_model.mstSparsifier()
    return mst_features.build_prune_features(model, graph)


def compute_fh_edges(graph):
    """ Compute the cutting solution features for each edge """

    problem = mip_model.tspProblem(
        solver="scip",
        var_type="continuous",
        graph=graph,
        verbose=False,
        get_quick=True,
    )

    rounds = int(np.ceil(np.log2(len(graph.edges))))
    problem.optimise(max_rounds=rounds)
    return problem.get_varvals()


def compute_fk_edges(graph):
    """ Compute the cutting solution features for each edge """

    problem = mip_model.tspProblem(
        solver="xpress",
        var_type="continuous",
        graph=graph,
        verbose=False,
        get_quick=False,
    )

    problem.perform_relaxation()
    costs = np.array(problem.get_redcosts())
    return costs / costs.max()


def compute_fi_edges_scip(graph):
    """ Compute the cutting reduced cost features for each edge """

    problem = mip_model.tspProblem(
        solver="scip",
        var_type="continuous",
        graph=graph,
        verbose=False,
        get_quick=False,
    )

    rounds = int(np.ceil(np.log2(len(graph.edges))))
    problem.optimise(max_rounds=rounds)
    costs = np.array(problem.get_redcosts())
    return costs / costs.max()


def compute_fi_edges_xpress(graph):
    """ Compute the cutting reduced cost features for each edge """

    problem = mip_model.tspProblem(
        solver="xpress",
        var_type="binary",
        graph=graph,
        verbose=False,
        get_quick=False,
    )
    
    rounds = int(np.ceil(np.log2(len(graph.edges))))
    
    problem.optimise(max_rounds=rounds)
    costs = np.array(problem.get_redcosts())
    return costs / costs.max()


def compute_fi_edges(graph):

    return compute_fi_edges_scip(graph)


def compute_fj_edges(graph, rounds=None, perturb=True):
    """ Compute the bet-and-run relaxation reduced costs for each edge """

    problem = mip_model.tspProblem(
        solver="scip",
        var_type="continuous",
        graph=graph,
        verbose=False,
        shuffle_columns=False,
        perturb=perturb,
        get_quick=False,
    )
    
    if rounds is None:
        rounds = int(np.ceil(np.log2(len(graph.edges))))
    costs = []

    for i in range(rounds):
        problem.optimise(max_rounds=1)
        costs.append(problem.get_redcosts())
        problem.problem.freeTransform()
        problem.set_objective(graph)
        costs = [(np.array(item) / np.max(item)).tolist() for item in costs]
        print(len(costs))
        return np.mean(costs, axis=0)


def compute_fp_edges(graph):
    """ Compute the product of the degree centralities for each edge """

    centralities = nx.degree_centrality(graph)
    centralities = np.array(list(centralities.values()))

    if np.min(graph.nodes) > 0:
        offset = - int(np.min(graph.nodes))
    else:
        offset = int(0)
    
    edges = np.array(list(graph.edges))
    left_item = centralities[edges[:, 0] + offset] 
    right_item = centralities[edges[:, 1] + offset]
    product = left_item * right_item 

    return product / product.max() 


def compute_fm_edges(graph, max_iter=1000):
    """ Compute the product of the eigen centralities for each edge """

    centralities = nx.eigenvector_centrality(graph, max_iter=max_iter)
    centralities = np.array(list(centralities.values()))
    
    edges = np.array(list(graph.edges))
    left_item = centralities[edges[:, 0]] 
    right_item = centralities[edges[:, 1]]
    product = left_item * right_item 

    return product / product.max() 



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
    "compute_f12_edges": compute_f11_edges,
    "compute_f13_edges": compute_f11_edges,
    "compute_f1_vertices": compute_f1_vertices,
    "compute_f2_vertices": compute_f2_vertices,
    "compute_f3_vertices": compute_f3_vertices,
    "compute_f4_vertices": compute_f4_vertices,
    "compute_fa_edges": compute_fa_edges,
    "compute_fb_edges": compute_fb_edges,
    "compute_fc_edges": compute_fc_edges,
    "compute_fd_edges": compute_fd_edges,
    "compute_fe_edges": compute_fe_edges,
    "compute_ff_edges": compute_ff_edges,
    "compute_fg_edges": compute_fg_edges,
    "compute_fh_edges": compute_fh_edges,
    "compute_fi_edges": compute_fi_edges,
    "compute_fj_edges": compute_fj_edges,
    "compute_fk_edges": compute_fk_edges,
    "compute_fp_edges": compute_fp_edges,
    "compute_fm_edges": compute_fm_edges,    
    }

