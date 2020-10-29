import numpy as np

from optlearn import graph_utils

from optlearn.quad import quad_utils


def slow_quadrilateral_frequency(graph, edge):
    """ Compute frequency for a single edge in a single quadrilateral """

    vertices = quad_utils.generate_quadrilateral(graph, edge)
    endpoints = quad_utils.get_asymmetric_endpoints(vertices)
    four_paths = quad_utils.get_four_path_pairs(endpoints, vertices)
    shortest_paths = quad_utils.get_shortest_paths(graph, four_paths)
    paths_edges = quad_utils.get_paths_edges(shortest_paths)
    return quad_utils.compute_frequencies(hash(edge), graph_utils.hash_edges(paths_edges))


def slow_quadrilateral_frequencies(graph, edge, N=100):
    """ Compute the mean edge frequency for N quadrilaterals """

    if len(np.unique(edge)) < 2:
        return -1
    else:
        frequencies = [slow_quadrilateral_frequency(graph, edge) for i in range(N)]
        return np.mean(frequencies)


def fast_quadrilateral_frequency(graph, edges):
    """ Compute the normalised frequencies using only opposing edge weights """

    edge_pairs = quad_utils.get_opposite_edge_pairs(edges)
    pair_sums = quad_utils.compute_pair_sums(graph, edge_pairs)
    pair_freqs = quad_utils.normalised_frequencies(pair_sums)
    return quad_utils.get_edge_frequencies(edges, edge_pairs, pair_freqs)


def update_frequencies(edges, freqs, all_freqs, all_counts, order, min_vertex):
    """ Update the current estimate of the frequencies for each given edge """

    for (edge, freq) in zip(edges, freqs):
        index = graph_utils.compute_vector_index(edge, order, min_vertex)
        current_count, current_freq = all_counts[index], all_freqs[index]
        update_value = (current_count * current_freq + freq) / (current_count + 1)
        all_freqs[index] = update_value
        all_counts[index] += 1
    return all_freqs, all_counts


def fast_quadrilateral_frequencies(graph, edges, rounds=100):
    """ Estimate the quadrilateral frequencies using opposing edges, given the edges to use """

    size = graph_utils.get_size(graph)
    all_freqs, all_counts = np.ones((size)) * 0.5, np.ones((size))
    order, min_vertex = graph_utils.get_order(graph), graph_utils.get_min_vertex(graph)
    for edge in edges:
        for round in range(rounds):
            quad_vertices = quad_utils.generate_quadrilateral(graph, edge)
            quad_edges = quad_utils.get_symmetric_endpoints(quad_vertices)
            quad_freqs = fast_quadrilateral_frequency(graph, quad_edges)
            all_freqs, all_counts = update_frequencies(quad_edges,
                                                       quad_freqs,
                                                       all_freqs,
                                                       all_counts,
                                                       order,
                                                       min_vertex
            )
    return all_freqs

