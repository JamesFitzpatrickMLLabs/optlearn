import numpy as np

from optlearn import graph_utils


def build_indicator_vector(length, indices):
    """ Build a binary vector with nonzeros only at the given indices """

    vector = np.zeros((length))
    for index in indices:
        vector[index] = 1
    return vector


def compute_labels_from_edges(edges, order, min_vertex, symmetric=True):
    """ Build the binary label vector, given the edges """

    length = int((order * (order - 1)) / (int(symmetric) + 1))
    print(length)
    indices = [graph_utils.compute_vector_index(edge, order, min_vertex, symmetric)
               for edge in edges]
    return build_indicator_vector(length, indices)
    

def compute_labels_from_tour(tour, symmetric=True):
    """ Compute the label vector given the tour """

    edges = graph_utils.get_tour_edges_symmetric(tour)
    order, min_vertex = len(tour), np.min(tour)
    return compute_labels_from_edges(edges, order, min_vertex ,symmetric=symmetric)


