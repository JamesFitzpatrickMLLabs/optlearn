import itertools
import networkx

import random

import numpy as np


def get_edge_weight(graph, vertex_a, vertex_b):
    """ Get edge weight between two vertices """

    return graph[vertex_a][vertex_b]["weight"]


def get_neighbours(graph, vertex):
    """ Get neighbours of vertex (including self) """

    return list(graph[vertex].keys())


def get_edge_weights(graph, vertex, out=True):
    """ Get edge weights for edges for a given vertex """

    if out: 
        return [get_edge_weight(graph, vertex, item)
                for item in get_neighbours(graph, vertex)]
    else:
        return [get_edge_weight(graph, item, vertex)
                for item in get_neighbours(graph, vertex)]


def minimum_weight(graph, vertex, out=True):
    """ Compute minimum edge weight incident with vertex """

    return np.min(get_edge_weights(graph, vertex, out=out))


def max_weight(graph, vertex, out=True):
    """ Compute maximum edge weight incident with vertex """

    return np.max(get_edge_weights(graph, vertex, out=out))


def mean_weight(graph, vertex, out=True):
    """ Compute mean edge weight incident with vertex """

    return np.mean(get_edge_weights(graph, vertex, out=out))


def get_vertices(graph):
    """ Get all of the graph vertices in a sorted array """
    
    return np.array(list(graph.nodes))


def get_edges(graph):
    """ Get all of the graph edges in a sorted array """

    pair = [get_vertices(graph), ] * 2 
    return np.array(list(itertools.product(*pair)))


def get_weights(graph):
    """ Get all of the graph edges in the same order as the edges """

    vertices = get_vertices(graph)
    return [get_edge_weights(graph, vertex) for vertex in vertices]


def sample_tsp_tour(graph):
    """ Randomly sample a feasible TSP tour """

    vertices = get_vertices(graph).tolist()
    return np.array(random.sample(vertices, len(vertices)))


def sample_tsp_tours(graph, tours=1):
    """ Randomly sample feasible TSP tours """

    return np.array([sample_tsp_tour(graph).tolist() for num in range(tours)])


def append_last_node(tour):
    """ Take a tour and append the first node to the end """
    
    return np.concatenate((tour, tour[0:1]))


def get_tour_edges(tour):
    """ Get an array of edges for a given tour """

    items = append_last_node(tour)
    return np.array([[items[num], items[num+1]] for num in range(len(tour))])


def get_tour_lengths(graph, tour):
    """ Get an array of edge lengths for a tour """

    edges = get_tour_edges(tour)
    return np.array([get_edge_weight(graph, *edge) for edge in edges])

    
def compute_tour_length(graph, tour):
    """ Compute the length of a given tour """

    return get_tour_lengths(graph, tour).sum()


def compute_tour_lengths(graph, tours):
    """ Compute the lengths of the given tours """

    return np.array([compute_tour_length(graph, tour) for tour in tours])


def check_edges_in_tour(edges, tour):
    """ Check if the given edges are in the tour """

    tour_edges = get_tour_edges(tour)
    return np.array([edge.tolist() in tour_edges.tolist() for edge in edges])


def check_edges_in_tours(edges, tours):
    """ Check if the given edges are in the tours """

    return np.array([check_edges_in_tour(edges, tour).tolist() for tour in tours])


def hash_edges(edges):
    """ Hash edges by converting them to tuples """

    return [hash(tuple(edge)) for edge in edges]


def hash_edges_in_tours(edges, tours):
    """ Check if the given edges are in the tours by hashing them first"""

    tours_hashes = np.vstack([hash_edges(get_tour_edges(tour)) for tour in tours])
    edges_hashes = np.array([hash_edges(edges)]).flatten()
        
    return np.array([np.in1d(edges_hashes, hash).tolist() for hash in tours_hashes]) 


def sort_tours_by_length(graph, tours, ascending=True):
    """ Sort an array of tours by their objective value """

    tour_lengths = compute_tour_lengths(graph, tours)
    tour_ranks = tour_lengths.argsort()
    if ascending:
        ordered_tours = tours[tour_ranks]
    else:
        ordered_tours = tours[tour_ranks[::-1]]
    return ordered_tours
        

def sample_sorted_tsp_tours(graph, tours=1, ascending=True):
    """ Randomly sample feasible TSP tours and sort them by length """

    tours = np.array([sample_tsp_tour(graph).tolist() for num in range(tours)])
    return sort_tours_by_length(graph, tours, ascending=ascending)


def infinite_self_weights(graph):
    """ Sets all of the self-weights in a graph to infinity """

    vertices = get_vertices(graph)
    for vertex in vertices:
        graph[vertex][vertex]["weight"] = 99999999999
    return graph
