import copy

import random

import numpy as np
import networkx as nx


def get_order(graph):
    """ Get the order of a graph """

    return len(graph.nodes)


def get_size(graph):
    """ Get the size of the graph """

    return len(graph.edges)


def get_min_vertex(graph):
    """ Get the value of the minimum vertex """

    return np.min(graph.nodes)


def get_edge_weight(graph, vertex_a, vertex_b):
    """ Get edge weight between two vertices, if there is no edge return np.inf """

    vertex = graph[vertex_a]
    try:
        weight = vertex[vertex_b]["weight"]
        return weight
    except KeyError:
        return np.inf


def get_edges_weights(graph, edges):
    """ Get the weights between the given edges """

    return [get_edge_weight(graph, *edge) for edge in edges]

    
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

    return list(graph.edges)


def get_weights(graph, weight="weight"):
    """ Get all of the graph edges in the same order as the edges """

    edges = get_edges(graph)
    return [graph[edge[0]][edge[1]][weight] for edge in edges]


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


def get_tour_edges_asymmetric(tour):
    """ Get an array of edges for a given tour, assuming a directed graph """

    items = append_last_node(tour)
    edges = np.array([[items[num], items[num+1]] for num in range(len(tour))])
    return edges
    

def get_tour_edges_symmetric(tour):
    """ Get an array of edges for a given tour, with (i, j): i > j """

    edges = get_tour_edges_asymmetric(tour)
    return np.sort(edges, axis=1)


def get_tour_edges(tour, symmetric=True):
    """ Get an array of edges for a given tour, with (i, j): i > j """

    if symmetric:
        return get_tour_edges_symmetric(tour)
    else:
        return get_tour_edges_asymmetric(tour)


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
    return np.array([list(edge) in tour_edges.tolist() for edge in edges])


def check_edges_in_edges(edges, other_edges):
    """ Check if the given edges are in the tour """

    return np.array([list(edge) in other_edges.tolist() for edge in edges])


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


def largify_weights(graph, edges):
    """ Sets weights of the given edges in a graph to large values """

    vertices = get_vertices(graph)
    for edge in edges:
        graph[edge[0]][edge[1]]["weight"] = 10000000
    return graph


def infinite_self_weights(graph):
    """ Sets all of the self-weights in a graph to infinity """

    vertices = get_vertices(graph)
    for vertex in vertices:
        graph[vertex][vertex]["weight"] = 99999999999
    return graph


def delete_self_weights(graph):
    """ Deletes all of the self-weights in a graph """

    vertices = get_vertices(graph)
    for vertex in vertices:
        graph.remove_edge(vertex, vertex)
    return graph


def to_undirected(graph):
    """ Creates a copy of the graph in undirected form """

    new_graph = copy.deepcopy(graph)
    edges = get_edges(new_graph).to_directe()
    new_graph.remove_edges_from([edge for edge in edges if edge[1] > edge[0]])
    return new_graph


def check_cycle(graph):
    """ Checks for cycles in the given graph and returns one or an empty list """

    # return nx.find_cycle(graph)
    try:
        return nx.find_cycle(graph)
    except:
        return []


def compute_mincut(graph, vertex_a, vertex_b, capacity="weight"):
    """ Compute the mincut for the given graph """

    return nx.algorithms.flow.minimum_cut(graph, vertex_a, vertex_b, capacity=capacity)


def compute_mincut_values(graph, capacity="weight"):
    """ Compute the mincut values for each edge of the given graph """

    mincut_values = []
    for edge in graph.edges:
        mincut_values.append(compute_mincut(graph, *edge, capacity)[0])
    return mincut_values


def compute_unique_mincut_values(graph, capacity="weight"):
    """ Compute the unique mincut values of the given graph """

    mincut_values = compute_mincut_values(graph, capacity=capacity)
    return np.unique(mincut_values)


def check_graph(graph):
    """ Check if the given graph is undirected """

    return type(graph) == nx.Graph


def check_digraph(graph):
    """ Check of the graph is a directed graph """

    return type(graph) == nx.DiGraph()


def logceil(graph, clip=True):
    """ Get the ceiling of log_{2}(n) """

    value = int(np.ceil(np.log2(len(graph.nodes))))

    if clip:
        return max(5, value)
    else:
        return value


def build_graph_from_edges(edges, weights=None, symmetric=True):
    """ Using the given edges, build a graph with unit weights """

    if symmetric:
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()
    weights = weights or [1, ] * len(edges)

    graph.add_edges_from(edges)

    return graph


def compute_vector_index_symmetric(edge, order, min_vertex):
    """ Compute the index of the given edge in the edge vector """
    
    (i, j) = edge

    if i >= j:
        raise ValueError("Indices not symmetric: {}, {}".format(i, j))
    
    vertical_offset = np.sum([order - k + (min_vertex - 1)
                              for k in range(1 - (1 - min_vertex), i)])
    horizontal_offset = j - i - 1 
    return int(vertical_offset + horizontal_offset)


def compute_vector_index(edge, order, min_vertex, symmetric=True):
    """ Compute the index of the given edge in the edge vector """

    if symmetric:
        return compute_vector_index_symmetric(edge, order, min_vertex)
    else:
        raise Exception("Assymetric edge index computer not implemented!")
        

def compute_indicator_vector(graph, edges):
    """ Build an indicator vector of length size(graph) for the given edges """

    symmetric = check_graph(graph)
    size, order = get_size(graph), get_order(graph)
    min_vertex = get_min_vertex(graph)
    indices = [compute_vector_index(edge, order, min_vertex, symmetric) for edge in edges]
    return build_indicator_vector(size, indices)


def build_indicator_vector(length, nonzeros):
    """ Build an indicator vector with the nonzero indices specified """

    vector = np.zeros((length))
    vector[nonzeros] = 1
    return vector
