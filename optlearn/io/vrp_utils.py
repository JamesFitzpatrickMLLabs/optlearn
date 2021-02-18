import itertools

import numpy as np
import networkx as nx

from scipy.spatial.distance import cdist

from optlearn.io import xml_utils


def read_vrp_xml(fname):
    """ Read and parse the VRP xml file """

    xml_root = xml_utils.read_xml(fname).getroot()
    return xml_utils.parse_vrp_file_xml(xml_root)


def euclidean(x, y):
    """ Compute euclidean distance between x and y """

    return np.sqrt(np.sum(np.power(x - y, 2)))


def manhattan(x, y):
    """ Compute manhattan distance between x and y """

    return np.sum(np.abs(x - y))


def add_node_info(graph, info_dict):
    """ Add the nodes in the info_dict to the graph """

    graph.add_nodes_from(info_dict["network"]["node_info"].items())
    return graph


def add_node_requests(graph, info_dict):
    """ Add the requests from the info_dict to the nodes in the graph """

    requests = info_dict["requests"]
    nx.set_node_attributes(graph, requests, "demand")
    return graph


def add_fleet_info(graph, info_dict):
    """ Add fleet information to the graph """

    for (vehicle, vehicle_dict) in info_dict["fleet"].items():
        graph.graph[vehicle] = vehicle_dict 
    return graph


def get_neighbours_symmetric(node, nodes):
    """ Get the indices of the neighbour nodes to the given node """

    return [item for item in nodes if item > node]


def get_neighbours_asymmetric(node, nodes):
    """ Get the indices of the neighbour nodes to the given node """

    return [item for item in nodes if item != node]


def get_nodes_neighbours(info_dict, symmetric=True):
    """ Build an iterator giving each (node, [...neighbours...]) combination """

    nodes = list(info_dict["network"]["node_info"].keys())

    if symmetric:
        func = get_neighbours_symmetric
    if not symmetric:
        func = get_neighbours_asymmetric

    return [(node, func(node, nodes)) for node in nodes[:-1]]


def get_node_neighbour_edges(node_neighbours):
    """ Get the node-neighbour edges for given pairs """

    node, neighbours = node_neighbours
    return [(node, neighbour) for neighbour in neighbours]


def get_node_coordinates(info_dict):
    """ Get the coordinates of the given nodes """

    node_info = info_dict["network"]["node_info"]
    return [node_info[key]["xy"] for key in node_info.keys()]


def compute_pairwise_distances(coordinates, metric, rounding, symmetric=True):
    """ Compute pairwise distances between coordinates """

    distances = cdist(coordinates, coordinates, metric=metric).round(rounding)

    if symmetric:
        return distances[np.triu_indices_from(distances, 1)]
    else:
        return distances.flatten()


def get_complete_edges(info_dict, symmetric=True):
    """ Get all the possible node pairs """

    keys = list(info_dict["network"]["node_info"].keys())
    
    if symmetric:
        return np.array(list(itertools.combinations(keys, 2)))
    else:
        return np.array(list(itertools.permutations(keys, 2)))


def add_weighted_edges(graph, edges, weights):
    """ Add the given edges to the graph with the specified weights """

    graph.add_weighted_edges_from([(*edge, {"weight": weight})
                                   for ((edge), weight) in zip(edges, weights)])
    return graph


def add_distance_edges(graph, info_dict, symmetric=True, rounding=0, metric="euclidean"):
    """ Compute and add distance weights """

    coordinates = get_node_coordinates(info_dict)
    weights = compute_pairwise_distances(coordinates, metric=metric,
                                         rounding=rounding, symmetric=True)
    edges = map(tuple, get_complete_edges(info_dict, symmetric=symmetric))
    # return edges, weights
    weighted_edges = [edge + (weight,) for (edge, weight) in zip(edges, weights)]
    graph.add_weighted_edges_from(weighted_edges)
    return graph


_metrics = {
    "euclidean": euclidean,
    "manhattan": manhattan,
}
