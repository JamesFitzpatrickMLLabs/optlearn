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
    requests = {key: {"demand": value} for (key, value) in requests.items()}
    nx.set_node_attributes(graph, requests)
    return graph


def add_fleet_info(graph, info_dict):
    """ Add fleet information to the graph """

    for (vehicle, vehicle_dict) in info_dict["fleet"].items():
        graph.graph["fleet"] = {}
        graph.graph["fleet"][vehicle] = vehicle_dict 
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
    return {key: node_info[key]["xy"] for key in node_info.keys()}


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


def add_coordinates(graph, info_dict):
    """ Add the node coordinates to the graph """

    graph.graph["coord_dict"] = get_node_coordinates(info_dict)
    return graph


def add_weighted_edges(graph, edges, weights):
    """ Add the given edges to the graph with the specified weights """

    graph.add_weighted_edges_from([(*edge, {"weight": weight})
                                   for ((edge), weight) in zip(edges, weights)])
    return graph


def add_distance_edges(graph, info_dict, symmetric=True, rounding=0, metric="euclidean"):
    """ Compute and add distance weights """

    coordinates = get_node_coordinates(info_dict)
    coordinates = [coordinates[key] for key in sorted(list(coordinates.keys()))]
    weights = compute_pairwise_distances(coordinates, metric=metric,
                                         rounding=rounding, symmetric=True)
    edges = map(tuple, get_complete_edges(info_dict, symmetric=symmetric))
    weighted_edges = [edge + (weight,) for (edge, weight) in zip(edges, weights)]
    graph.add_weighted_edges_from(weighted_edges)
    return graph


def parse_vehicles(filename):
    """ Parse the number of vehicles from the filename """

    return int(filename.split(".")[0].split("-k")[1])


def add_vehicle_count(graph, filename):
    """ Add the number of vehicles to the graph object """

    graph.graph["vehicle_num"] = parse_vehicles(filename)
    return graph


def check_fleet(graph):
    """ Check if the fleet is composed on one vehicle type and duplicate if required """

    vehicle_keys = list(graph.graph["fleet"].keys())
    vehicle_num = graph.graph["vehicle_num"]
    
    if len(vehicle_keys) == 1 and vehicle_num > 1:
        for num in range(graph.graph["vehicle_num"]):
            graph.graph["fleet"][vehicle_keys[0] + num] = graph.graph["fleet"][vehicle_keys[0]]
    return graph

    
def read_vrp_problem_from_xml(filename):
    """ Read an xml vrp problem file into a networkx graph """

    graph = nx.Graph()
    dict = read_vrp_xml(filename)
    graph = add_node_info(graph, dict)
    graph = add_node_requests(graph, dict)
    graph = add_fleet_info(graph, dict)
    graph = add_distance_edges(graph, dict)
    graph = add_vehicle_count(graph, filename)
    graph = check_fleet(graph)
    graph = add_coordinates(graph, dict)
    
    return graph


_metrics = {
    "euclidean": euclidean,
    "manhattan": manhattan,
}
