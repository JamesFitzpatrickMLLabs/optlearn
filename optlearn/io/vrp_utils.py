import itertools

import numpy as np
import networkx as nx

from scipy.spatial.distance import cdist


from optlearn.io import vrp_xml_utils
from optlearn.io import vrp_txt_utils


def euclidean(x, y):
    """ Compute euclidean distance between x and y """

    return np.sqrt(np.sum(np.power(x - y, 2)))


def manhattan(x, y):
    """ Compute manhattan distance between x and y """

    return np.sum(np.abs(x - y))


def latitude_longitude_to_radians(coordinate_pair):
    """ Convert the given coordinate pair to radians """

    difference = coordinate_pair - np.round(coordinate_pair)
    radians = np.pi * (np.round(coordinate_pair) + 5 * difference / 3) / 180

    return radians


def sphere_geodesic_distance(x_radians, y_radians, radius=6378.388):
    """ Compute the shortest distance between the two radian pairs on the given sphere """

    a = np.cos(x_radians[0] - y_radians[0])
    b = np.cos(x_radians[1] - y_radians[1])
    c = np.cos(x_radians[1] + y_radians[1])

    distance = int(radius * np.arccos(0.5 * ((1 + a) * b - (1 - a) * c)) + 1)

    return distance


def geopositional(x, y):
    """ Compute the distance between x and y, where they are lon-lat pairs """

    x_radians = latitude_longitude_to_radians(x)
    y_radians = latitude_longitude_to_radians(y)
    distance = sphere_geodesic_distance(x_radians, y_radians)
    
    return distance
    

def add_nodes_info(graph, info_dict):
    """ Add the nodes information from the dict to the graph """

    attribute_dict = info_dict["network"]["nodes"]
    graph.add_nodes_from(attribute_dict.keys())
    _ = nx.set_node_attributes(graph, attribute_dict)
    
    return graph


def add_network_info(graph, info_dict):
    """ Add the non-node network information from the dict to the graph """

    graph.graph = {
        **graph.graph,
        "decimals": info_dict["network"]["decimals"],
    }

    return graph
    
    
def add_requests_info(graph, info_dict):
    """ Add the requests information from the dict to the graph """

    attribute_dict = info_dict["requests"]
    graph.add_nodes_from(attribute_dict.keys())
    _ = nx.set_node_attributes(graph, attribute_dict)
    
    return graph


def add_fleet_info(graph, info_dict):
    """ Add fleet information to the graph """

    graph.graph = {
        **graph.graph,
        "fleet": info_dict["fleet"],
    }

    return graph


def get_neighbours_symmetric(node, nodes):
    """ Get the indices of the neighbour nodes to the given node """

    return [item for item in nodes if item > node]


def get_neighbours_asymmetric(node, nodes):
    """ Get the indices of the neighbour nodes to the given node """

    return [item for item in nodes if item != node]


def get_nodes_neighbours(info_dict, symmetric=True):
    """ Build an iterator giving each (node, [...neighbours...]) combination """

    nodes = list(info_dict["network"]["nodes"].keys())

    if symmetric:
        func = get_neighbours_symmetric
    if not symmetric:
        func = get_neighbours_asymmetric

    return [(node, func(node, nodes)) for node in nodes[:-1]]


def get_node_neighbour_edges(node_neighbours):
    """ Get the node-neighbour edges for given pairs """

    node, neighbours = node_neighbours
    return [(node, neighbour) for neighbour in neighbours]


def get_node_coordinates(nodes_dict, node_id):
    """ Get the coordinates of the given node from the nodes dictionary """
    
    coordinates = [
        nodes_dict[node_id]["cx"],
        nodes_dict[node_id]["cy"]
    ]

    return coordinates


def get_nodes_coordinates(info_dict):
    """ Get the coordinates of the given nodes """

    nodes_dict = info_dict["network"]["nodes"]
    nodes_coordinates = {
        node_id: get_node_coordinates(nodes_dict, node_id) for node_id in nodes_dict.keys()
    }

    return nodes_coordinates


def compute_pairwise_distances(coordinates, metric, rounding, symmetric=True):
    """ Compute pairwise distances between coordinates """

    distances = cdist(coordinates, coordinates, metric=metric).round(rounding)

    if symmetric:
        return distances[np.triu_indices_from(distances, 1)]
    else:
        return distances.flatten()


def get_complete_edges(info_dict, symmetric=True):
    """ Get all the possible node pairs """

    keys = list(info_dict["network"]["nodes"].keys())
    
    if symmetric:
        return np.array(list(itertools.combinations(keys, 2)))
    else:
        return np.array(list(itertools.permutations(keys, 2)))


def add_coordinates(graph, info_dict):
    """ Add the node coordinates to the graph """

    graph.graph["coord_dict"] = get_nodes_coordinates(info_dict)
    
    return graph


def add_weighted_edges(graph, edges, weights):
    """ Add the given edges to the graph with the specified weights """

    graph.add_weighted_edges_from([(*edge, {"weight": weight})
                                   for ((edge), weight) in zip(edges, weights)])
    return graph


def add_distance_edges(graph, info_dict, symmetric=True, rounding=0, metric="euclidean"):
    """ Compute and add distance weights """

    coordinates = get_nodes_coordinates(info_dict)
    coordinates = [coordinates[key] for key in sorted(list(coordinates.keys()))]
    weights = compute_pairwise_distances(coordinates, metric=metric,
                                         rounding=rounding, symmetric=True)
    edges = map(tuple, get_complete_edges(info_dict, symmetric=symmetric))
    weighted_edges = [edge + (weight,) for (edge, weight) in zip(edges, weights)]
    graph.add_weighted_edges_from(weighted_edges)
    
    return graph


def parse_vehicles_cvrp(filename):
    """ Parse the number of vehicles from the filename (CVRP instances) """

    return int(filename.split(".")[0].split("-k")[1])


def parse_vehicles_evrpnl(filename):
    """ Parse the number of vehicles from the filename (EVRP-NL instances) """

    return None


def parse_vehicles_gvrp(filename):
    """ Parse the number of vehicles from the filename (GVRP instances) """

    num_vehicles = vrp_txt_utils.get_vehicle_num(filename)
    
    return num_vehicles


def add_vehicle_count(graph, filename, filename_parser=None):
    """ Add the number of vehicles to the graph object """

    if filename_parser is None:
        graph.graph["vehicle_num"] = None
    else:
        graph.graph["vehicle_num"] = filename_parser(filename)
    
    return graph


def get_vehicle_keys(graph):
    """ Find the vehicles in the given graph metadata dictionary """

    fleet_keys = list(graph.graph["fleet"].keys())
    vehicle_keys = [item for item in fleet_keys if "vehicle" in item]

    return vehicle_keys


def get_vehicle_number(vehicle_key):
    """ Get the number of the given vehicle from the key """

    vehicle_number = int(vehicle_key.split("_")[-1])

    return vehicle_number


def get_vehicle_numbers(vehicle_keys):
    """ Get the number of the vehicle for each vehicle key given """

    vehicle_numbers = [get_vehicle_number(key) for key in vehicle_keys]

    return vehicle_numbers


def get_max_vehicle_number(vehicle_keys):
    """ Get the maximum vehicle_number """

    vehicle_numbers = get_vehicle_numbers(vehicle_keys)
    max_vehicle_number = max(vehicle_numbers)

    return max_vehicle_number


def duplicate_vehicle(graph, vehicle_key):
    """ Duplicate the given vehicle """

    vehicle_keys = get_vehicle_keys(graph)
    max_vehicle_number = get_max_vehicle_number(vehicle_keys)
    new_vehicle_key = f"vehicle_{max_vehicle_number + 1}"
    graph.graph["fleet"][new_vehicle_key] = graph.graph["fleet"][vehicle_key]

    return graph


def check_fleet(graph):
    """ Check if the fleet is composed on one vehicle type and duplicate if required """

    vehicle_keys = get_vehicle_keys(graph)
    vehicle_num = graph.graph["vehicle_num"]

    if vehicle_num is not None:
        if len(vehicle_keys) == 1 and vehicle_num > 1:
            for num in range(vehicle_num):
                duplicate_vehicle(graph, vehicle_keys[0])
    return graph


def read_vrp_problem(filename, problem_type):
    """ Read an xml vrp problem file into a networkx graph """

    graph = nx.Graph()
    info_dict = _vehicle_readers.get(problem_type)(filename)
    graph = add_nodes_info(graph, info_dict)
    graph = add_network_info(graph, info_dict)
    graph = add_requests_info(graph, info_dict)
    graph = add_fleet_info(graph, info_dict)
    graph = add_distance_edges(graph, info_dict, metric=_vehicle_metrics.get(problem_type))
    graph = add_vehicle_count(graph, filename, _vehicle_num_parsers.get(problem_type))
    graph = check_fleet(graph)
    graph = add_coordinates(graph, info_dict)
    
    return graph


def read_cvrp_problem(filename):
    """ Read an cvrp problem file into a networkx graph """

    graph = read_vrp_problem(filename, problem_type="cvrp")
    
    return graph


def read_gvrp_problem(filename):
    """ Read an gvrp problem file into a networkx graph """

    graph = read_vrp_problem(filename, problem_type="gvrp")
    
    return graph


def read_evrpnl_problem(filename):
    """ Read an  evrpnl problem file into a networkx graph """

    graph = read_vrp_problem(filename, problem_type="evrpnl")
    
    return graph



_metrics = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "geopositional": geopositional,
}

_vehicle_readers = {
    "cvrp": vrp_xml_utils.read_vrp_xml,
    "gvrp": vrp_txt_utils.read_vrp_txt,
    "evrpnl": vrp_xml_utils.read_vrp_xml,

}

_vehicle_metrics = {
    "cvrp": euclidean,
    "gvrp": geopositional,
    "evrpnl": euclidean,

}

_vehicle_num_parsers = {
    "cvrp": parse_vehicles_cvrp,
    "gvrp": parse_vehicles_gvrp,
    "evrpnl": parse_vehicles_evrpnl,
}
