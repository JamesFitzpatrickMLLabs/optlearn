import numpy as np

from scipy.spatial.distance import cdist


_metrics = {
    "euclidean": euclidean,
    "manhattan": manhattan,
}


def read_vrp_xml(fname):
    """ Read and parse the VRP xml file """

    xml_root = read_xml(fname).getroot()
    return parse_vrp_file_xml(xml_root)



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


def get_node_coordinates(info_dict, nodes):
    """ Get the coordinates of the given nodes """

    node_info = info_dict["network"]["node_info"] 
    return [node_info[key]["xy"] for key in nodes]


def compute_pairwise_distances(coordinate, coordinates, metric, rounding):
    """ Compute pairwise distances between coordinates """

    return cdist(coordinate, coordinates, metric=metric).round(rounding)


def compute_distance_weights(info_dict, node, neighbours, symmetric=True):
    """ Compute the distances between the given node and its given neighbours as edge weights """
    
    coordinate = get_node_coordinates(info_dict, [node])
    coordinates = get_node_coordinates(info_dict, neighbours)
    metric = _metrics[info_dict["network"]["metric"]]
    rounding = info_dict["network"]["decimals"]
    return compute_pairwise_distances(coordinate, coordinates, metric, rounding)


def add_weighted_edges(graph, edges, weights):
    """ Add the given edges to the graph with the specified weights """

    graph.add_weighted_edges_from([(*edge, {"weight": weight})
                                   for (edge, weight) in zip(edges, weights)])
    return graph


def add_distance_edges(graph, info_dict, symmetric=True):
    """ Compute and add distance weights """

    node_pairs = list(get_all_pairs(info_dict, symmetric=symmetric))
    node_edges = [get_node_neighbour_edges(node_pair) for node_pair in node_pairs]
    edge_weights = [compute_distance_weights(info_dict, node, neighbours, symmetric)
                    for (node, neighbours) in node_pairs]
    zipped_items = zip(node_edges, edge_weights)
    for item in zipped_items:
        edges, weights = item
        graph = add_weighted_edges(graph, edges, weights[0])
    return graph



test_graph = nx.Graph()
test_dict = read_vrp_xml(fname)
test_graph = add_node_info(test_graph, test_dict)
test_graph = add_node_requests(test_graph, test_dict)
test_graph = add_fleet_info(test_graph, test_dict)
test_graph = add_distance_edges(test_graph, test_dict)
