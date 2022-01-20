import numpy as np
import networkx as nx

from optlearn.graph import graph_utils
from optlearn.feature.vrp import feature_utils


def store_node_attributes(graph, attribute_dict, name):
    """ Store the given node attributes in the graph """

    for node in attribute_dict.keys():
        graph = graph_utils.store_node_attribute(graph, node, attribute_dict[node], name)

    return graph


def get_station_reachable_customers(graph, station, reachability_radius):
    """ Compute customers that are reachable for the given station """

    reachable_customers = feature_utils.get_reachable_customers(
        graph, station, reachability_radius)

    return reachable_customers


def compute_stations_reachable_customers(graph, reachability_radius):
    """ Get the reachable customers for each station in the given graph """

    station_nodes = feature_utils.get_stations(graph)
    reachable_customers = {
        station_node: get_station_reachable_customers(
            graph, station_node, reachability_radius) for station_node in station_nodes
    }
    graph = store_node_attributes(graph, reachable_customers,
                                  "reachable_customers")

    return graph


def get_station_strictly_reachable_customers(graph, station, reachability_radius):
    """ Compute customers that are strictly reachable for the given station """

    strictly_reachable_customers = feature_utils.get_strictly_reachable_customers(
        graph, station, reachability_radius)

    return strictly_reachable_customers


def compute_stations_strictly_reachable_customers(graph, reachability_radius):
    """ Get the strictly reachable customers for each station in the given graph """

    station_nodes = feature_utils.get_stations(graph)
    strictly_reachable_customers = {
        station_node: get_station_strictly_reachable_customers(
            graph, station_node, reachability_radius) for station_node in station_nodes
    }
    graph = store_node_attributes(graph, strictly_reachable_customers,
                                  "strictly_reachable_customers")
    
    return graph


def get_depot_reachable_customers(graph, depot, reachability_radius):
    """ Compute customers that are reachable for the given depot """

    reachable_customers = feature_utils.get_reachable_customers(
        graph, depot, reachability_radius)

    return reachable_customers


def compute_depots_reachable_customers(graph, reachability_radius):
    """ Get the reachable customers for each depot in the given graph """

    depot_nodes = feature_utils.get_depots(graph)
    reachable_customers = {
        depot_node: get_depot_reachable_customers(
            graph, depot_node, reachability_radius) for depot_node in depot_nodes
    }

    graph = store_node_attributes(graph, reachable_customers,
                                  "reachable_customers")

    return graph


def get_depot_strictly_reachable_customers(graph, depot, reachability_radius):
    """ Compute customers that are strictly reachable for the given depot """

    strictly_reachable_customers = feature_utils.get_strictly_reachable_customers(
        graph, depot, reachability_radius)

    return strictly_reachable_customers


def compute_depots_strictly_reachable_customers(graph, reachability_radius):
    """ Get the strictly reachable customers for each depot in the given graph """

    depot_nodes = feature_utils.get_depots(graph)
    strictly_reachable_customers = {
        depot_node: get_depot_strictly_reachable_customers(
            graph, depot_node, reachability_radius) for depot_node in depot_nodes
    }

    graph = store_node_attributes(graph, strictly_reachable_customers,
                                  "strictly_reachable_customers")
    
    return graph


def get_nondepot_reachable_customers(graph, station):
    """ Assuming we know the reachable customers, find which are not depot-reachable """

    depot_nodes = feature_utils.get_depots(graph)
    station_reachable_customers = graph_utils.get_node_attribute(
            graph, station, "reachable_customers")
    depot_reachable_customers = []
    for depot_node in depot_nodes:
        depot_reachable_customers += graph_utils.get_node_attribute(
            graph, depot_node, "reachable_customers")
    nondepot_reachable_customers = [customer for customer in station_reachable_customers
                                    if customer not in depot_reachable_customers]

    return nondepot_reachable_customers


def compute_nondepot_reachable_customers(graph):
    """ Assuming we know the reachable customers, find which are not depot-reachable """

    station_nodes = feature_utils.get_stations(graph)
    nondepot_reachable_customers = {
        station: get_nondepot_reachable_customers(graph, station) for station in station_nodes
    }
    graph = store_node_attributes(graph, nondepot_reachable_customers,
                                  "nondepot_reachable_customers")
    
    return graph


def get_nondepot_strictly_reachable_customers(graph, station):
    """ Assuming we know the reachable customers, find which are not depot-reachable """

    depot_nodes = feature_utils.get_depots(graph)
    station_reachable_customers = graph_utils.get_node_attribute(
            graph, station, "strictly_reachable_customers")
    depot_reachable_customers = []
    for depot_node in depot_nodes:
        depot_reachable_customers += graph_utils.get_node_attribute(
            graph, depot_node, "strictly_reachable_customers")
    nondepot_reachable_customers = [customer for customer in station_reachable_customers
                                    if customer not in depot_reachable_customers]

    return nondepot_reachable_customers


def compute_nondepot_strictly_reachable_customers(graph):
    """ Assuming we know the reachable customers, find which are not depot-reachable """

    station_nodes = feature_utils.get_stations(graph)
    nondepot_reachable_customers = {
        station: get_nondepot_strictly_reachable_customers(graph, station)
        for station in station_nodes
        }

    graph = store_node_attributes(graph, nondepot_reachable_customers,
                                  "nondepot_strictly_reachable_customers")
    
    return graph


def get_unique_reachable_customers(graph, station):
    """ Assuming we know the reachable customers, find which (if any) are unique """

    station_and_depot_nodes = feature_utils.get_stations_and_depots(graph)
    station_and_depot_nodes = [item for item in station_and_depot_nodes if item != station]
    station_reachable_customers = graph_utils.get_node_attribute(
            graph, station, "reachable_customers")
    other_reachable_customers = []
    for other_node in station_and_depot_nodes:
        other_reachable_customers += graph_utils.get_node_attribute(
            graph, other_node, "reachable_customers")
    unique_reachable_customers = [customer for customer in station_reachable_customers
                                    if customer not in other_reachable_customers]

    return unique_reachable_customers


def compute_unique_reachable_customers(graph):
    """ Assuming we know the reachable customers, find which (if any) are unique """

    station_and_depot_nodes = feature_utils.get_stations_and_depots(graph)
    unique_reachable_customers = {
        node: get_unique_reachable_customers(graph, node) for node in station_and_depot_nodes
    }
    graph = store_node_attributes(graph, unique_reachable_customers,
                                  "unique_reachable_customers")
    
    return graph


def get_unique_strictly_reachable_customers(graph, station):
    """ Assuming we know the stricltly reachable customers, find which (if any) are unique """

    station_and_depot_nodes = feature_utils.get_stations_and_depots(graph)
    station_and_depot_nodes = [item for item in station_and_depot_nodes if item != station]
    station_reachable_customers = graph_utils.get_node_attribute(
            graph, station, "strictly_reachable_customers")
    other_reachable_customers = []
    for other_node in station_and_depot_nodes:
        other_reachable_customers += graph_utils.get_node_attribute(
            graph, other_node, "strictly_reachable_customers")
    unique_reachable_customers = [customer for customer in station_reachable_customers
                                    if customer not in other_reachable_customers]

    return unique_reachable_customers


def compute_unique_strictly_reachable_customers(graph):
    """ Assuming we know the strictly reachable customers, find which (if any) are unique """

    station_and_depot_nodes = feature_utils.get_stations_and_depots(graph)
    unique_reachable_customers = {
        node: get_unique_strictly_reachable_customers(graph, node)
        for node in station_and_depot_nodes
    }
    graph = store_node_attributes(graph, unique_reachable_customers,
                                  "unique_strictly_reachable_customers")
    
    return graph


def process_reachability_orbits(graph, reachability_radius):
    """ Process the reachability orbits for all stations and depots """

    graph = compute_stations_reachable_customers(graph, reachability_radius)
    graph = compute_depots_reachable_customers(graph, reachability_radius)
    graph = compute_nondepot_reachable_customers(graph)
    graph = compute_unique_reachable_customers(graph)

    return graph


def process_strict_reachability_orbits(graph, reachability_radius):
    """ Process the reachability orbits for all stations and depots """

    graph = compute_stations_strictly_reachable_customers(graph, reachability_radius)
    graph = compute_depots_strictly_reachable_customers(graph, reachability_radius)
    graph = compute_nondepot_strictly_reachable_customers(graph)
    graph = compute_unique_strictly_reachable_customers(graph)

    return graph


def compute_station_reachability_number(graph, station):
    """ Get the reachability number of a given station """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "reachable_customers")
    reachability_number = len(reachability_set)

    return reachability_number


def compute_station_strict_reachability_number(graph, station):
    """ Get the strict reachability number of a given station """

    strict_reachability_set = graph_utils.get_node_attribute(
        graph, station, "strictly_reachable_customers")
    strict_reachability_number = len(strict_reachability_set)

    return strict_reachability_number


def compute_station_nondepot_reachability_number(graph, station):
    """ Get the reachability number of a given station """

    nondepot_reachability_set = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachable_customers")
    nondepot_reachability_number = len(nondepot_reachability_set)

    return nondepot_reachability_number


def compute_station_nondepot_strict_reachability_number(graph, station):
    """ Get the reachability number of a given station """

    nondepot_strict_reachability_set = graph_utils.get_node_attribute(
        graph, station, "nondepot_strictly_reachable_customers")
    nondepot_strict_reachability_number = len(nondepot_strict_reachability_set)

    return nondepot_strict_reachability_number


def compute_station_unique_reachability_number(graph, station):
    """ Get the reachability number of a given station """

    unique_reachability_set = graph_utils.get_node_attribute(
        graph, station, "unique_reachable_customers")
    unique_reachability_number = len(unique_reachability_set)

    return unique_reachability_number


def compute_station_nondepot_strict_reachability_number(graph, station):
    """ Get the reachability number of a given station """

    unique_strict_reachability_set = graph_utils.get_node_attribute(
        graph, station, "unique_strictly_reachable_customers")
    unique_strict_reachability_number = len(unique_strict_reachability_set)

    return unique_strict_reachability_number


def compute_station_technology(graph, station):
    """ Get the technology of the given station """

    if graph.nodes[station]["cs_type"] == "slow":
        return -1
    elif graph.nodes[station]["cs_type"] == "normal":
        return 0
    elif graph.nodes[station]["cs_type"] == "fast":
        return 1


def get_depot_cosine(graph, node):
    """ Compute cosine of vector pointing from the origin at the depot to the node  """

    depot = feature_utils.get_depot(graph)
    coord_vectors = feature_utils.get_nodes_coordinate_vectors(graph, [node, depot])
    difference_vector = coord_vectors[0] - coord_vectors[1]
    depot_cosine = feature_utils.get_cosine(difference_vector, np.array([1, 0]))

    return depot_cosine


def compute_depot_station_cosines(graph):
    """ Compute the depot-station cosines for each station """

    stations = feature_utils.get_stations(graph)
    depot_station_cosines = {
        station: get_depot_cosine(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, depot_station_cosines, "depot_station_cosine")

    return graph


def get_depot_sine(graph, station):
    """ Compute sine of vector pointing from the origin at the depot to the node  """

    depot_station_cosine = graph_utils.get_node_attribute(
        graph, station, "depot_station_cosine")
    depot_station_sine = np.sin(np.arccos(depot_station_cosine))
    
    return depot_station_sine


def compute_depot_station_sines(graph):
    """ Compute the depot-station sines for each station """

    stations = feature_utils.get_stations(graph)
    depot_station_sines = {
        station: get_depot_sine(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, depot_station_sines, "depot_station_sine")

    return graph


def get_depot_distance(graph, node):
    """ Compute length of vector pointing from the depot to the node  """

    depot = feature_utils.get_depot(graph)
    coord_vectors = feature_utils.get_nodes_coordinate_vectors(graph, [node, depot])
    difference_vector = coord_vectors[0] - coord_vectors[1]
    vector_norm = feature_utils.get_vector_norm(difference_vector)

    return vector_norm


def compute_depot_station_distances(graph):
    """ Compute the depot-station distance for each station """

    stations = feature_utils.get_stations(graph)
    depot_station_distances = {
        station: get_depot_distance(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, depot_station_distances, "depot_station_distance")

    return graph


def get_reachability_set_distances(graph, station):
    """ Given the reachability set, get the distances to the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(graph, station, "reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_distances = feature_utils.compute_distances_to_node(
        graph, station, reachability_set)

    return reachability_distances


def compute_reachability_set_distances(graph):
    """ Given the reachability set, get the distances to the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_distances = {
        station: get_reachability_set_distances(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_distances, "reachability_distances")

    return graph


def get_strict_reachability_set_distances(graph, station):
    """ Given the reachability set, get the distances to the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "strictly_reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_distances = feature_utils.compute_distances_to_node(
        graph, station, reachability_set)

    return reachability_distances


def compute_strict_reachability_set_distances(graph):
    """ Given the reachability set, get the distances to the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_distances = {
        station: get_strict_reachability_set_distances(graph, station) for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_distances, "strict_reachability_distances")

    return graph


def get_nondepot_reachability_set_distances(graph, station):
    """ Given the reachability set, get the distances to the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_distances = feature_utils.compute_distances_to_node(
        graph, station, reachability_set)

    return reachability_distances


def compute_nondepot_reachability_set_distances(graph):
    """ Given the reachability set, get the distances to the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_distances = {
        station: get_nondepot_reachability_set_distances(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_distances, "nondepot_reachability_distances")

    return graph


def get_nondepot_strict_reachability_set_distances(graph, station):
    """ Given the reachability set, get the distances to the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "nondepot_strictly_reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_distances = feature_utils.compute_distances_to_node(
        graph, station, reachability_set)

    return reachability_distances


def compute_nondepot_strict_reachability_set_distances(graph):
    """ Given the reachability set, get the distances to the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_distances = {
        station: get_nondepot_strict_reachability_set_distances(graph, station)
        for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_distances, "nondepot_strict_reachability_distances")

    return graph


def get_unique_reachability_set_distances(graph, station):
    """ Given the reachability set, get the distances to the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "unique_reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_distances = feature_utils.compute_distances_to_node(
        graph, station, reachability_set)

    return reachability_distances


def compute_unique_reachability_set_distances(graph):
    """ Given the reachability set, get the distances to the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_distances = {
        station: get_unique_reachability_set_distances(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_distances, "unique_reachability_distances")

    return graph


def get_unique_strict_reachability_set_distances(graph, station):
    """ Given the reachability set, get the distances to the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "unique_strictly_reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_distances = feature_utils.compute_distances_to_node(
        graph, station, reachability_set)

    return reachability_distances


def compute_unique_strict_reachability_set_distances(graph):
    """ Given the reachability set, get the distances to the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_distances = {
        station: get_unique_strict_reachability_set_distances(graph, station)
        for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_distances, "unique_strict_reachability_distances")

    return graph


def process_reachability_distances(graph):
    """ Process the reachability orbits for all stations and depots """

    graph = compute_depot_station_distances(graph)
    graph = compute_depot_station_cosines(graph)
    graph = compute_depot_station_sines(graph)
    graph = compute_reachability_set_distances(graph)
    graph = compute_nondepot_reachability_set_distances(graph)
    graph = compute_unique_reachability_set_distances(graph)

    return graph


def process_strict_reachability_distances(graph):
    """ Process the reachability orbits for all stations and depots """

    graph = compute_depot_station_distances(graph)
    graph = compute_depot_station_cosines(graph)
    graph = compute_depot_station_sines(graph)
    graph = compute_strict_reachability_set_distances(graph)
    graph = compute_nondepot_strict_reachability_set_distances(graph)
    graph = compute_unique_strict_reachability_set_distances(graph)

    return graph


def get_reachability_set_separations(graph, station):
    """ Given the reachability set, get the separations between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(graph, station, "reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_separations = feature_utils.compute_pairwise_distances(
        graph, reachability_set)

    return reachability_separations


def compute_reachability_set_separations(graph):
    """ Given the reachability set, get the separations between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_separations = {
        station: get_reachability_set_separations(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_separations, "reachability_separations")

    return graph


def get_strict_reachability_set_separations(graph, station):
    """ Given the reachability set, get the separations between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "strictly_reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_separations = feature_utils.compute_pairwise_distances(
        graph, reachability_set)

    return reachability_separations


def compute_strict_reachability_set_separations(graph):
    """ Given the reachability set, get the separations between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_separations = {
        station: get_strict_reachability_set_separations(graph, station) for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_separations, "strict_reachability_separations")

    return graph


def get_nondepot_reachability_set_separations(graph, station):
    """ Given the reachability set, get the separations between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_separations = feature_utils.compute_pairwise_distances(
        graph, reachability_set)

    return reachability_separations


def compute_nondepot_reachability_set_separations(graph):
    """ Given the reachability set, get the separations between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_separations = {
        station: get_nondepot_reachability_set_separations(graph, station) for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_separations, "nondepot_reachability_separations")

    return graph


def get_nondepot_strict_reachability_set_separations(graph, station):
    """ Given the reachability set, get the separations between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "nondepot_strictly_reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_separations = feature_utils.compute_pairwise_distances(
        graph, reachability_set)

    return reachability_separations


def compute_nondepot_strict_reachability_set_separations(graph):
    """ Given the reachability set, get the separations between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_separations = {
        station: get_nondepot_strict_reachability_set_separations(graph, station)
        for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_separations, "nondepot_strict_reachability_separations")

    return graph


def get_unique_reachability_set_separations(graph, station):
    """ Given the reachability set, get the separations between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "unique_reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_separations = feature_utils.compute_pairwise_distances(
        graph, reachability_set)

    return reachability_separations


def compute_unique_reachability_set_separations(graph):
    """ Given the reachability set, get the separations between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_separations = {
        station: get_unique_reachability_set_separations(graph, station) for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_separations, "unique_reachability_separations")

    return graph


def get_unique_strict_reachability_set_separations(graph, station):
    """ Given the reachability set, get the separations between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "unique_strictly_reachable_customers")
    if len(reachability_set) == 0:
        return []
    reachability_separations = feature_utils.compute_pairwise_distances(
        graph, reachability_set)

    return reachability_separations


def compute_unique_strict_reachability_set_separations(graph):
    """ Given the reachability set, get the separations between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_separations = {
        station: get_unique_strict_reachability_set_separations(graph, station)
        for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_separations, "unique_strict_reachability_separations")

    return graph


def process_reachability_separations(graph):
    """ Process the reachability orbits for all stations and depots """

    graph = compute_reachability_set_separations(graph)
    graph = compute_nondepot_reachability_set_separations(graph)
    graph = compute_unique_reachability_set_separations(graph)

    return graph


def process_strict_reachability_separations(graph):
    """ Process the reachability orbits for all stations and depots """

    graph = compute_strict_reachability_set_separations(graph)
    graph = compute_nondepot_strict_reachability_set_separations(graph)
    graph = compute_unique_strict_reachability_set_separations(graph)

    return graph


def get_reachability_set_cosines(graph, station):
    """ Given the reachability set, get the cosines between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(graph, station, "reachable_customers")
    if len(reachability_set) == 0:
        return []
    station_vector = feature_utils.get_node_coordinate_vector(graph, station)
    other_vectors = feature_utils.get_nodes_coordinate_vectors(graph, reachability_set)
    difference_vectors = feature_utils.get_difference_vectors(station_vector, other_vectors)
    reachability_cosines = feature_utils.get_cosines(np.array([0, 1]), difference_vectors)

    return reachability_cosines


def compute_reachability_set_cosines(graph):
    """ Given the reachability set, get the cosines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_cosines = {
        station: get_reachability_set_cosines(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_cosines, "reachability_cosines")

    return graph


def get_strict_reachability_set_cosines(graph, station):
    """ Given the reachability set, get the cosines between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "strictly_reachable_customers")
    if len(reachability_set) == 0:
        return []
    station_vector = feature_utils.get_node_coordinate_vector(graph, station)
    other_vectors = feature_utils.get_nodes_coordinate_vectors(graph, reachability_set)
    difference_vectors = feature_utils.get_difference_vectors(station_vector, other_vectors)
    reachability_cosines = feature_utils.get_cosines(np.array([0, 1]), difference_vectors)

    return reachability_cosines


def compute_strict_reachability_set_cosines(graph):
    """ Given the reachability set, get the cosines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_cosines = {
        station: get_strict_reachability_set_cosines(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_cosines, "strict_reachability_cosines")

    return graph


def get_nondepot_reachability_set_cosines(graph, station):
    """ Given the reachability set, get the cosines between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachable_customers")
    if len(reachability_set) == 0:
        return []
    station_vector = feature_utils.get_node_coordinate_vector(graph, station)
    other_vectors = feature_utils.get_nodes_coordinate_vectors(graph, reachability_set)
    difference_vectors = feature_utils.get_difference_vectors(station_vector, other_vectors)
    reachability_cosines = feature_utils.get_cosines(np.array([0, 1]), difference_vectors)
    
    return reachability_cosines


def compute_nondepot_reachability_set_cosines(graph):
    """ Given the reachability set, get the cosines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_cosines = {
        station: get_nondepot_reachability_set_cosines(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_cosines, "nondepot_reachability_cosines")

    return graph


def get_nondepot_strict_reachability_set_cosines(graph, station):
    """ Given the reachability set, get the cosines between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "nondepot_strictly_reachable_customers")
    if len(reachability_set) == 0:
        return []
    station_vector = feature_utils.get_node_coordinate_vector(graph, station)
    other_vectors = feature_utils.get_nodes_coordinate_vectors(graph, reachability_set)
    difference_vectors = feature_utils.get_difference_vectors(station_vector, other_vectors)
    reachability_cosines = feature_utils.get_cosines(np.array([0, 1]), difference_vectors)

    return reachability_cosines


def compute_nondepot_strict_reachability_set_cosines(graph):
    """ Given the reachability set, get the cosines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_cosines = {
        station: get_nondepot_strict_reachability_set_cosines(graph, station)
        for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_cosines, "nondepot_strict_reachability_cosines")

    return graph


def get_unique_reachability_set_cosines(graph, station):
    """ Given the reachability set, get the cosines between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "unique_reachable_customers")
    if len(reachability_set) == 0:
        return []
    station_vector = feature_utils.get_node_coordinate_vector(graph, station)
    other_vectors = feature_utils.get_nodes_coordinate_vectors(graph, reachability_set)
    difference_vectors = feature_utils.get_difference_vectors(station_vector, other_vectors)
    reachability_cosines = feature_utils.get_cosines(np.array([0, 1]), difference_vectors)

    return reachability_cosines


def compute_unique_reachability_set_cosines(graph):
    """ Given the reachability set, get the cosines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_cosines = {
        station: get_unique_reachability_set_cosines(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_cosines, "unique_reachability_cosines")

    return graph


def get_unique_strict_reachability_set_cosines(graph, station):
    """ Given the reachability set, get the cosines between the reachable_customers """

    reachability_set = graph_utils.get_node_attribute(
        graph, station, "unique_strictly_reachable_customers")
    if len(reachability_set) == 0:
        return []
    station_vector = feature_utils.get_node_coordinate_vector(graph, station)
    other_vectors = feature_utils.get_nodes_coordinate_vectors(graph, reachability_set)
    difference_vectors = feature_utils.get_difference_vectors(station_vector, other_vectors)
    reachability_cosines = feature_utils.get_cosines(np.array([0, 1]), difference_vectors)

    return reachability_cosines


def compute_unique_strict_reachability_set_cosines(graph):
    """ Given the reachability set, get the cosines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_cosines = {
        station: get_unique_strict_reachability_set_cosines(graph, station)
        for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_cosines, "unique_strict_reachability_cosines")

    return graph


def process_reachability_cosines(graph):
    """ Process the reachability orbits for all stations and depots """

    graph = compute_reachability_set_cosines(graph)
    graph = compute_nondepot_reachability_set_cosines(graph)
    graph = compute_unique_reachability_set_cosines(graph)

    return graph


def process_strict_reachability_cosines(graph):
    """ Process the reachability orbits for all stations and depots """

    graph = compute_strict_reachability_set_cosines(graph)
    graph = compute_nondepot_strict_reachability_set_cosines(graph)
    graph = compute_unique_strict_reachability_set_cosines(graph)

    return graph


def get_reachability_set_sines(graph, station):
    """ Given the reachability set, get the sines between the reachable_customers """

    reachability_cosines = graph_utils.get_node_attribute(graph, station, "reachability_cosines")
    if len(reachability_cosines) == 0:
        return []
    reachability_sines = np.sin(np.arccos(reachability_cosines))

    return reachability_sines


def compute_reachability_set_sines(graph):
    """ Given the reachability set, get the sines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_sines = {
        station: get_reachability_set_sines(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_sines, "reachability_sines")

    return graph


def get_strict_reachability_set_sines(graph, station):
    """ Given the reachability set, get the sines between the reachable_customers """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "strict_reachability_cosines")
    if len(reachability_cosines) == 0:
        return []
    reachability_sines = np.sin(np.arccos(reachability_cosines))

    return reachability_sines


def compute_strict_reachability_set_sines(graph):
    """ Given the reachability set, get the sines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_sines = {
        station: get_strict_reachability_set_sines(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_sines, "strict_reachability_sines")

    return graph


def get_nondepot_reachability_set_sines(graph, station):
    """ Given the reachability set, get the sines between the reachable_customers """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachability_cosines")
    if len(reachability_cosines) == 0:
        return []
    reachability_sines = np.sin(np.arccos(reachability_cosines))

    return reachability_sines


def compute_nondepot_reachability_set_sines(graph):
    """ Given the reachability set, get the sines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_sines = {
        station: get_nondepot_reachability_set_sines(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_sines, "nondepot_reachability_sines")

    return graph


def get_nondepot_strict_reachability_set_sines(graph, station):
    """ Given the reachability set, get the sines between the reachable_customers """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "nondepot_strict_reachability_cosines")
    if len(reachability_cosines) == 0:
        return []
    reachability_sines = np.sin(np.arccos(reachability_cosines))

    return reachability_sines


def compute_nondepot_strict_reachability_set_sines(graph):
    """ Given the reachability set, get the sines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_sines = {
        station: get_nondepot_strict_reachability_set_sines(graph, station) for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_sines, "nondepot_strict_reachability_sines")

    return graph


def get_unique_reachability_set_sines(graph, station):
    """ Given the reachability set, get the sines between the reachable_customers """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "unique_reachability_cosines")
    if len(reachability_cosines) == 0:
        return []
    reachability_sines = np.sin(np.arccos(reachability_cosines))

    return reachability_sines


def compute_unique_reachability_set_sines(graph):
    """ Given the reachability set, get the sines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_sines = {
        station: get_unique_reachability_set_sines(graph, station) for station in stations
    }
    graph = store_node_attributes(graph, reachability_sines, "unique_reachability_sines")

    return graph


def get_unique_strict_reachability_set_sines(graph, station):
    """ Given the reachability set, get the sines between the reachable_customers """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "unique_strict_reachability_cosines")
    if len(reachability_cosines) == 0:
        return []
    reachability_sines = np.sin(np.arccos(reachability_cosines))

    return reachability_sines


def compute_unique_strict_reachability_set_sines(graph):
    """ Given the reachability set, get the sines between the reachable_customers """

    stations = feature_utils.get_stations(graph)
    reachability_sines = {
        station: get_unique_strict_reachability_set_sines(graph, station) for station in stations
    }
    graph = store_node_attributes(
        graph, reachability_sines, "unique_strict_reachability_sines")

    return graph


def process_reachability_sines(graph):
    """ Process the reachability orbits for all stations and depots """

    graph = compute_reachability_set_sines(graph)
    graph = compute_nondepot_reachability_set_sines(graph)
    graph = compute_unique_reachability_set_sines(graph)

    return graph


def process_strict_reachability_sines(graph):
    """ Process the reachability orbits for all stations and depots """

    graph = compute_strict_reachability_set_sines(graph)
    graph = compute_nondepot_strict_reachability_set_sines(graph)
    graph = compute_unique_strict_reachability_set_sines(graph)

    return graph


def process_reachabilities(graph, reachability_radius):
    """ Process the reachability orbits of the given graph """

    graph = process_reachability_orbits(graph, reachability_radius)
    graph = process_reachability_distances(graph)
    graph = process_reachability_separations(graph)
    graph = process_reachability_cosines(graph)
    graph = process_reachability_sines(graph)

    return graph


def process_strict_reachabilities(graph, reachability_radius):
    """ Process the strict reachability orbits of the given graph """

    graph = process_strict_reachability_orbits(graph, reachability_radius)
    graph = process_strict_reachability_distances(graph)
    graph = process_strict_reachability_separations(graph)
    graph = process_strict_reachability_cosines(graph)
    graph = process_strict_reachability_sines(graph)

    return graph


def get_reachability_number(graph, station):
    """ Get the reachability number of a given station """

    reachable_customers = graph_utils.get_node_attribute(graph, station, "reachable_customers")
    reachability_number = len(reachable_customers)

    return reachability_number


def get_strict_reachability_number(graph, station):
    """ Get the strict reachability number of a given station """

    strictly_reachable_customers = graph_utils.get_node_attribute(
        graph, station, "strictly_reachable_customers")
    strict_reachability_number = len(strictly_reachable_customers)

    return strict_reachability_number


def get_nondepot_reachability_number(graph, station):
    """ Get the nondepot reachability number of a given station """

    nondepot_reachable_customers = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachable_customers")
    nondepot_reachability_number = len(nondepot_reachable_customers)

    return nondepot_reachability_number


def get_nondepot_strict_reachability_number(graph, station):
    """ Get the strict nondepot reachability number of a given station """

    nondepot_strictly_reachable_customers = graph_utils.get_node_attribute(
        graph, station, "nondepot_strictly_reachable_customers")
    nondepot_strict_reachability_number = len(nondepot_strictly_reachable_customers)

    return nondepot_strict_reachability_number


def get_unique_reachability_number(graph, station):
    """ Get the unique reachability number of a given station """

    unique_reachable_customers = graph_utils.get_node_attribute(
        graph, station, "unique_reachable_customers")
    unique_reachability_number = len(unique_reachable_customers)

    return unique_reachability_number


def get_unique_strict_reachability_number(graph, station):
    """ Get the strict unique reachability number of a given station """

    unique_strictly_reachable_customers = graph_utils.get_node_attribute(
        graph, station, "unique_strictly_reachable_customers")
    unique_strict_reachability_number = len(unique_strictly_reachable_customers)

    return unique_strict_reachability_number


def compute_normalised_reachability_number(graph, station):
    """ Compute the normalised reachability number """

    customers = feature_utils.get_customers(graph)
    reachability_number = get_reachability_number(graph, station)
    normalised_reachability_number = reachability_number / len(customers)

    return normalised_reachability_number


def compute_normalised_strict_reachability_number(graph, station):
    """ Compute the normalised strict reachability number """

    customers = feature_utils.get_customers(graph)
    reachability_number = get_strict_reachability_number(graph, station)
    normalised_reachability_number = reachability_number / len(customers)

    return normalised_reachability_number


def compute_normalised_nondepot_reachability_number(graph, station):
    """ Compute the normalised reachability number """

    customers = feature_utils.get_customers(graph)
    reachability_number = get_nondepot_reachability_number(graph, station)
    normalised_reachability_number = reachability_number / len(customers)

    return normalised_reachability_number


def compute_normalised_nondepot_strict_reachability_number(graph, station):
    """ Compute the normalised strict reachability number """

    customers = feature_utils.get_customers(graph)
    reachability_number = get_nondepot_strict_reachability_number(graph, station)
    normalised_reachability_number = reachability_number / len(customers)

    return normalised_reachability_number


def compute_normalised_unique_reachability_number(graph, station):
    """ Compute the normalised reachability number """

    customers = feature_utils.get_customers(graph)
    reachability_number = get_unique_reachability_number(graph, station)
    normalised_reachability_number = reachability_number / len(customers)

    return normalised_reachability_number


def compute_normalised_unique_strict_reachability_number(graph, station):
    """ Compute the normalised strict reachability number """

    customers = feature_utils.get_customers(graph)
    reachability_number = get_unique_strict_reachability_number(graph, station)
    normalised_reachability_number = reachability_number / len(customers)

    return normalised_reachability_number


def compute_reachability_mean_cosine(graph, station):
    """ Compute the mean cosine of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    mean_cosine = np.mean(reachability_cosines)

    return mean_cosine


def compute_reachability_std_cosine(graph, station):
    """ Compute the standard deviation of cosines of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    std_cosine = np.std(reachability_cosines)

    return std_cosine


def compute_reachability_mean_sine(graph, station):
    """ Compute the mean sine of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    mean_sine = np.mean(reachability_sines)

    return mean_sine


def compute_reachability_std_sine(graph, station):
    """ Compute the standard deviation of sines of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    std_sine = np.std(reachability_sines)

    return std_sine


def compute_strict_reachability_mean_cosine(graph, station):
    """ Compute the mean cosine of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "strict_reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    mean_cosine = np.mean(reachability_cosines)

    return mean_cosine


def compute_strict_reachability_std_cosine(graph, station):
    """ Compute the standard deviation of cosines of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "strict_reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    std_cosine = np.std(reachability_cosines)

    return std_cosine


def compute_strict_reachability_mean_sine(graph, station):
    """ Compute the mean sine of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "strict_reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    mean_sine = np.mean(reachability_sines)

    return mean_sine


def compute_strict_reachability_std_sine(graph, station):
    """ Compute the standard deviation of sines of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "strict_reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    std_sine = np.std(reachability_sines)

    return std_sine


def compute_nondepot_reachability_mean_cosine(graph, station):
    """ Compute the mean cosine of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    mean_cosine = np.mean(reachability_cosines)

    return mean_cosine


def compute_nondepot_reachability_std_cosine(graph, station):
    """ Compute the standard deviation of cosines of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    std_cosine = np.std(reachability_cosines)

    return std_cosine


def compute_nondepot_reachability_mean_sine(graph, station):
    """ Compute the mean sine of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    mean_sine = np.mean(reachability_sines)

    return mean_sine


def compute_nondepot_reachability_std_sine(graph, station):
    """ Compute the standard deviation of sines of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    std_sine = np.std(reachability_sines)

    return std_sine


def compute_nondepot_strict_reachability_mean_cosine(graph, station):
    """ Compute the mean cosine of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "nondepot_strict_reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    mean_cosine = np.mean(reachability_cosines)

    return mean_cosine


def compute_nondepot_strict_reachability_std_cosine(graph, station):
    """ Compute the standard deviation of cosines of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "nondepot_strict_reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    std_cosine = np.std(reachability_cosines)

    return std_cosine


def compute_nondepot_strict_reachability_mean_sine(graph, station):
    """ Compute the mean sine of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "nondepot_strict_reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    mean_sine = np.mean(reachability_sines)

    return mean_sine


def compute_nondepot_strict_reachability_std_sine(graph, station):
    """ Compute the standard deviation of sines of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "nondepot_strict_reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    std_sine = np.std(reachability_sines)

    return std_sine


def compute_unique_reachability_mean_cosine(graph, station):
    """ Compute the mean cosine of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "unique_reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    mean_cosine = np.mean(reachability_cosines)

    return mean_cosine


def compute_unique_reachability_std_cosine(graph, station):
    """ Compute the standard deviation of cosines of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "unique_reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    std_cosine = np.std(reachability_cosines)

    return std_cosine


def compute_unique_reachability_mean_sine(graph, station):
    """ Compute the mean sine of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "unique_reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    mean_sine = np.mean(reachability_sines)

    return mean_sine


def compute_unique_reachability_std_sine(graph, station):
    """ Compute the standard deviation of sines of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "unique_reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    std_sine = np.std(reachability_sines)

    return std_sine


def compute_unique_strict_reachability_mean_cosine(graph, station):
    """ Compute the mean cosine of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "unique_strict_reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    mean_cosine = np.mean(reachability_cosines)

    return mean_cosine


def compute_unique_strict_reachability_std_cosine(graph, station):
    """ Compute the standard deviation of cosines of the given station's reachability set """

    reachability_cosines = graph_utils.get_node_attribute(
        graph, station, "unique_strict_reachability_cosines")
    if len(reachability_cosines) == 0:
        return -1
    std_cosine = np.std(reachability_cosines)

    return std_cosine


def compute_unique_strict_reachability_mean_sine(graph, station):
    """ Compute the mean sine of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "unique_strict_reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    mean_sine = np.mean(reachability_sines)

    return mean_sine


def compute_unique_strict_reachability_std_sine(graph, station):
    """ Compute the standard deviation of sines of the given station's reachability set """

    reachability_sines = graph_utils.get_node_attribute(
        graph, station, "unique_strict_reachability_sines")
    if len(reachability_sines) == 0:
        return -1
    std_sine = np.std(reachability_sines)

    return std_sine


def compute_reachability_mean_distance(graph, station, reachability_radius):
    """ Compute the mean distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    mean_distance = np.mean(reachability_distances) / reachability_radius

    return mean_distance


def compute_reachability_std_distance(graph, station, reachability_radius):
    """ Compute the standard deviation of distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    std_distance = np.std(reachability_distances) / reachability_radius

    return std_distance


def compute_reachability_mean_separation(graph, station, reachability_radius):
    """ Compute the mean separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    mean_separation = np.mean(reachability_separations) / reachability_radius

    return mean_separation


def compute_reachability_std_separation(graph, station, reachability_radius):
    """ Compute the standard deviation of separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    std_separation = np.std(reachability_separations) / reachability_radius

    return std_separation


def compute_strict_reachability_mean_distance(graph, station, reachability_radius):
    """ Compute the mean distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "strict_reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    mean_distance = np.mean(reachability_distances) / reachability_radius

    return mean_distance


def compute_strict_reachability_std_distance(graph, station, reachability_radius):
    """ Compute the standard deviation of distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "strict_reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    std_distance = np.std(reachability_distances) / reachability_radius

    return std_distance


def compute_strict_reachability_mean_separation(graph, station, reachability_radius):
    """ Compute the mean separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "strict_reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    mean_separation = np.mean(reachability_separations) / reachability_radius

    return mean_separation


def compute_strict_reachability_std_separation(graph, station, reachability_radius):
    """ Compute the standard deviation of separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "strict_reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    std_separation = np.std(reachability_separations) / reachability_radius

    return std_separation


def compute_nondepot_reachability_mean_distance(graph, station, reachability_radius):
    """ Compute the mean distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    mean_distance = np.mean(reachability_distances) / reachability_radius

    return mean_distance


def compute_nondepot_reachability_std_distance(graph, station, reachability_radius):
    """ Compute the standard deviation of distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    std_distance = np.std(reachability_distances) / reachability_radius

    return std_distance


def compute_nondepot_reachability_mean_separation(graph, station, reachability_radius):
    """ Compute the mean separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    mean_separation = np.mean(reachability_separations) / reachability_radius

    return mean_separation


def compute_nondepot_reachability_std_separation(graph, station, reachability_radius):
    """ Compute the standard deviation of separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "nondepot_reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    std_separation = np.std(reachability_separations) / reachability_radius

    return std_separation


def compute_nondepot_strict_reachability_mean_distance(graph, station, reachability_radius):
    """ Compute the mean distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "nondepot_strict_reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    mean_distance = np.mean(reachability_distances) / reachability_radius

    return mean_distance


def compute_nondepot_strict_reachability_std_distance(graph, station, reachability_radius):
    """ Compute the standard deviation of distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "nondepot_strict_reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    std_distance = np.std(reachability_distances) / reachability_radius

    return std_distance


def compute_nondepot_strict_reachability_mean_separation(graph, station, reachability_radius):
    """ Compute the mean separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "nondepot_strict_reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    mean_separation = np.mean(reachability_separations) / reachability_radius

    return mean_separation


def compute_nondepot_strict_reachability_std_separation(graph, station, reachability_radius):
    """ Compute the standard deviation of separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "nondepot_strict_reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    std_separation = np.std(reachability_separations) / reachability_radius

    return std_separation


def compute_unique_reachability_mean_distance(graph, station, reachability_radius):
    """ Compute the mean distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "unique_reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    mean_distance = np.mean(reachability_distances) / reachability_radius

    return mean_distance


def compute_unique_reachability_std_distance(graph, station, reachability_radius):
    """ Compute the standard deviation of distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "unique_reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    std_distance = np.std(reachability_distances) / reachability_radius

    return std_distance


def compute_unique_reachability_mean_separation(graph, station, reachability_radius):
    """ Compute the mean separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "unique_reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    mean_separation = np.mean(reachability_separations) / reachability_radius

    return mean_separation


def compute_unique_reachability_std_separation(graph, station, reachability_radius):
    """ Compute the standard deviation of separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "unique_reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    std_separation = np.std(reachability_separations) / reachability_radius

    return std_separation


def compute_unique_strict_reachability_mean_distance(graph, station, reachability_radius):
    """ Compute the mean distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "unique_strict_reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    mean_distance = np.mean(reachability_distances) / reachability_radius

    return mean_distance


def compute_unique_strict_reachability_std_distance(graph, station, reachability_radius):
    """ Compute the standard deviation of distance from the reachable customers to the station """

    reachability_distances = graph_utils.get_node_attribute(
        graph, station, "unique_strict_reachability_distances")
    if len(reachability_distances) == 0:
        return -1
    std_distance = np.std(reachability_distances) / reachability_radius

    return std_distance


def compute_unique_strict_reachability_mean_separation(graph, station, reachability_radius):
    """ Compute the mean separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "unique_strict_reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    mean_separation = np.mean(reachability_separations) / reachability_radius

    return mean_separation


def compute_unique_strict_reachability_std_separation(graph, station, reachability_radius):
    """ Compute the standard deviation of separation from the reachable customers to the station """

    reachability_separations = graph_utils.get_node_attribute(
        graph, station, "unique_strict_reachability_separations")
    if len(reachability_separations) == 0:
        return -1
    std_separation = np.std(reachability_separations) / reachability_radius

    return std_separation


def compute_normalised_station_depot_distance(graph, station, reachability_radius):
    """ Compute the normalised distance from the station to the depot """

    depot_distance = graph_utils.get_node_attribute(graph, station, "depot_station_distance")
    normalised_distance = depot_distance / reachability_radius
    if normalised_distance > 1:
        normalised_distance = -1
        
    return normalised_distance


def get_closest_station(graph, node):
    """Get the closest station to the given node """
    
    stations = feature_utils.get_stations(graph)
    if len(stations) == 1 and node in stations:
        return -1
    stations = [item for item in stations if item != node]
    closest_station = feature_utils.get_closest_node(graph, node, stations)

    return closest_station


def get_normalised_closest_station_distance(graph, node, reachability_radius):
    """ Get the normalised distance to the closest station """

    closest_station = get_closest_station(graph, node)
    if closest_station == -1:
        return -1
    distance = feature_utils.get_shortest_distance(graph, node, closest_station)
    normalised_distance = distance / reachability_radius
    if normalised_distance >= 1:
        return -1

    return normalised_distance


def get_closest_station_cosine(graph, station):
    """ Get the normalised cosine for the closest station """

    closest_station = get_closest_station(graph, station)
    cosine = graph_utils.get_node_attribute(graph, closest_station, "depot_station_cosine")
    
    return cosine


def get_closest_station_sine(graph, station):
    """ Get the normalised sine for the closest station """

    closest_station = get_closest_station(graph, station)
    sine = graph_utils.get_node_attribute(graph, closest_station, "depot_station_sine")
    
    return sine


def remove_unreachable_edges(graph, reachability_radius):
    """ Remove edges that are not reachable (use only for clone graphs!) """

    edges, weights = graph_utils.get_all_edges(graph), graph_utils.get_all_weights(graph)
    removeable_edges = [edge for (edge, weight) in zip(edges, weights)
                        if weight > reachability_radius]
    graph.remove_edges_from(removeable_edges)

    return graph


def remove_station_strictly_unreachable_edges(graph, reachability_radius):
    """ Remove station edges that are not strictly reachable (use only for clone graphs!) """

    stations, customers = feature_utils.get_stations(graph), feature_utils.get_customers(graph)
    station_customer_edges = [edge for edge in graph_utils.get_all_edges(graph)
                              if edge[0] in stations and edge[1] in customers]
    customer_station_edges = [edge for edge in graph_utils.get_all_edges(graph)
                              if edge[1] in stations and edge[0] in customers]
    edges = station_customer_edges + customer_station_edges
    weights = graph_utils.get_edges_weights(graph, edges)
    removeable_edges = [edge for (edge, weight) in zip(edges, weights)
                        if weight > reachability_radius / 2]
    graph.remove_edges_from(removeable_edges)

    return graph


def get_strict_eigenvector_centrality(graph, reachability_radius):
    """ Get the eigenvector centrality with only strictly reachable edges """

    clone_graph = graph_utils.clone_graph(graph)
    clone_graph = remove_unreachable_edges(clone_graph, reachability_radius)
    clone_graph = remove_station_strictly_unreachable_edges(clone_graph, reachability_radius)
    eigenvector_centralities = nx.eigenvector_centrality(clone_graph, weight="weight")

    return eigenvector_centralities


def get_strict_closeness_centrality(graph, reachability_radius):
    """ Get the eigenvector centrality with only strictly reachable edges """

    clone_graph = graph_utils.clone_graph(graph)
    clone_graph = remove_unreachable_edges(clone_graph, reachability_radius)
    clone_graph = remove_station_strictly_unreachable_edges(clone_graph, reachability_radius)
    closeness_centralities = {
        node: nx.closeness_centrality(clone_graph, node, distance="weight")
        for node in clone_graph.nodes
    }

    return closeness_centralities


def get_strict_betweenness_centrality(graph, reachability_radius):
    """ Get the eigenvector centrality with only strictly reachable edges """

    clone_graph = graph_utils.clone_graph(graph)
    clone_graph = remove_unreachable_edges(clone_graph, reachability_radius)
    clone_graph = remove_station_strictly_unreachable_edges(clone_graph, reachability_radius)
    betweenness_centralities = nx.betweenness_centrality(clone_graph, weight="weight")

    return betweenness_centralities



def compute_station_reachability_features(graph, station, reachability_radius):
    """ Compute the reachability features for a given station """

    features = np.array([
        compute_normalised_reachability_number(graph, station),
        compute_normalised_nondepot_reachability_number(graph, station),
        compute_normalised_unique_reachability_number(graph, station),
        compute_reachability_mean_cosine(graph, station),
        compute_reachability_std_cosine(graph, station),
        compute_reachability_mean_sine(graph, station),
        compute_reachability_std_sine(graph, station),
        compute_nondepot_reachability_mean_cosine(graph, station),
        compute_nondepot_reachability_std_cosine(graph, station),
        compute_nondepot_reachability_mean_sine(graph, station),
        compute_nondepot_reachability_std_sine(graph, station),
        compute_unique_reachability_mean_cosine(graph, station),
        compute_unique_reachability_std_cosine(graph, station),
        compute_unique_reachability_mean_sine(graph, station),
        compute_unique_reachability_std_sine(graph, station),
        compute_reachability_mean_distance(graph, station, reachability_radius),
        compute_reachability_std_distance(graph, station, reachability_radius),
        compute_reachability_mean_separation(graph, station, reachability_radius),
        compute_reachability_std_separation(graph, station, reachability_radius),
        compute_nondepot_reachability_mean_distance(graph, station, reachability_radius),
        compute_nondepot_reachability_std_distance(graph, station, reachability_radius),
        compute_nondepot_reachability_mean_separation(graph, station, reachability_radius),
        compute_nonepot_reachability_std_separation(graph, station, reachability_radius),
        compute_unique_reachability_mean_distance(graph, station, reachability_radius),
        compute_unique_reachability_std_distance(graph, station, reachability_radius),
        compute_unique_reachability_mean_separation(graph, station, reachability_radius),
        compute_nonepot_reachability_std_separation(graph, station, reachability_radius),
        compute_normalised_station_depot_distance(graph, station, reachability_radius),
        compute_station_technology(graph, station),
        get_depot_cosine(graph, station),
        get_depot_sine(graph, station),
        get_normalised_closest_station_distance(graph, station, reachability_radius),
        get_closest_station_cosine(graph, station),
        get_closest_station_sine(graph, station),
    ])

    return features


def compute_station_strict_reachability_features(graph, station, reachability_radius):
    """ Compute the strict reachability features for a given station """

    features = np.array([
        compute_normalised_strict_reachability_number(graph, station),
        compute_normalised_nondepot_strict_reachability_number(graph, station),
        compute_normalised_unique_strict_reachability_number(graph, station),
        compute_strict_reachability_mean_cosine(graph, station),
        compute_strict_reachability_std_cosine(graph, station),
        compute_strict_reachability_mean_sine(graph, station),
        compute_strict_reachability_std_sine(graph, station),
        compute_nondepot_strict_reachability_mean_cosine(graph, station),
        compute_nondepot_strict_reachability_std_cosine(graph, station),
        compute_nondepot_strict_reachability_mean_sine(graph, station),
        compute_nondepot_strict_reachability_std_sine(graph, station),
        compute_unique_strict_reachability_mean_cosine(graph, station),
        compute_unique_strict_reachability_std_cosine(graph, station),
        compute_unique_strict_reachability_mean_sine(graph, station),
        compute_unique_strict_reachability_std_sine(graph, station),
        compute_strict_reachability_mean_distance(graph, station, reachability_radius),
        compute_strict_reachability_std_distance(graph, station, reachability_radius),
        compute_strict_reachability_mean_separation(graph, station, reachability_radius),
        compute_strict_reachability_std_separation(graph, station, reachability_radius),
        compute_nondepot_strict_reachability_mean_distance(graph, station, reachability_radius),
        compute_nondepot_strict_reachability_std_distance(graph, station, reachability_radius),
        compute_nondepot_strict_reachability_mean_separation(graph, station, reachability_radius),
        compute_nondepot_strict_reachability_std_separation(graph, station, reachability_radius),
        compute_unique_strict_reachability_mean_distance(graph, station, reachability_radius),
        compute_unique_strict_reachability_std_distance(graph, station, reachability_radius),
        compute_unique_strict_reachability_mean_separation(graph, station, reachability_radius),
        compute_unique_strict_reachability_std_separation(graph, station, reachability_radius),
        compute_normalised_station_depot_distance(graph, station, reachability_radius),
        compute_station_technology(graph, station),
        get_depot_cosine(graph, station),
        get_depot_sine(graph, station),
        get_normalised_closest_station_distance(graph, station, reachability_radius),
        get_closest_station_cosine(graph, station),
        get_closest_station_sine(graph, station),
    ])

    return features


node_functions = [
        compute_normalised_strict_reachability_number,
        compute_normalised_nondepot_strict_reachability_number,
        compute_normalised_unique_strict_reachability_number,
        compute_strict_reachability_mean_cosine,
        compute_strict_reachability_std_cosine,
        compute_strict_reachability_mean_sine,
        compute_strict_reachability_std_sine,
        compute_nondepot_strict_reachability_mean_cosine,
        compute_nondepot_strict_reachability_std_cosine,
        compute_nondepot_strict_reachability_mean_sine,
        compute_nondepot_strict_reachability_std_sine,
        compute_unique_strict_reachability_mean_cosine,
        compute_unique_strict_reachability_std_cosine,
        compute_unique_strict_reachability_mean_sine,
        compute_unique_strict_reachability_std_sine,
        compute_strict_reachability_mean_distance,
        compute_strict_reachability_std_distance,
        compute_strict_reachability_mean_separation,
        compute_strict_reachability_std_separation,
        compute_nondepot_strict_reachability_mean_distance,
        compute_nondepot_strict_reachability_std_distance,
        compute_nondepot_strict_reachability_mean_separation,
        compute_nondepot_strict_reachability_std_separation,
        compute_unique_strict_reachability_mean_distance,
        compute_unique_strict_reachability_std_distance,
        compute_unique_strict_reachability_mean_separation,
        compute_unique_strict_reachability_std_separation,
        compute_normalised_station_depot_distance,
        compute_station_technology,
        get_depot_cosine,
        get_depot_sine,
        get_normalised_closest_station_distance,
        get_closest_station_cosine,
        get_closest_station_sine,
]

graph_functions = [
    get_strict_eigenvector_centrality,
    get_strict_closeness_centrality,
    get_strict_betweenness_centrality,
]
