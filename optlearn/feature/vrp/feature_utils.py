import numpy as np
import networkx as nx

from scipy.spatial.distance import cdist

from optlearn.graph import graph_utils


def get_stations(graph):
    """ Get the stations of the graph """

    stations = graph.graph["node_types"]["station"]

    return stations


def get_depot(graph):
    """ Get the depot of the graph """

    depot = graph.graph["node_types"]["depot"][0]

    return depot


def get_depots(graph):
    """ Get the depot of the graph """

    depots = graph.graph["node_types"]["depot"]

    return depots


def get_customers(graph):
    """ Get the customers of the graph """

    customers = graph.graph["node_types"]["customer"]

    return customers


def get_stations_and_depots(graph):
    """ Get the stations and depots """

    stations = get_stations(graph)
    depots = get_depots(graph)
    stations_and_depots = stations + depots

    return stations_and_depots


def get_prime_stations(graph):
    """ Get only the prime stations, no copies """

    stations = get_stations(graph)
    is_primes = [graph.nodes[station].get("prime_node") for station in stations]
    primes = [station
              for (station, is_prime) in zip(stations, is_primes) if is_prime is None]

    return primes


def get_clone_stations(graph):
    """ Get the clone stations """

    stations = get_stations(graph)
    primes = get_prime_stations(graph)
    clones = [station for station in stations if station not in primes]

    return clones


def get_station_prime(graph, station_node):
    """ Get the prime station for the given station """

    prime = graph.nodes[station_node].get("prime_node")

    return prime


def get_stations_primes(graph, station_nodes):
    """ Get the prime stations for the given stations """

    primes = [get_station_prime(graph, station_node) for station_node in station_nodes]

    return primes


def get_station_clones(graph, station_node):
    """ Get the clones of the given station, if there are any """

    stations = get_stations(graph)
    station_primes = get_stations_primes(graph, stations)
    clones = [station for (station, prime) in zip(stations, station_primes)
              if prime == station_node]

    return clones


def get_node_coordinate(graph, node):
    """ Get the coordinate for a given node """

    coord = graph.graph["coord_dict"][node]

    return coord


def get_nodes_coordinates(graph, nodes):
    """ Get the coordinate for the given nodes """

    coords = [get_node_coordinate(graph, node) for node in nodes]

    return coords


def get_node_coordinate_vector(graph, node):
    """ Get the coordinate vector for a given node """

    coord = get_node_coordinate(graph, node)
    coord_vector = np.array(coord)

    return coord_vector


def get_nodes_coordinate_vectors(graph, nodes):
    """ Get the coordinate vectors for the given nodes """

    coords = get_nodes_coordinates(graph, nodes)
    coord_vectors = np.array(coords)

    return coord_vectors


def get_edge_vector(graph, edge):
    """ Get the vector pointing from i to j """

    node_coord_vectors = get_nodes_coordinate_vectors(graph, edge)
    edge_vector = np.diff(node_coord_vectors, axis=0).squeeze()
    
    return edge_vector


def get_edge_vectors(graph, edges):
    """ Get the vectors pointing from i to j for each edge """

    nodes_coord_vectors = [get_nodes_coordinate_vectors(graph, edge) for edge in edges]
    edge_vectors = np.array([np.diff(nodes_coord_vector, axis=0).squeeze().tolist()
                             for nodes_coord_vector in nodes_coord_vectors])
    return edge_vectors


def get_vector_norm(vector):
    """ Get the Euclidean norm of the given vector """

    norm = np.sqrt(np.sum(np.power(vector, 2)))

    return norm


def get_vectors_norms(vectors):
    """ Get the Euclidean norms of the given vectors """

    norms = np.sqrt(np.sum(np.power(vectors, 2), axis=1))

    return norms


def get_difference_vector(first_vector, second_vector):
    """ Get the difference between the two vectors """

    difference_vector = first_vector - second_vector

    return difference_vector


def get_difference_vectors(first_vector, other_vectors):
    """ Get the difference betweem the given vector and the others """

    difference_vectors = other_vectors - first_vector

    return difference_vectors


def get_cosine(first_vector, second_vector):
    """ Get the cosine between the two vectors """

    first_norm = get_vector_norm(first_vector)
    second_norm = get_vector_norm(second_vector)
    numerator = np.dot(first_vector, second_vector)
    denominator = first_norm * second_norm
    cosine = numerator / denominator

    return cosine
    

def get_cosines(first_vector, other_vectors):
    """ Get the cosines between the given vector and the other vectors """
    
    first_norm = get_vector_norm(first_vector)
    other_norms = get_vectors_norms(other_vectors)
    
    numerators = np.dot(other_vectors, first_vector)
    denominators = first_norm * other_norms
    cosines = numerators / denominators
        
    return cosines


def get_edges_cosine(graph, first_edge, second_edge):
    """ Get the cosine of the angle between the two edges """

    first_edge_vector = get_edge_vector(graph, first_edge)
    second_edge_vector = get_edge_vector(graph, second_edge)
    edges_cosine = get_cosine(first_edge_vector, second_edge_vector)

    return edges_cosine


def get_edges_cosines(graph, first_edge, other_edges):
    """ Get the cosines between the given edge and the others """

    first_edge_vector = get_edge_vector(graph, first_edge)
    other_edge_vectors = get_edge_vectors(graph, other_edges)
    edges_cosines = get_cosines(first_vector, other_vectors)
    
    return edges_cosines


def get_edges_weight_difference(graph, first_edge, second_edge):
    """ Get the difference in weight between the two given edges """

    first_weight = graph_utils.get_edge_weight(first_edge)
    second_weight = graph_utils.get_edge_weight(second_edge)
    weight_difference = second_weight - first_weight

    return weight_difference


def get_edges_weight_differences(graph, first_edge, other_edges):
    """ Get the difference in weight between the first edge and the other given edges """

    first_weight = graph_utils.get_edge_weight(first_edge)
    other_weights = graph_utils.get_edges_weights(other_edges)
    weight_differences = np.array(other_weights) - np.array(first_weight)

    return weight_differences


def get_closest_node(graph, node, other_nodes):
    """ Find the closest node to the given node in the other nodes """

    node_coord_vector = get_node_coordinate_vector(graph, node)
    other_coord_vectors = get_nodes_coordinate_vectors(graph, other_nodes)
    difference_vectors = other_coord_vectors - node_coord_vector
    difference_norms = get_vectors_norms(difference_vectors)
    closest_node = other_nodes[np.argmin(difference_norms)]

    return closest_node


def get_closest_station(graph, node):
    """ Find the closest station node to the given node """

    station_nodes = get_stations(graph)
    closest_station = get_closest_node(graph, node, sttaion_nodes)
    
    return closest_node    


def get_reachable_nodes(graph, node, other_nodes, reachability_radius):
    """ Get the other nodes that lie within the given reachability radius """

    node_coord_vector = get_node_coordinate_vector(graph, node)
    other_coord_vectors = get_nodes_coordinate_vectors(graph, other_nodes)
    difference_vectors = other_coord_vectors - node_coord_vector
    difference_norms = get_vectors_norms(difference_vectors)
    reachable_nodes = [node for (node, norm) in zip(other_nodes, difference_norms)
             if norm <= reachability_radius]

    return reachable_nodes


def get_strictly_reachable_nodes(graph, node, other_nodes, reachability_radius):
    """ Get the other nodes that lie within the given reachability radius """

    node_coord_vector = get_node_coordinate_vector(graph, node)
    other_coord_vectors = get_nodes_coordinate_vectors(graph, other_nodes)
    difference_vectors = other_coord_vectors - node_coord_vector
    difference_norms = get_vectors_norms(difference_vectors)
    reachable_nodes = [node for (node, norm) in zip(other_nodes, difference_norms)
                       if norm <= reachability_radius / 2]

    return reachable_nodes


def get_reachable_customers(graph, node, reachability_radius):
    """ Get the customers that reachable for the given node """

    customer_nodes = get_customers(graph)
    reachable_customers = get_reachable_nodes(graph, node, customer_nodes,
                                              reachability_radius)

    return reachable_customers


def get_strictly_reachable_customers(graph, node, reachability_radius):
    """ Get the customers that reachable for the given node """

    customer_nodes = get_customers(graph)
    strictly_reachable_customers = get_strictly_reachable_nodes(
        graph, node, customer_nodes, reachability_radius)

    return strictly_reachable_customers


def get_depot_reachable_customers(graph, reachability_radius):
    """ Get the customers that reachable for the depot nodes """

    depot_nodes, customer_nodes = get_depots(graph), get_customers(graph)
    depot_reachable_customers = []
    for depot_node in depot_nodes:
        depot_reachable_customers += get_strictly_reachable_customers(
            graph, depot_node, reachability_radius)

    return depot_reachable_customers
    

def get_shortest_distance(graph, source_node, sink_node):
    """ Get the shortest path distance from the source to the given sink node """

    shortest_distance = nx.shortest_path_length(graph, source_node, sink_node, "weight")

    return shortest_distance


def get_shortest_distances(graph, source_node, sink_nodes):
    """ Get the shortest path distance from the source to the given sink nodes """

    shortest_distances = nx.shortest_path_length(graph, source_node, weight="weight")
    shortest_distances = [shortest_distances[sink_node] for sink_node in sink_nodes]

    return shortest_distances


def compute_pairwise_distances(graph, nodes):
    """ Compute pairwise distances between the given nodes """

    is_symmetric = graph_utils.is_undirected(graph)
    coordinate_vectors = get_nodes_coordinate_vectors(graph, nodes)
    distances = cdist(coordinate_vectors, coordinate_vectors)

    if is_symmetric:
        return distances[np.triu_indices_from(distances, 1)]
    else:
        return distances.flatten()


def compute_distances_to_node(graph, first_node, other_nodes):
    """ Compute the distances of the given nodes to the first node """

    coordinate_vector = get_node_coordinate_vector(graph, first_node)
    other_coordinate_vectors = get_nodes_coordinate_vectors(graph, other_nodes)
    difference_vectors = get_difference_vectors(coordinate_vector, other_coordinate_vectors)
    difference_norms = get_vectors_norms(difference_vectors)

    return difference_norms
    

def get_inward_incident_edges(graph, node):
    """ Get the inward incident edges for the given node """

    inward_edges = [edge for edge in graph.edges if edge[1] == node]

    return inward_edges


def get_outward_incident_edges(graph, node):
    """ Get the outward incident edges for the given node """

    outward_edges = [edge for edge in graph.edges if edge[0] == node]

    return outward_edges
    

