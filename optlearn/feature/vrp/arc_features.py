import numpy as np
import networkx as nx

from optlearn.graph import graph_utils
from optlearn.solve import evrpnl_duplication

from optlearn.feature.vrp import feature_utils 
from optlearn.feature.vrp import node_features 

from optlearn.heuristic import evrpnl_heuristic


def get_customer_customer_arcs(graph):
    """ Get the customer-customer arcs of a graph """
    
    customers = feature_utils.get_customers(graph)
    customer_customer_arcs = [
        edge for edge in feature_utils.get_all_edges(graph)
        if edge[0] in customers and edge[1] in customers
    ]

    return customer_customer_arcs


def store_edge_features(graph, features, name, edges=None):
    """ Store the given edge features with the given name """

    edges = edge or graph.edges
    graph = graph_utils.store_edge_attributes(graph, edges, features, name)

    return graph


def get_edge_features(graph, name, edges=None):
    """ Get the given feature from the given edges """

    edges = edges or graph.edges
    features = graph_utils.get_edge_attributes(graph, edges, name)

    return features

    
def compute_normalised_edge_weight(graph, edge, reachability_radius):
    """ Compute the normalised distance feature for each edge """

    weight = np.array(graph_utils.get_edge_weight(graph, edge))
    normalised_weight = weight / reachability_radius

    return normalised_weight


def get_maximum_left_inward_incident_weight_ratio(graph, edge):
    """ Compare the weight of the given edge to all those incident with the left node """

    left_edges = [item for item in graph.edges if item[1]==edge[0]]
    left_edge_weights = graph_utils.get_edges_weights(graph, left_edges)
    weight_comparison = graph_utils.get_edge_weight(graph, edge) / np.max(left_edge_weights)

    return weight_comparison


def get_minimum_left_inward_incident_weight_ratio(graph, edge):
    """ Compare the weight of the given edge to all those incident with the left node """

    left_edges = [item for item in graph.edges if item[1]==edge[0]]
    left_edge_weights = graph_utils.get_edges_weights(graph, left_edges)
    weight_comparison = np.min(left_edge_weights) / graph_utils.get_edge_weight(graph, edge) 

    return weight_comparison


def get_normalised_left_inward_incident_weight_std(graph, edge, reachability_radius):
    """ Compare the weight of the given edge to all those incident with the left node """

    left_edges = [item for item in graph.edges if item[1]==edge[0]]
    left_edge_weights = graph_utils.get_edges_weights(graph, left_edges)
    normalised_std = np.std(np.array(left_edge_weights) / reachability_radius) 

    return normalised_std


def get_maximum_left_outward_incident_weight_ratio(graph, edge):
    """ Compare the weight of the given edge to all those incident with the left node """

    left_edges = [item for item in graph.edges if item[0]==edge[0]]
    left_edge_weights = graph_utils.get_edges_weights(graph, left_edges)
    weight_comparison = graph_utils.get_edge_weight(graph, edge) / np.max(left_edge_weights)

    return weight_comparison


def get_minimum_left_outward_incident_weight_ratio(graph, edge):
    """ Compare the weight of the given edge to all those incident with the left node """

    left_edges = [item for item in graph.edges if item[0]==edge[0]]
    left_edge_weights = graph_utils.get_edges_weights(graph, left_edges)
    weight_comparison = np.min(left_edge_weights) / graph_utils.get_edge_weight(graph, edge) 

    return weight_comparison


def get_normalised_left_outward_incident_weight_std(graph, edge, reachability_radius):
    """ Compare the weight of the given edge to all those incident with the left node """

    left_edges = [item for item in graph.edges if item[0]==edge[0]]
    left_edge_weights = graph_utils.get_edges_weights(graph, left_edges)
    normalised_std = np.std(np.array(left_edge_weights) / reachability_radius) 

    return normalised_std


def get_maximum_right_inward_incident_weight_ratio(graph, edge):
    """ Compare the weight of the given edge to all those incident with the right node """

    right_edges = [item for item in graph.edges if item[1]==edge[1]]
    right_edge_weights = graph_utils.get_edges_weights(graph, right_edges)
    weight_comparison = graph_utils.get_edge_weight(graph, edge) / np.max(right_edge_weights)

    return weight_comparison


def get_minimum_right_inward_incident_weight_ratio(graph, edge):
    """ Compare the weight of the given edge to all those incident with the right node """

    right_edges = [item for item in graph.edges if item[1]==edge[1]]
    right_edge_weights = graph_utils.get_edges_weights(graph, right_edges)
    weight_comparison = np.min(right_edge_weights) / graph_utils.get_edge_weight(graph, edge) 

    return weight_comparison


def get_normalised_right_inward_incident_weight_std(graph, edge, reachability_radius):
    """ Compare the weight of the given edge to all those incident with the right node """

    right_edges = [item for item in graph.edges if item[1]==edge[1]]
    right_edge_weights = graph_utils.get_edges_weights(graph, left_edges)
    normalised_std = np.std(np.array(right_edge_weights) / reachability_radius) 

    return normalised_std


def get_maximum_right_outward_incident_weight_ratio(graph, edge):
    """ Compare the weight of the given edge to all those incident with the right node """

    right_edges = [item for item in graph.edges if item[0]==edge[1]]
    right_edge_weights = graph_utils.get_edges_weights(graph, right_edges)
    weight_comparison = graph_utils.get_edge_weight(graph, edge) / np.max(right_edge_weights)

    return weight_comparison


def get_minimum_right_inward_incident_weight_ratio(graph, edge):
    """ Compare the weight of the given edge to all those incident with the right node """

    right_edges = [item for item in graph.edges if item[0]==edge[1]]
    right_edge_weights = graph_utils.get_edges_weights(graph, right_edges)
    weight_comparison = np.min(right_edge_weights) / graph_utils.get_edge_weight(graph, edge) 

    return weight_comparison


def get_normalised_right_inward_incident_weight_std(graph, edge, reachability_radius):
    """ Compare the weight of the given edge to all those incident with the right node """

    right_edges = [item for item in graph.edges if item[0]==edge[1]]
    right_edge_weights = graph_utils.get_edges_weights(graph, right_edges)
    normalised_std = np.std(np.array(right_edge_weights) / reachability_radius) 

    return normalised_std


def get_maximum_left_outward_incident_weight_ratio(graph, edge):
    """ Compare the weight of the given edge to all those incident with the right node """

    right_edges = [item for item in graph.edges if item[0]==edge[1]]
    right_edge_weights = graph_utils.get_edges_weights(graph, right_edges)
    weight_comparison = graph_utils.get_edge_weight(graph, edge) / np.max(right_edge_weights)

    return weight_comparison


def get_minimum_right_outward_incident_weight_ratio(graph, edge):
    """ Compare the weight of the given edge to all those incident with the right node """

    right_edges = [item for item in graph.edges if item[0]==edge[1]]
    right_edge_weights = graph_utils.get_edges_weights(graph, right_edges)
    weight_comparison = np.min(right_edge_weights) / graph_utils.get_edge_weight(graph, edge) 

    return weight_comparison


def get_normalised_right_outward_incident_weight_std(graph, edge, reachability_radius):
    """ Compare the weight of the given edge to all those incident with the right node """

    right_edges = [item for item in graph.edges if item[0]==edge[1]]
    right_edge_weights = graph_utils.get_edges_weights(graph, right_edges)
    normalised_std = np.std(np.array(right_edge_weights) / reachability_radius) 

    return normalised_std


def get_normalised_left_vertex_distance_to_depot(graph, edge, reachability_radius):
    """ Get the distance from the left vertex to the depot and normalise it """

    distance = node_features.get_depot_distance(graph, edge[0])
    normalised_distance = distance / reachability_radius
    if normalised_distance > 1:
        normalised_distance = -1
        
    return normalised_distance


def get_normalised_right_vertex_distance_to_depot(graph, edge, reachability_radius):
    """ Get the distance from the right vertex to the depot and normalise it """

    distance = node_features.get_depot_distance(graph, edge[1])
    normalised_distance = distance / reachability_radius
    if normalised_distance > 1:
        normalised_distance = -1
        
    return normalised_distance


def get_normalised_left_closest_station_distance(graph, edge, reachability_radius):
    """ Get the distance to the closest station to the left vertex and normalise it """

    distance = node_features.get_normalised_closest_station_distance(
        graph, edge[0], reachability_radius)

    return distance


def get_normalised_right_closest_station_distance(graph, edge, reachability_radius):
    """ Get the distance to the closest station to the left vertex and normalise it """

    distance = node_features.get_normalised_closest_station_distance(
        graph, edge[1], reachability_radius)

    return distance


def get_left_closest_station_technology(graph, edge, reachability_radius):
    """ Get the technology of the closest station to the left vertex """

    closest_station = node_features.get_closest_station(graph, edge[0])
    station_technology = node_features.compute_station_technology(graph, closest_station)

    return station_technology


def get_right_closest_station_technology(graph, edge, reachability_radius):
    """ Get the technology of the closest station to the right vertex """

    closest_station = node_features.get_closest_station(graph, edge[1])
    station_technology = node_features.compute_station_technology(graph, closest_station)

    return station_technology


def are_left_and_right_closest_stations_the_same(graph, edge):
    """ Check if the left and right closest stations are the same station """

    left_closest_station = node_features.get_closest_station(graph, edge[0])
    right_closest_station = node_features.get_closest_station(graph, edge[1])
    are_closest_stations_the_same = int(left_closest_station == right_closest_station)

    return are_closest_stations_the_same


def get_left_depot_cosine(graph, edge):
    """ Get the left depot cosine for the given edge """

    left_depot_cosine = node_features.get_depot_cosine(graph, edge[0])

    return left_depot_cosine


def get_right_depot_cosine(graph, edge):
    """ Get the right depot cosine for the given edge """

    right_depot_cosine = node_features.get_depot_cosine(graph, edge[1])

    return right_depot_cosine


def get_left_closest_station_depot_cosine(graph, edge):
    """ Get the depot cosine of the station closest to the left vertex """

    left_closest_station = node_features.get_closest_station(graph, edge[0])
    left_depot_cosine = node_features.get_depot_cosine(graph, edge[0])

    return left_depot_cosine


def get_right_closest_station_depot_cosine(graph, edge):
    """ Get the depot cosine of the station closest to the right vertex """

    right_closest_station = node_features.get_closest_station(graph, edge[1])
    right_depot_cosine = node_features.get_depot_cosine(graph, edge[1])

    return right_depot_cosine


def get_strict_edge_betweenness_centrality(graph, reachability_radius):
    """ Get the eigenvector centrality with only strictly reachable edges """

    clone_graph = graph_utils.clone_graph(graph)
    clone_graph = node_features.remove_unreachable_edges(
        clone_graph, reachability_radius)
    clone_graph = node_features.remove_station_strictly_unreachable_edges(
        clone_graph, reachability_radius)
    betweenness_centralities = nx.edge_betweenness_centrality(clone_graph, weight="weight")

    return betweenness_centralities


def get_minimum_arborescence_values(graph):
    """ Get the minimum arboresecence values """

    arboresecence_graph = nx.minimum_spanning_arborescence(graph)
    arboresecence_values = nx.get_edge_attributes(arboresecence_graph, "weight")

    return arboresecence_values


def get_heuristic_edge_frequency(graph):
    """ Use the heuristic to generate routes, see how often each edge appears """

    heuristic_solver = evrpnl_heuristic.heuristicSolver(graph, relabel=True)
    heuristic_solver.set_temporary_tsp_filename("/home/james/temp.tsp")
    _ = heuristic_solver.solve_problem()
    edge_frequency_dict = heuristic_solver.count_edges_occurrences()

    return edge_frequency_dict
    

def get_evrpnl_problem_relaxation_solution(graph, reachability_radius):
    """ Get the evrpnl problem relexation solution """

    customers = feature_utils.get_customers(graph)
    solver = evrpnl_duplication.duplicationSolver(
        graph, iteration_limit=len(customers), time_limit=300)
    solver.lpsolve_problem(reachability_radius)
    solution = solver.get_primed_travel_solution_dict()

    return solution


def get_evrpnl_problem_relaxation_solution(graph, reachability_radius):
    """ Get the evrpnl problem relexation solution """

    customers = feature_utils.get_customers(graph)
    solver = evrpnl_duplication.duplicationSolver(
        graph, iteration_limit=len(customers), time_limit=300)
    solver.lpsolve_problem(reachability_radius)
    solution = solver.get_primed_travel_solution_dict()

    return solution

    
def get_evrpnl_problem_relaxation_reduced_costs(graph):
    """ Get the evrpnl problem relexation solution """

    return None


def get_arc_type(graph, arc):
    """ Get the type of the given arc """

    depots = feature_utils.get_depots(graph)
    stations = feature_utils.get_stations(graph)
    customers = feature_utils.get_customers(graph)

    if arc[0] in depots:
        if arc[1] in depots:
            return [-1, -1]
        if arc[1] in stations:
            return [-1, 0]
        if arc[1] in customers:
            return [-1, 1]
    if arc[0] in stations:
        if arc[1] in depots:
            return [0, -1]
        if arc[1] in stations:
            return [0, 0]
        if arc[1] in customers:
            return [0, 1]
    if arc[0] in customers:
        if arc[1] in depots:
            return [1, -1]
        if arc[1] in stations:
            return [1, 0]
        if arc[1] in customers:
            return [1, 1]


def get_arc_types(graph, arcs=None):
    """ Get the types of all the arcs """

    all_arcs = arcs or graph_utils.get_all_edges(graph)
    arc_types = {arc: get_arc_type(graph, arc) for arc in all_arcs}

    return arc_types

    
def compute_arc_features(graph, edge, reachability_radius):
    """ Compute the arc features for the given arc """


    arc_features = np.array([
        compute_normalised_edge_weight(graph, edge, reachability_radius),
        get_maximum_left_inward_incident_weight_ratio(graph, edge),
        get_minimum_left_inward_incident_weight_ratio(graph, edge),
        get_maximum_left_outward_incident_weight_ratio(graph, edge),
        get_minimum_left_outward_incident_weight_ratio(graph, edge),
        get_normalised_left_inward_incident_weight_std(graph, edge, reachability_radius),
        get_normalised_left_outward_incident_weight_std(graph, edge, reachability_radius),
        get_maximum_right_inward_incident_weight_ratio(graph, edge),
        get_minimum_right_inward_incident_weight_ratio(graph, edge),
        get_maximum_right_outward_incident_weight_ratio(graph, edge),
        get_minimum_right_outward_incident_weight_ratio(graph, edge),
        get_normalised_right_inward_incident_weight_std(graph, edge, reachability_radius),
        get_normalised_right_outward_incident_weight_std(graph, edge, reachability_radius),
        get_normalised_left_vertex_distance_to_depot(graph, edge, reachability_radius),
        get_normalised_right_vertex_distance_to_depot(graph, edge, reachability_radius),
        get_normalised_left_closest_station_distance(graph, edge, reachability_radius),
        get_normalised_right_closest_station_distance(graph, edge, reachability_radius),
        get_left_closest_station_technology(graph, edge, reachability_radius),
        get_right_closest_station_technology(graph, edge, reachability_radius),
        are_left_and_right_closest_stations_the_same(graph, edge),
        get_left_depot_cosine(graph, edge),
        get_right_depot_cosine(graph, edge),
        get_left_closest_station_depot_cosine(graph, edge),
        get_right_closest_station_depot_cosine(graph, edge),        
    ])

    return arc_features


arc_functions = [
    compute_normalised_edge_weight,
    get_maximum_left_inward_incident_weight_ratio,
    get_minimum_left_inward_incident_weight_ratio,
    get_maximum_left_outward_incident_weight_ratio,
    get_minimum_left_outward_incident_weight_ratio,
    get_normalised_left_inward_incident_weight_std,
    get_normalised_left_outward_incident_weight_std,
    get_maximum_right_inward_incident_weight_ratio,
    get_minimum_right_inward_incident_weight_ratio,
    get_maximum_right_outward_incident_weight_ratio,
    get_minimum_right_outward_incident_weight_ratio,
    get_normalised_right_inward_incident_weight_std,
    get_normalised_right_outward_incident_weight_std,
    get_normalised_left_vertex_distance_to_depot,
    get_normalised_right_vertex_distance_to_depot,
    get_normalised_left_closest_station_distance,
    get_normalised_right_closest_station_distance,
    get_left_closest_station_technology,
    get_right_closest_station_technology,
    are_left_and_right_closest_stations_the_same,
    get_left_depot_cosine,
    get_right_depot_cosine,
    get_left_closest_station_depot_cosine,
    get_right_closest_station_depot_cosine,
]

graph_functions = [
    get_strict_edge_betweenness_centrality,
    get_minimum_arborescence_values,
    get_evrpnl_problem_relaxation_solution,
    # get_heuristic_edge_frequency,
] 
