import numpy as np
import networkx as nx

from optlearn.graph import graph_utils


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
        
    ])

    return arc_features
