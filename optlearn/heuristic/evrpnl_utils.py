from optlearn.graph import graph_utils
from optlearn.feature.vrp import feature_utils


def remove_stations_from_graph(graph):
    """ Remove the stations from the graph """

    stations = feature_utils.get_stations(graph)
    graph = graph_utils.remove_nodes_from_graph(graph, stations)

    return graph


def remove_depots_from_graph(graph):
    """ Remove the depots from the graph """

    depots = feature_utils.get_depots(graph)
    graph = graph_utils.remove_nodes_from_graph(graph, depots)

    return graph


def remove_depots_and_stations_from_graph(graph):
    """ Remove the depots and stations from the given graph """

    graph = remove_depots_from_graph(graph)
    graph = remove_stations_from_graph(graph)

    return graph
