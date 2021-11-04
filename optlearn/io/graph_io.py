import numpy as np
import networkx as nx

from optlearn import graph_utils
from optlearn.io import io_utils


def initialise_graph():
    """ Set up an empty undirected graph """
    
    return nx.Graph()


def initialise_digraph():
    """ Set up an empty directed graph """

    return nx.DiGraph()


def build_attribute_dict(objects, values):
    """ Build a dictionary assigning the given values to the given object (edge/vertex) """

    assert (len(objects) == len(values)), "Value and Edges should have same length!"

    return {object: value for (object, value) in zip (objects, values)}


def remove_edge_attribute(graph, attribute):
    """ Remove the given edge attribute from every edge """

    _ = [graph[a][b].pop(attribute, None) for (a, b) in graph.edges]
    
    return graph


def remove_vertex_attribute(graph, attribute):
    """ Remove the given node attribute from every edge """

    _ = [graph[a].pop(attribute, None) for a in graph.nodes]
    
    return graph


def set_edge_labels(graph, labels):
    """ Set the edge labels for the graph """

    label_dict = build_attribute_dict(graph.edges, labels)
    nx.set_edge_attributes(graph, label_dict, "label")
    
    return graph


def set_vertex_coordinates(graph, coord_dict):
    """ Given the coordinate dictionary, set the coordinates as vertex attributes """

    nx.set_node_attributes(graph, coord_dict, "coord")

    return graph


def write_adjacency(graph, fname):
    """ Write the graph as an adjacency list """

    nx.write_multiline_adjlist(graph, fname)


def read_adjacency(fname):
    """ Read the graph as an adjacency list """

    return nx.read_multiline_adjlist(fname, nodetype=int)


def write_pickle(graph, fname):
    """ Write the graph as a pickle """

    nx.write_gpickle(graph, fname)


def read_pickle(fname):
    """ Read the graph as pickle """

    return nx.read_gpickle(fname)
