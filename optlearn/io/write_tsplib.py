import os

import numpy as np

from optlearn import graph_utils


_problem_types = {
    "tsp": "TSP",
    "vrp": "VRP",
    }

_edge_weight_types = {
    "euclidean": "EUC_2D",
    "explicit": "EXPLICIT",
    }

_edge_weight_formats = {
    "triangular": "UPPER_ROW",
    "full": "FULL_MATRIX",
    }


def write_name(fid, name=None):
    """ Write the name line """

    name = name or "unknown"
    
    fid.write("NAME: {}\n".format(name))


def write_problem_type(fid, problem_type):
    """ Write the problem type line """

    type_string = _problem_types.get(problem_type)

    if type_string is None:
        raise ValueError("Problem type \"{}\" not recognised!".format(problem_type))
    
    fid.write("TYPE: {}\n".format(type_string))

    
def write_dimension(fid, dimension):
    """ Write the dimension line """

    if type(dimension) != int:
        raise ValueError("Only integer dimensions are valid!")

    
    dimension = str(dimension)
    
    fid.write("DIMENSION: {}\n".format(dimension))

    
def write_edge_weight_type(fid, edge_weight_type):
    """ Write the edge weight type line """

    edge_weight_type = _edge_weight_types.get(edge_weight_type)

    if edge_weight_type is None:
        raise ValueError("Edge weight type \"{}\" not recognised!".format(edge_weight_type))
    
    fid.write("EDGE_WEIGHT_TYPE: {}\n".format(edge_weight_type))


def write_edge_weight_format(fid, edge_weight_format):
    """ Write the edge weight format line """

    edge_weight_format = _edge_weight_formats.get(edge_weight_format)

    if edge_weight_format is None:
        raise ValueError("Edge weight format \"{}\" not recognised!".format(edge_weight_format))
    
    fid.write("EDGE_WEIGHT_FORMAT: {}\n".format(edge_weight_format))


def write_node_coord(fid, id, x, y):
    """ Write a node coordinate line """

    fid.write("{} {} {}\n".format(id, x, y))


def write_node_coord_section(fid, coord_dict):
    """ Write the node coords line by line """
    
    fid.write("NODE_COORD_SECTION\n")
    for key in sorted(list(coord_dict.keys())):
        x, y = coord_dict[key]
        write_node_coord(fid, key, x, y)


def write_edges_weights(fid, weights):
    """ Write the edge weights for a given node """

    empty = "{} " * len(weights) 
    string = empty[:-1].format(*weights) + "\n"

    fid.write(string)


def write_edge_weights_full(fid, graph):
    """ Write the edge weights in full matrix form """

    num_nodes = len(graph.nodes)
    num_edges = len(graph.edges)
    
    if num_edges != num_nodes ** 2:
        raise AssertionError("Not all edges are in the graph, cannot write in full!")

    for node in graph.nodes:
        weights = graph_utils.get_edge_weights(graph, node)
        write_edges_weights(fid, weights)


def write_edge_weights_triangular(fid, graph):
    """ Write the edge weights in full matrix form """

    num_nodes = len(graph.nodes)
    num_edges = len(graph.edges)
    min_vertex = np.min(graph.nodes)
    
    if num_edges != num_nodes * (num_nodes - 1) / 2:
        raise AssertionError("Wrong number of edges in the graph, cannot write triangular!")

    for node in graph.nodes:
        weights = graph_utils.get_edge_weights(graph, node)[node - min_vertex:]
        write_edges_weights(fid, weights)


def write_edge_weight_section(fid, graph):
    """ Write the edge weight section """

    fid.write("EDGE_WEIGHT_SECTION\n")

    if graph.is_directed():
        write_edge_weights_full(fid, graph)
    else:
        write_edge_weights_triangular(fid, graph)
    

def write_eof(fid):
    """ Write the end of file line """

    fid.write("EOF")


def write_tsp_preamble(fid, graph, name=None, explicit=False):
    """ Write the preamble for a given graph """

    dimension = len(graph.nodes)
    problem_type = "tsp"
    
    if graph.is_directed():
        weight_format = "full"
    else:
        weight_format = "triangular"
    if not explicit:
        weight_type = "euclidean"
    else:
        weight_type = "explicit"
    
    write_name(fid, name)
    write_problem_type(fid, problem_type)
    write_dimension(fid, dimension)
    write_edge_weight_type(fid, weight_type)
    if explicit:
        write_edge_weight_format(fid, weight_format)
    

def write_tsp_explicit(fname, graph, name=None):
    """ Write a graph with weights given explicitly """

    with open(fname, "w") as fid:
        write_tsp_preamble(fid, graph, name=name, explicit=True)
        write_edge_weight_section(fid, graph)
        write_eof(fid)

    
