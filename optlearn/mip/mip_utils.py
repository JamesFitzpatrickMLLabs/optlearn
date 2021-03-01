import operator

import networkx as nx
import numpy as np

from optlearn import graph_utils


def name_variable(edge, prefix="x"):
    """ Create a variable name """
    
    return "{}_{},{}".format(prefix, *edge)


def get_func(func_dict, func_key):
    """ Get a function from the given dictionary using the key """

    return func_dict[func_key]


def compute_edge_term(weight, variable):
    """ Compute a single edge term in the objective """
    
    return variable * weight


def define_edge_term(variable_dict, graph, edge, prefix="x",
                     perturb=False, max_weight=None):
    """ Define a single edge term in the objective """

    weight = graph_utils.get_edge_weight(graph, *edge)
    name = name_variable(edge, prefix=prefix)
    return compute_edge_term(weight, variable_dict[name])


def define_edge_objective(variable_dict, graph, perturb=False):
    """ Define all terms for the edge objective """

    weights = np.array(graph_utils.get_weights(graph))
    if perturb:
        perturbs = ((weights - weights.mean())/weights.mean()) ** 2
        perturbs  = np.clip(perturbs * 2 -1, -1, 1)
        randoms = np.random.uniform(low=0, high=0.3, size=len(weights))
        perturbs = np.ones_like(randoms)
        weights = weights + weights * perturbs * randoms
    variables = [variable_dict["x_{},{}".format(*edge)] for edge in graph.edges]
    return [var * weight for (var, weight) in zip(variables, weights)]


def get_variable_tuple(string):
    """ Get the (vertex_a, vertex_b) integer tuple from a variable name """

    string = string.split("_")[1]
    string = string.split(",")
    return (int(string[0]), int(string[1]))


def get_variable(variable_dict, edge, prefix="x"):
    """ Get a single variable by specifying its edge and prefix """
    
    key = "{}_{},{}".format(prefix, *edge)
    return variable_dict[key]


def get_variables(variable_dict, vertex, prefix="x"):
    """ Get all variables that have a given vertex  """

    keys = [key for key in variable_dict.keys() if prefix in key]
    keys = [key for key in keys if vertex in get_edge_from_varname(key)]
    return [variable_dict[key] for key in keys]


def get_outward_variables(variable_dict, vertex, prefix="x"):
    """ Get the outward variables for a vertex """

    keys = [key for key in variable_dict.keys() if prefix in key]
    keys = [key for key in keys if get_variable_tuple(key)[0] == vertex]
    return [variable_dict[key] for key in keys]


def get_inward_variables(variable_dict, vertex, prefix="x"):
    """ Get the inward variables for a vertex """
    
    keys = [key for key in variable_dict.keys() if prefix in key]
    keys = [key for key in keys if get_variable_tuple(key)[1] == vertex]
    return [variable_dict[key] for key in keys]


def is_graph(graph, verbose=False):
    """ Check if the graph is a symmetric graph """

    value =  type(graph) == type(nx.Graph())

    if verbose and value:
        print("Graph is symmetric\n")

    return value


def is_digraph(graph, verbose=False):
    """ Check if the graph is an asymmetric graph """

    value = type(graph) == type(nx.DiGraph())

    if verbose and value:
        print("Graph is asymmetric\n")

    return value


def get_edge_from_varname(varname):
    """ Get the edge tuple for a given variable name """
    (a, b) = varname.split(",")
    (a, b) = (a.split("_")[1], b)
        
    return tuple([int(a), int(b)])


def find_vertex_in_edges(edges, vertex):
    """ Return the index of the first edge the vertex is found in """

    for num, edge in enumerate(edges):
        if vertex in edge:
            return num
    return -1

        
def get_other_vertex(edge, vertex):
    """ Get the other vertex from the edge """

    return edge[np.logical_not(edge.index(vertex))]
    

def check_tour(edges, start_vertex):
    """ Starting at a given vertex, check if they all form a single tour """

    index, vertex = None, start_vertex
    while (vertex != start_vertex or index != -1) and len(edges) > 0:
        index = find_vertex_in_edges(edges, vertex)
        vertex = get_other_vertex(edges[index], vertex)
        edges = [item for num, item in enumerate(edges) if num != index]
        if vertex == start_vertex:
            return len(edges) == 0
    return False


def get_edges_from_variables(variables):
    """ Get the associated edges for the given variables """

    names = [item.name for item in variables]
    return [get_edge_from_varname(item) for item in names]


def all_values_integer(values):
    """ Check if all given values are integer """

    return np.all([item.is_integer() for item in values])

_operators = {
    "+": operator.add,
    "-": operator.sub,
    "/": operator.truediv,
    "x": operator.mul,
    "==": operator.eq,
    ">=": operator.ge,
    "<=": operator.le,
    "<": operator.lt,
    ">": operator.gt
    }
