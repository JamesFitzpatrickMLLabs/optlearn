import networkx as nx

from optlearn.io.vrp import vrp_utils
from optlearn.io.tsp import tsp_utils
from optlearn.io.npy import npy_utils
from optlearn.io.graph import graph_utils


def floor_node_indices_graph(graph):
    """ Make sure the node indices are integers starting at zero """

    node_labels = list(graph.nodes)
    zero_indices = list(range(len(node_labels)))
    node_mapping = {old: new for (new, old) in zip(zero_indices, node_labels)}
    graph = nx.relabel_nodes(graph, node_mapping)

    return graph


def floor_node_indices_array(array):
    """ Make sure the node indices are integers starting at zero """

    node_labels = np.unique(array)
    zero_indices = list(range(len(node_labels)))
    zero_array = np.zeros_like(node_labels)
    for old, new in zip(zero_indices, node_labels):
        zero_array[node_labels==old] = new

    return zero_array


def read_and_preprocess_graph(filename, filereader):
    """ Read the given graph and do some basic processing on it """

    graph = filereader(filename)
    graph = floor_node_indices_graph(graph)

    return graph


def read_and_preprocess_array(filename, filereader):
    """ Read the given array and do some basic processing on it """

    array = filereader(filename)
    array = floor_node_indices_array(array)

    return array


def read_tsp_problem_from_tsplib(filename):
    """ Read the given TSP problem from the file """

    filereader = tsp_utils.read_tsp_problem_from_tsplib_file
    graph = read_and_preprocess_graph(filename, filereader)
    
    return graph


def read_tsp_solution_from_tsplib(filename):
    """ Read the given TSP problem from the file """

    filereader = tsp_utils.read_tsp_problem_from_tsplib_file
    array = read_and_preprocess_array(filename, filereader)
    
    return array


def read_cvrp_problem_from_xml(filename):
    """ Read the given CVRP problem from the file """

    filereader = vrp_utils.read_cvrp_problem
    graph = read_and_preprocess_graph(filename, filereader)
    
    return graph

    
def read_gvrp_problem_from_txt(filename):
    """ Read the given GVRP problem from the file """

    filereader = vrp_utils.read_gvrp_problem
    graph = read_and_preprocess_graph(filename, filereader)
    
    return graph


def read_evrpnl_problem_from_xml(filename):
    """ Read the given EVRP-NL problem from the file """

    filereader = vrp_utils.read_evrpnl_problem
    graph = read_and_preprocess_graph(filename, filereader)
    
    return graph
