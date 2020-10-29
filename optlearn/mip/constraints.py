import networkx as nx

from optlearn.mip import mip_utils


def initialise_mincut_graph(is_symmetric=True):
    """ Set the mincut graph """
        
    if is_symmetric:
        graph_type = nx.Graph
    else:
        graph_type = nx.DiGraph
    return graph_type()


def set_mincut_graph(graph, edges, values):
    """ Build a graph structure for mincut """

    edge_bunch = [(*edge, {"capacity": value}) for edge, value in zip(edges, values)]
    graph.add_edges_from(edge_bunch)
    return graph


def get_mincut_variables(graph, vertex_a, vertex_b, variable_dict, is_symmetric=True):
    """ Get the variables needed to build a mincut constraint for a specific vertex pair """

    cut_value, (set_a, set_b) = nx.minimum_cut(graph, vertex_a, vertex_b)
    
    if cut_value <= 0.99 + is_symmetric * 1:
        
        if len(set_a) <= len(set_b):
            set_c = set_a
        else:
            set_c = set_b
        
        keys = variable_dict.keys()
        keys = [key for key in keys if (mip_utils.get_variable_tuple(key)[0] in set_c and
                                        mip_utils.get_variable_tuple(key)[1] in set_c)]
        return [variable_dict[key] for key in keys]
    return None
