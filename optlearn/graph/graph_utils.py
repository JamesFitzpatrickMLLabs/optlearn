import copy

import networkx as nx


def create_graph():
    """ Create an empty graph """

    graph = nx.Graph()

    return graph


def create_digraph():
    """ Create an empty digraph """

    digraph = nx.DiGraph()

    return digraph


def clone_graph(graph):
    """ Clone the given graph """

    clone_graph = copy.deepcopy(graph)

    return clone_graph


def get_all_edges(graph):
    """ Get all of the graph's edges """

    edges = graph.edges

    return graph.edges


def get_edge_weight(graph, edge):
    """ Get the weight associated with the given edge """

    weight = graph[edge[0]][edge[1]].get("weight")

    return weight


def get_edges_weights(graph, edges):
    """ Get the weights associated with the given edges """

    weights = [get_edge_weight(graph, edge) for edge in edges]

    return weights


def get_all_weights(graph):
    """ Get all of the edge weights for a given graph """

    edges = get_all_edges(graph)
    weights = [get_edge_weight(graph, edge) for edge in edges]

    return weights


def store_edge_attribute(graph, edge, attribute, name):
    """ Store the given edge attribute with the given name at the given edge """

    graph[edge[0]][edge[1]][name] = attribute

    return graph


def store_edge_attributes(graph, edges, attributes, name):
    """ Store the given attributes for the given edges with the given name """

    attribute_dict = {edge: attribute for (edge, attribute) in zip(edges, attributes)}
    nx.set_edge_attributes(graph, values=attribute_dict, name=name)

    return graph


def store_node_attribute(graph, node, attribute, name):
    """ Store the given attribute with the given name at the given node """

    graph.nodes[node][name] = attribute

    return graph


def store_node_attributes(graph, nodes, attributes, name):
    """ Store the given node attributes at the given nodes """

    attribute_dict = {node: attribute for (node, attribute) in zip(nodes, attributes)}
    nx.set_node_attributes(graph, values=attribute_dict, name=name)

    return graph


def get_edge_attribute(graph, edge, name):
    """ Get the edge attribute with the given name """

    attribute = graph[edge[0]][edge[1]].get(name)

    return attribute


def get_edge_attributes(graph, edges, name):
    """ Get the edge attributes with the given name """

    attributes = [get_edge_attribute(graph, edge, name) for edge in edges]

    return attributes


def get_node_attribute(graph, node, name):
    """ Get the node attribute with the given name """

    attribute = graph.nodes[node].get(name)

    return attribute


def get_node_attributes(graph, nodes, name):
    """ Get the node attributes with the given name """

    attributes = [get_node_attribute(graph, node, name) for node in nodes]

    return attributes


def is_undirected(graph):
    """ Test if graph is undirected """

    if type(graph) == type(nx.Graph()):
        return True
    else:
        return False


def is_directed(graph):
    """ Test if graph is directed """

    if type(graph) == type(nx.DiGraph()):
        return True
    else:
        return False


def remove_node_from_graph(graph, node):
    """ Remove the node and all of its edges from the graph """

    prune_edges = [edge for edge in get_all_edges(graph) if node in edge]
    graph.remove_edges_from(prune_edges)
    graph.remove_nodes_from([node])
    del(graph.graph["coord_dict"][node])

    return graph


def remove_nodes_from_graph(graph, nodes):
    """ Remove the nodes and all of their edges from the graph """

    for node in nodes:
        graph = remove_node_from_graph(graph, node)
    
    return graph
